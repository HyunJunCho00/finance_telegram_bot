"""Hybrid AI router with backend-aware fallback and circuit breaking."""

import base64
import json
import mimetypes
import re
import threading
import time
from typing import Dict, List, Optional, Tuple

from google import genai
from google.genai import types
from loguru import logger
from openai import OpenAI
from requests import exceptions as requests_exceptions

from config.settings import settings


class CloudflareGenerator:
    """Cloudflare Workers AI helper for triage/rerank style requests."""

    def __init__(self, model: Optional[str] = None):
        self._model = model
        self._account_id = ""
        self._api_key = ""
        self._enabled = False
        self._session = None
        self._last_failure_kind = ""

    def _init(self):
        if self._enabled:
            return
        try:
            self._account_id = getattr(settings, "CLOUDFLARE_ACCOUNT_ID", "")
            self._api_key = getattr(settings, "CLOUDFLARE_AI_API_KEY", "")
            self._enabled = bool(self._account_id and self._api_key)
        except Exception:
            pass

    def _url(self, model: str) -> str:
        return (
            f"https://api.cloudflare.com/client/v4/accounts/{self._account_id}"
            f"/ai/run/{model}"
        )

    @staticmethod
    def _extract_text(payload: object) -> str:
        if payload is None:
            return ""
        if isinstance(payload, str):
            return payload.strip()
        if isinstance(payload, list):
            parts: List[str] = []
            for item in payload:
                text = CloudflareGenerator._extract_text(item)
                if text:
                    parts.append(text)
            return "\n".join(parts).strip()
        if not isinstance(payload, dict):
            return ""

        if "choices" in payload and isinstance(payload.get("choices"), list):
            for choice in payload.get("choices", []):
                text = CloudflareGenerator._extract_text(choice)
                if text:
                    return text

        if "message" in payload and isinstance(payload.get("message"), dict):
            message = payload.get("message") or {}
            for key in ("content", "reasoning_content", "reasoning", "output_text", "text"):
                text = CloudflareGenerator._extract_text(message.get(key))
                if text:
                    return text

        if "delta" in payload and isinstance(payload.get("delta"), dict):
            delta = payload.get("delta") or {}
            for key in ("content", "reasoning_content", "reasoning", "text"):
                text = CloudflareGenerator._extract_text(delta.get(key))
                if text:
                    return text

        result = payload.get("result")
        if isinstance(result, str) and result.strip():
            return result.strip()
        if isinstance(result, dict):
            for key in ("response", "output_text", "text", "content", "generated_text", "output"):
                text = CloudflareGenerator._extract_text(result.get(key))
                if text:
                    return text
            choices = result.get("choices")
            if isinstance(choices, list):
                for choice in choices:
                    text = CloudflareGenerator._extract_text(choice)
                    if text:
                        return text
            messages = result.get("messages")
            if isinstance(messages, list):
                for message in messages:
                    text = CloudflareGenerator._extract_text(message)
                    if text:
                        return text

        for key in ("response", "output_text", "text", "content", "generated_text", "output"):
            text = CloudflareGenerator._extract_text(payload.get(key))
            if text:
                return text

        message = payload.get("message")
        if isinstance(message, dict):
            text = CloudflareGenerator._extract_text(message.get("content"))
            if text:
                return text
        if isinstance(message, str) and message.strip():
            return message.strip()

        return ""

    def generate(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: int = 1000,
        model: Optional[str] = None,
    ) -> Optional[str]:
        self._init()
        if not self._enabled:
            return None
        self._last_failure_kind = ""

        import requests

        model_id = model or self._model or settings.MODEL_CF_TRIAGE
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "max_tokens": max_tokens,
        }

        try:
            if self._session is None:
                self._session = requests.Session()
                self._session.headers.update(
                    {
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    }
                )
            timeout_seconds = max(5.0, float(getattr(settings, "CLOUDFLARE_AI_TIMEOUT_SECONDS", 25.0)))
            resp = self._session.post(self._url(model_id), json=payload, timeout=timeout_seconds)
            resp.raise_for_status()
            body = resp.json()
            text = self._extract_text(body)
            if text:
                return text
            self._last_failure_kind = "empty_payload"
            body_keys = list(body.keys()) if isinstance(body, dict) else [type(body).__name__]
            preview = str(body)
            if len(preview) > 500:
                preview = preview[:500] + "...[truncated]"
            logger.warning(
                f"CloudflareGenerator empty/unknown payload ({model_id}). "
                f"keys={body_keys} body={preview}"
            )
            return None
        except requests_exceptions.ReadTimeout:
            self._last_failure_kind = "timeout"
            logger.warning(f"CloudflareGenerator read timeout ({model_id})")
            return None
        except requests_exceptions.HTTPError as e:
            status_code = getattr(getattr(e, "response", None), "status_code", None)
            self._last_failure_kind = "rate_limit" if status_code == 429 else "http_error"
            response_text = ""
            try:
                response_text = getattr(e.response, "text", "") or ""
            except Exception:
                response_text = ""
            if len(response_text) > 500:
                response_text = response_text[:500] + "...[truncated]"
            logger.warning(
                f"CloudflareGenerator HTTP error ({model_id}) "
                f"status={status_code} body={response_text}"
            )
            return None
        except Exception as e:
            self._last_failure_kind = "request_error"
            err_preview = str(e)
            if len(err_preview) > 300:
                err_preview = err_preview[:300] + "...[truncated]"
            logger.warning(f"CloudflareGenerator failed ({model_id}): {err_preview}")
            return None

    @property
    def last_failure_kind(self) -> str:
        return self._last_failure_kind


class AIClient:
    def __init__(self):
        judge_key = getattr(settings, "GEMINI_API_KEY_JUDGE", "")
        vlm_key = getattr(settings, "GEMINI_API_KEY_VLM", "")
        default_key = getattr(settings, "GEMINI_API_KEY", "") or judge_key or vlm_key
        can_use_vertex = bool(getattr(settings, "PROJECT_ID", ""))
        vertex_location = getattr(settings, "VERTEX_REGION_GEMINI", "") or settings.vertex_region or "global"

        def _make_gemini(key: str):
            if key:
                return genai.Client(api_key=key)
            if can_use_vertex:
                return genai.Client(
                    vertexai=True,
                    project=settings.PROJECT_ID,
                    location=vertex_location,
                )
            return None

        self._gemini_default = _make_gemini(default_key)
        self._gemini_judge = _make_gemini(judge_key) or self._gemini_default
        self._gemini_vlm = _make_gemini(vlm_key) or self._gemini_default

        if self._gemini_default:
            backend = "AI Studio key" if default_key else "Vertex AI"
            logger.info(f"Gemini clients initialized (default / judge / vlm) via {backend}")
        else:
            logger.warning("Gemini disabled: no API key and no PROJECT_ID for Vertex AI")

        self._claude_client = None
        self._cerebras_client = None
        self._groq_client = None
        self._openrouter_client = None
        self._cf_generator = CloudflareGenerator()

        self.default_model_id = "gemini-3-flash-preview"
        self.premium_model_id = settings.MODEL_JUDGE

        # Per-backend locks: gemini/cerebras/groq/cf 각각 독립적으로 rate-limit
        # 기존 _global_api_lock은 모든 백엔드를 직렬화해서 judge+meta_regime 동시 호출 시
        # 실제 소요 시간이 2배가 됐음 → 백엔드별 락으로 분리해 크로스 백엔드 병렬화 허용
        _backends = ["gemini", "cerebras", "groq", "cf"]
        self._backend_locks: dict[str, threading.Lock] = {b: threading.Lock() for b in _backends}
        self._last_call_by_backend: dict[str, float] = {b: 0.0 for b in _backends}
        self._MIN_GAP = 0.5

        self._GROQ_REASONING_POOL = [
            "openai/gpt-oss-120b",
            "llama-3.3-70b-versatile",
            "qwen/qwen3-32b",           # RPM 60으로 높지만 TPM 6k — reasoning 보조
            "meta-llama/llama-4-scout-17b-16e-instruct",
        ]
        self._GROQ_GENERAL_POOL = [
            "meta-llama/llama-4-scout-17b-16e-instruct",  # TPM 30k — 긴 텍스트 relay 1순위
            "qwen/qwen3-32b",                              # RPM 60 — 단기 버스트 대응
            "llama-3.3-70b-versatile",                    # TPM 12k — 품질 fallback
            "openai/gpt-oss-20b",
            "llama-3.1-8b-instant",                       # RPD 14,400 — 마지막 보루
        ]

        self._circuit_breakers = {}

        # Gemini context cache registry — keyed by md5(model_id + system_prompt)
        # Value: (cache_name: str, created_ts: float)
        # Avoids re-tokenising the Judge's ~1200-token static system prompt on every call
        self._gemini_system_caches: Dict[str, Tuple[str, float]] = {}
        self._gemini_cache_lock = threading.Lock()

    @property
    def claude_client(self):
        return None

    @property
    def cerebras_client(self):
        if self._cerebras_client is None:
            key = getattr(settings, "CEREBRAS_API_KEY", "")
            if not key:
                logger.warning("CEREBRAS_API_KEY not set")
                return None
            self._cerebras_client = OpenAI(
                base_url="https://api.cerebras.ai/v1",
                api_key=key,
            )
        return self._cerebras_client

    @property
    def groq_client(self):
        if self._groq_client is None and settings.GROQ_API_KEY:
            self._groq_client = OpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=settings.GROQ_API_KEY,
            )
        return self._groq_client

    @property
    def openrouter_client(self):
        if self._openrouter_client is None:
            key = getattr(settings, "OPENROUTER_API_KEY", "")
            if not key:
                logger.warning("OPENROUTER_API_KEY not set; openrouter route disabled")
                return None
            self._openrouter_client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=key,
            )
        return self._openrouter_client

    def _is_model_available(self, model_id: str) -> bool:
        expiry = self._circuit_breakers.get(model_id, 0)
        return time.time() > expiry

    def _mark_model_exhausted(self, model_id: str, duration: int = 3600):
        logger.warning(f"CIRCUIT BREAKER: Blocking [{model_id}] for {duration}s due to temporary failure/quota")
        self._circuit_breakers[model_id] = time.time() + duration

    def _get_route(self, role: str) -> Tuple[str, str, int]:
        role = (role or "general").lower()
        role_map = {
            "judge": ("gemini_judge", settings.MODEL_JUDGE, settings.MAX_INPUT_CHARS_JUDGE),
            "self_correction": ("gemini_judge", settings.MODEL_SELF_CORRECTION, settings.MAX_INPUT_CHARS_SELF_CORRECTION),
            "vlm_geometric": ("gemini_vlm", settings.MODEL_VLM_GEOMETRIC, getattr(settings, "MAX_INPUT_CHARS_VLM_GEOMETRIC", 15000)),
            "vlm_analysis": ("gemini_vlm", settings.MODEL_VLM_GEOMETRIC, 15000),
            "vlm_telegram_chart": ("gemini_vlm", settings.MODEL_VLM_TELEGRAM_CHART, settings.MAX_INPUT_CHARS_VLM_TELEGRAM_CHART),
            "rag_vision": ("gemini_vlm", settings.MODEL_VLM_GEOMETRIC, 15000),
            "meta_regime": ("cerebras", settings.MODEL_META_REGIME, 15000),
            "macro": ("cerebras", settings.MODEL_META_REGIME, settings.MAX_INPUT_CHARS_MACRO),
            "risk_eval": ("cerebras", settings.MODEL_RISK_EVAL, 10000),
            "risk_eval_fallback": ("groq", settings.MODEL_RISK_EVAL_FALLBACK, 10000),
            "rag_extraction": ("groq", settings.MODEL_RAG_EXTRACTION, settings.MAX_INPUT_CHARS_RAG_EXTRACTION),
            "news_summarize": ("groq", settings.MODEL_NEWS_SUMMARIZE, 10000),
            "trigger_veto": ("groq", settings.MODEL_TRIGGER_VETO, settings.MAX_INPUT_CHARS_TRIGGER_VETO),
            "news_cluster": ("groq", settings.MODEL_NEWS_CLUSTER, 12000),
            "news_brief_final": ("groq", settings.MODEL_NEWS_FINAL, 12000),
            "post_mortem": ("groq", settings.MODEL_RAG_EXTRACTION, 15000),
            "feedback": ("groq", settings.MODEL_RAG_EXTRACTION, 15000),
            "monitor_hourly": ("cerebras", settings.MODEL_MONITOR_HOURLY, 8000),
            "cloudflare_triage": ("cf", settings.MODEL_CF_TRIAGE, 5000),
            "cloudflare_rerank": ("cf", settings.MODEL_CF_RERANK, 5000),
            "triage": ("cf", settings.MODEL_CF_TRIAGE, 5000),
            "claude_standby": ("gemini_judge", settings.MODEL_JUDGE, 20000),
            "chat": ("gemini_default", settings.MODEL_CHAT, 10000),
        }
        if role in role_map:
            return role_map[role]
        if self.groq_client:
            return ("groq", "llama-3.3-70b-versatile", 15000)
        return ("gemini_default", self.default_model_id, 15000)

    def _trim_input(self, text: str, max_chars: int) -> str:
        if not text or len(text) <= max_chars:
            return text or ""
        return text[:max_chars] + f"\n\n[TRUNCATED {len(text) - max_chars} chars]"

    def _is_critical_role(self, role: str) -> bool:
        # meta_regime은 critical에서 제외: regime 분류 실패 시 Groq까지만 fallback하고
        # RANGE_BOUND 기본값으로 degrade. Gemini Pro RPD 낭비 방지.
        return role in ["judge", "risk_eval", "self_correction"]

    def _role_importance_tier(self, role: str) -> str:
        role = (role or "").lower()
        if role in {"judge", "self_correction"}:
            return "decision"
        if role in {"meta_regime", "macro", "risk_eval", "risk_eval_fallback", "trigger_veto", "monitor_hourly"}:
            return "strategy"
        if role in {"vlm_geometric", "vlm_analysis", "vlm_telegram_chart", "rag_vision"}:
            return "vision"
        return "light"

    def _parse_model_pool(self, raw_value: str) -> List[str]:
        return [item.strip() for item in str(raw_value or "").split(",") if item.strip()]

    def _cloudflare_pool_for_role(self, role: str) -> List[str]:
        tier = self._role_importance_tier(role)
        if tier == "decision":
            return self._parse_model_pool(getattr(settings, "MODEL_CF_POOL_DECISION", ""))
        if tier == "strategy":
            return self._parse_model_pool(getattr(settings, "MODEL_CF_POOL_STRATEGY", ""))
        if tier == "vision":
            return self._parse_model_pool(getattr(settings, "MODEL_CF_POOL_VISION", ""))
        return self._parse_model_pool(getattr(settings, "MODEL_CF_POOL_LIGHT", ""))

    def _try_cloudflare_fallback(
        self,
        role: str,
        system_prompt: str,
        msg: str,
        max_tokens: int,
    ) -> str:
        pool = self._cloudflare_pool_for_role(role)
        if not pool:
            return ""

        for model_id in pool:
            if not self._is_model_available(model_id):
                logger.debug(f"Circuit Breaker: Skipping Cloudflare candidate [{model_id}]")
                continue

            logger.info(f"Cloudflare Fallback ({role}): Trying [{model_id}]...")
            result = self._cf_generator.generate(
                system_prompt,
                msg,
                max_tokens=max_tokens,
                model=model_id,
            )
            if result:
                logger.success(f"Cloudflare Fallback SUCCEEDED for [{role}] with [{model_id}]")
                return result
            failure_kind = self._cf_generator.last_failure_kind
            if failure_kind == "rate_limit":
                cooldown = max(300, int(getattr(settings, "CLOUDFLARE_AI_RATE_LIMIT_EXHAUST_SECONDS", 1800)))
            elif failure_kind in {"timeout", "http_error", "request_error"}:
                cooldown = max(60, int(getattr(settings, "CLOUDFLARE_AI_TIMEOUT_EXHAUST_SECONDS", 180)))
            else:
                cooldown = 60
            logger.warning(f"Cloudflare Fallback yielded no usable text for [{role}] with [{model_id}]")
            self._mark_model_exhausted(model_id, duration=cooldown)

        return ""

    def _cerebras_fallback_model(self, role: str) -> str:
        role = (role or "").lower()
        if role == "news_brief_final":
            return settings.MODEL_NEWS_FINAL_FALLBACK
        if role in ("meta_regime", "monitor_hourly"):
            return "qwen/qwen3-32b"
        if role == "risk_eval":
            return settings.MODEL_RISK_EVAL_FALLBACK
        return settings.MODEL_RAG_EXTRACTION

    def _select_gemini_client(self, backend: str):
        if backend == "gemini_judge":
            return self._gemini_judge
        if backend == "gemini_vlm":
            return self._gemini_vlm
        return self._gemini_default

    def generate_response(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: int = 4000,
        temperature: float = 0.7,
        chart_image_b64: Optional[str] = None,
        use_premium: bool = False,
        role: str = "general",
    ) -> str:
        # 백엔드 결정 → 해당 백엔드 락만 획득 (크로스 백엔드 병렬 허용)
        backend_key, _, _ = self._get_route(role)
        provider = backend_key.split("_")[0]  # "gemini_judge" → "gemini"
        lock = self._backend_locks.get(provider, self._backend_locks["gemini"])

        with lock:
            now = time.time()
            elapsed = now - self._last_call_by_backend.get(provider, 0.0)
            if elapsed < self._MIN_GAP:
                time.sleep(self._MIN_GAP - elapsed)

            result = self._execute_routed_call(
                system_prompt=system_prompt,
                user_message=user_message,
                max_tokens=max_tokens,
                temperature=temperature,
                chart_image_b64=chart_image_b64,
                use_premium=use_premium,
                role=role,
            )
            self._last_call_by_backend[provider] = time.time()
            return self._strip_thinking_tags(result)

    @staticmethod
    def _strip_thinking_tags(text: str) -> str:
        """Strip <think>...</think> blocks produced by reasoning models (Qwen3, DeepSeek-R1, etc.)."""
        if not text or "<think>" not in text:
            return text
        stripped = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        return stripped.strip()

    def _execute_routed_call(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: int,
        temperature: float,
        chart_image_b64: Optional[str],
        use_premium: bool,
        role: str,
    ) -> str:
        backend, model_id, cap = self._get_route(role)
        msg = self._trim_input(user_message, cap)

        if backend.startswith("gemini"):
            return self._handle_gemini_backend(
                backend, model_id, role, system_prompt, msg, max_tokens, temperature, chart_image_b64
            )
        if backend == "cerebras":
            return self._handle_cerebras_backend(
                model_id, role, system_prompt, msg, max_tokens, temperature, chart_image_b64
            )
        if backend == "groq":
            return self._handle_groq_backend(
                model_id, role, system_prompt, msg, max_tokens, temperature, chart_image_b64
            )
        if backend == "openrouter":
            return self._handle_openrouter_backend(
                model_id, system_prompt, msg, max_tokens, temperature
            )
        if backend == "cf":
            return self._cf_generator.generate(system_prompt, msg, max_tokens, model=model_id) or self._generate_gemini(
                self._gemini_default,
                self.default_model_id,
                system_prompt,
                msg,
                max_tokens,
                temperature,
                None,
            )
        if backend == "claude":
            logger.error("Claude backend blocked. Falling back to Gemini judge.")
            return self._generate_gemini(
                self._gemini_judge,
                settings.MODEL_JUDGE,
                system_prompt,
                msg,
                max_tokens,
                temperature,
                chart_image_b64,
            )

        return self._generate_gemini(
            self._gemini_default,
            self.default_model_id,
            system_prompt,
            msg,
            max_tokens,
            temperature,
            chart_image_b64,
        )

    def _handle_gemini_backend(
        self,
        backend: str,
        model_id: str,
        role: str,
        system_prompt: str,
        msg: str,
        max_tokens: int,
        temperature: float,
        chart_image_b64: Optional[str],
    ) -> str:
        client_obj = self._select_gemini_client(backend)
        if client_obj and self._is_model_available(model_id):
            # judge / self_correction: use context caching for the static system prompt
            if role in ("judge", "self_correction"):
                result = self._generate_gemini_with_cache(
                    client_obj, model_id, system_prompt, msg, max_tokens, temperature, chart_image_b64
                )
            else:
                result = self._generate_gemini(
                    client_obj, model_id, system_prompt, msg, max_tokens, temperature, chart_image_b64
                )
            if result:
                return result

        # For judge-critical decisions, try a Gemini fallback model before leaving Gemini.
        if client_obj and role in ("judge", "self_correction"):
            judge_fallback = getattr(settings, "MODEL_JUDGE_FALLBACK", "") or "gemini-3-flash-preview"
            if judge_fallback != model_id and self._is_model_available(judge_fallback):
                logger.warning(
                    f"Gemini primary {model_id} failed for role {role}. "
                    f"Trying judge fallback {judge_fallback}..."
                )
                result = self._generate_gemini(
                    client_obj, judge_fallback, system_prompt, msg, max_tokens, temperature, chart_image_b64
                )
                if result:
                    logger.success(f"Gemini judge fallback succeeded with {judge_fallback}")
                    return result

        logger.warning(f"Gemini {model_id} exhausted or failed for role {role}. Triggering relay...")
        if self._is_critical_role(role) and self.cerebras_client:
            result = self._generate_openai_compat(
                self.cerebras_client,
                settings.MODEL_META_REGIME,
                system_prompt,
                msg,
                max_tokens,
                temperature,
                None,
                name="Cerebras(Relay)",
            ) or self._generate_openai_compat(
                self.groq_client,
                "llama-3.3-70b-versatile",
                system_prompt,
                msg,
                max_tokens,
                temperature,
                None,
                name="Groq(Relay)",
            )
            if result:
                return result
            result = self._try_cloudflare_fallback(role, system_prompt, msg, max_tokens)
            if result:
                return result
            return ""

        result = self._generate_openai_compat(
            self.groq_client,
            "llama-3.1-8b-instant",
            system_prompt,
            msg,
            max_tokens,
            temperature,
            None,
            name="Groq(Relay)",
        )
        if result:
            return result
        result = self._try_cloudflare_fallback(role, system_prompt, msg, max_tokens)
        if result:
            return result
        return ""

    def _handle_cerebras_backend(
        self,
        model_id: str,
        role: str,
        system_prompt: str,
        msg: str,
        max_tokens: int,
        temperature: float,
        chart_image_b64: Optional[str],
    ) -> str:
        result = self._generate_openai_compat(
            self.cerebras_client,
            model_id,
            system_prompt,
            msg,
            max_tokens,
            temperature,
            chart_image_b64,
            timeout=120.0,  # [FIX] 늘어난 임아웃 (실패 방)
            name="Cerebras",
        )
        if result:
            return result
        logger.warning(f"Cerebras failed for {model_id}, falling back to Groq")
        fallback_model = self._cerebras_fallback_model(role)
        if self.groq_client:
            result = self._handle_groq_backend(
                fallback_model,
                role,
                system_prompt,
                msg,
                max_tokens,
                temperature,
                None,
            )
            if result:
                return result
        result = self._try_cloudflare_fallback(role, system_prompt, msg, max_tokens)
        if result:
            return result
        logger.warning("Groq client unavailable after Cerebras failure. Falling back to Gemini.")
        return self._generate_gemini(
            self._gemini_default,
            self.default_model_id,
            system_prompt,
            msg,
            max_tokens,
            temperature,
            None,
        )

    def _handle_groq_backend(
        self,
        model_id: str,
        role: str,
        system_prompt: str,
        msg: str,
        max_tokens: int,
        temperature: float,
        chart_image_b64: Optional[str],
    ) -> str:
        if self._is_model_available(model_id):
            result = self._generate_openai_compat(
                self.groq_client,
                model_id,
                system_prompt,
                msg,
                max_tokens,
                temperature,
                chart_image_b64,
                timeout=60.0,
                name="Groq",
            )
            if result:
                return result
        else:
            logger.debug(f"Circuit Breaker: Skipping primary [{model_id}]")

        is_critical = self._is_critical_role(role)
        relay_pool = self._GROQ_REASONING_POOL if is_critical else self._GROQ_GENERAL_POOL
        logger.warning(f"Groq primary ({model_id}) unavailable or hit limit for role [{role}]. Searching relay pool...")

        for alt_model in relay_pool:
            if alt_model == model_id:
                continue
            if not self._is_model_available(alt_model):
                logger.debug(f"Circuit Breaker: Skipping relay candidate [{alt_model}]")
                continue

            logger.info(f"Groq Relay ({role}): Trying [{alt_model}]...")
            result = self._generate_openai_compat(
                self.groq_client,
                alt_model,
                system_prompt,
                msg,
                max_tokens,
                temperature,
                chart_image_b64,
                timeout=20.0,
                name=f"Groq-Relay({alt_model})",
            )
            if result:
                logger.success(f"Groq Relay SUCCEEDED for [{role}] with [{alt_model}]")
                return result

        result = self._try_cloudflare_fallback(role, system_prompt, msg, max_tokens)
        if result:
            return result

        if is_critical:
            if not self._is_model_available(settings.MODEL_JUDGE):
                logger.warning(
                    f"All Groq models exhausted for critical role [{role}], "
                    f"but Gemini Pro ({settings.MODEL_JUDGE}) is also exhausted. Returning empty."
                )
                return ""
            logger.warning(f"All Groq models exhausted for critical role [{role}]. Trying Gemini fallback...")
            return self._generate_gemini(
                self._gemini_judge,
                settings.MODEL_JUDGE,
                system_prompt,
                msg,
                max_tokens,
                temperature,
                chart_image_b64,
            )

        logger.warning(f"All Groq relay models failed for [{role}], falling back to basic providers")
        return self._generate_gemini(
            self._gemini_default,
            self.default_model_id,
            system_prompt,
            msg,
            max_tokens,
            temperature,
            None,
        )

    def _handle_openrouter_backend(
        self,
        model_id: str,
        system_prompt: str,
        msg: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        result = self._generate_openai_compat(
            self.openrouter_client,
            model_id,
            system_prompt,
            msg,
            max_tokens,
            temperature,
            None,
            timeout=90.0,
            name="OpenRouter",
        )
        if result:
            return result
        logger.warning("OpenRouter failed, falling back to Groq")
        return self._generate_openai_compat(
            self.groq_client,
            "llama-3.3-70b-versatile",
            system_prompt,
            msg,
            max_tokens,
            temperature,
            None,
            timeout=25.0,
            name="Groq",
        ) or ""

    def _generate_gemini(
        self,
        client,
        model_id: str,
        system_prompt: str,
        user_message: str,
        max_tokens: int,
        temperature: float,
        chart_image_b64: Optional[str] = None,
    ) -> str:
        max_retries = 3
        base_delay = 5.0

        for attempt in range(max_retries):
            try:
                parts = []
                if chart_image_b64:
                    image_bytes = base64.b64decode(chart_image_b64)
                    mime = mimetypes.guess_type("chart.png")[0] or "image/png"
                    parts.append(types.Part.from_bytes(data=image_bytes, mime_type=mime))
                parts.append(types.Part.from_text(text=user_message))

                config_kwargs = {
                    "system_instruction": system_prompt,
                    "max_output_tokens": max_tokens,
                    "temperature": temperature,
                }
                # Support thinking_config for gemini-3 and gemini-2.5 series
                lowercased_id = model_id.lower()
                if "gemini-3" in lowercased_id or "gemini-2.5" in lowercased_id:
                    thinking_level = "HIGH" if "pro" in lowercased_id else "LOW"
                    config_kwargs["thinking_config"] = types.ThinkingConfig(
                        thinking_level=thinking_level
                    )

                response = client.models.generate_content(
                    model=model_id,
                    contents=[types.Content(role="user", parts=parts)],
                    config=types.GenerateContentConfig(**config_kwargs),
                )
                return response.text or ""
            except Exception as e:
                err_str = str(e).lower()
                if "429" in err_str or "resource" in err_str or "503" in err_str or "exhausted" in err_str:
                    self._mark_model_exhausted(model_id, 3600)
                    if attempt < max_retries - 1:
                        sleep_t = base_delay * (2 ** attempt)
                        logger.warning(f"Gemini rate limit ({model_id}), retry in {sleep_t}s")
                        time.sleep(sleep_t)
                        continue
                logger.error(f"Gemini error ({model_id}): {e}")
                return ""
        return ""

    def _generate_openai_compat(
        self,
        client,
        model_id: str,
        system_prompt: str,
        user_message: str,
        max_tokens: int,
        temperature: float,
        chart_image_b64: Optional[str],
        timeout: float = 30.0,
        name: str = "OpenAI-compat",
    ) -> str:
        if client is None:
            return ""

        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            if chart_image_b64:
                user_content = []
                mime = mimetypes.guess_type("chart.png")[0] or "image/png"
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{chart_image_b64}"},
                    }
                )
                user_content.append({"type": "text", "text": user_message})
                messages.append({"role": "user", "content": user_content})
            else:
                # ------------------------ FIX ( ) ------------------------
                messages.append({"role": "user", "content": user_message})

            resp = client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "rate_limit" in err_str or "quota" in err_str:
                self._mark_model_exhausted(model_id, 3600)
            logger.error(f"{name} error ({model_id}): {e}")
            return ""

    def _generate_gemini_with_cache(
        self,
        client,
        model_id: str,
        system_prompt: str,
        user_message: str,
        max_tokens: int,
        temperature: float,
        chart_image_b64: Optional[str] = None,
    ) -> str:
        """Gemini generation with context caching for the system prompt.

        Caches the static system instruction (Judge BASE_PROMPT + rules + DEBATE_APPENDIX)
        for 2 hours so it is not re-tokenised on every analysis cycle.
        Falls back silently to a standard call if the model/quota does not support caching.
        """
        import hashlib

        cache_key = hashlib.md5(f"{model_id}:{system_prompt}".encode()).hexdigest()
        cache_ttl = 7200  # 2 hours in seconds
        now = time.time()

        # Look up existing valid cache
        cache_name: Optional[str] = None
        with self._gemini_cache_lock:
            entry = self._gemini_system_caches.get(cache_key)
            if entry:
                stored_name, stored_ts = entry
                if now - stored_ts < cache_ttl:
                    cache_name = stored_name

        # Create a new cache if we don't have one
        if cache_name is None:
            try:
                cache = client.caches.create(
                    model=model_id,
                    config=types.CreateCachedContentConfig(
                        system_instruction=system_prompt,
                        ttl=f"{cache_ttl}s",
                    ),
                )
                cache_name = cache.name
                with self._gemini_cache_lock:
                    self._gemini_system_caches[cache_key] = (cache_name, now)
                logger.info(f"Gemini context cache created for {model_id}: {cache_name[:24]}…")
            except Exception as exc:
                logger.debug(f"Gemini context caching unavailable ({exc}); using standard call")
                return self._generate_gemini(
                    client, model_id, system_prompt, user_message,
                    max_tokens, temperature, chart_image_b64,
                )

        # Generate using the cached system instruction
        # Thinking 모델(Pro)은 응답이 수 분 걸릴 수 있으므로 300s 하드 타임아웃 적용
        # → hang이 발생해도 분석 스레드가 무기한 점유되지 않도록 방지
        _CACHE_GEN_TIMEOUT = 300  # seconds
        try:
            parts = []
            if chart_image_b64:
                image_bytes = base64.b64decode(chart_image_b64)
                mime = mimetypes.guess_type("chart.png")[0] or "image/png"
                parts.append(types.Part.from_bytes(data=image_bytes, mime_type=mime))
            parts.append(types.Part.from_text(text=user_message))

            config_kwargs: Dict = {
                "cached_content": cache_name,
                "max_output_tokens": max_tokens,
                "temperature": temperature,
            }
            lowercased_id = model_id.lower()
            if "gemini-3" in lowercased_id or "gemini-2.5" in lowercased_id:
                thinking_level = "HIGH" if "pro" in lowercased_id else "LOW"
                config_kwargs["thinking_config"] = types.ThinkingConfig(
                    thinking_level=thinking_level
                )

            import concurrent.futures as _cf
            with _cf.ThreadPoolExecutor(max_workers=1) as _pool:
                _future = _pool.submit(
                    client.models.generate_content,
                    model=model_id,
                    contents=[types.Content(role="user", parts=parts)],
                    config=types.GenerateContentConfig(**config_kwargs),
                )
                try:
                    response = _future.result(timeout=_CACHE_GEN_TIMEOUT)
                except _cf.TimeoutError:
                    logger.warning(
                        f"Gemini cached generation TIMEOUT ({_CACHE_GEN_TIMEOUT}s) for {model_id}; "
                        "falling back to standard call"
                    )
                    with self._gemini_cache_lock:
                        self._gemini_system_caches.pop(cache_key, None)
                    return self._generate_gemini(
                        client, model_id, system_prompt, user_message,
                        max_tokens, temperature, chart_image_b64,
                    )
            return response.text or ""
        except Exception as exc:
            logger.warning(f"Gemini cached generation failed ({exc}); retrying without cache")
            with self._gemini_cache_lock:
                self._gemini_system_caches.pop(cache_key, None)
            return self._generate_gemini(
                client, model_id, system_prompt, user_message,
                max_tokens, temperature, chart_image_b64,
            )

    def _generate_claude(
        self,
        model_id: str,
        system_prompt: str,
        user_message: str,
        max_tokens: int,
        temperature: float,
        chart_image_b64: Optional[str] = None,
    ) -> str:
        logger.error(f"_generate_claude({model_id}) called but Anthropic is disabled. Using Gemini fallback.")
        return self._generate_gemini(
            self._gemini_judge,
            self.premium_model_id,
            system_prompt,
            user_message,
            max_tokens,
            temperature,
            chart_image_b64,
        )

    def _get_role_model_and_cap(self, role: str, use_premium: bool) -> Tuple[str, int]:
        _, model_id, cap = self._get_route(role)
        return model_id, cap

    def generate_with_context(
        self,
        system_prompt: str,
        conversation_history: List[Dict],
        max_tokens: int = 4000,
        temperature: float = 0.7,
        use_premium: bool = False,
        role: str = "general",
    ) -> str:
        flat_text = "\n".join(
            [f"{m.get('role', 'user')}: {m.get('content', '')}" for m in conversation_history]
        )
        return self.generate_response(
            system_prompt=system_prompt,
            user_message=flat_text,
            max_tokens=max_tokens,
            temperature=temperature,
            use_premium=use_premium,
            role=role,
        )


ai_client = AIClient()
