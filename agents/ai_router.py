"""Hybrid AI client with multi-LLM routing — 2026 role policy table.

Role → API → Model (single source of truth):
  judge            → Gemini AI Studio Project A → gemini-3.1-pro-preview
  vlm_geometric    → Gemini AI Studio Project B → gemini-3.1-pro-preview
  vlm_telegram_chart → Gemini AI Studio Project B → gemini-3.1-pro-preview
  meta_regime      → Cerebras              → gpt-oss-120b
  risk_eval        → Cerebras              → gpt-oss-120b
  risk_eval_fallback → Groq               → openai/gpt-oss-20b
  rag_extraction   → Groq                 → openai/gpt-oss-20b
  news_summarize   → Groq                 → openai/gpt-oss-20b
  monitor_hourly   → Cerebras              → gpt-oss-120b
  cloudflare_triage → Cloudflare Workers AI → @cf/meta/llama-3-8b-instruct-awq
  cloudflare_rerank → Cloudflare Workers AI → @cf/baai/bge-reranker-base
  claude_standby   → Anthropic            → claude-sonnet-4-6  (manual only)
  liquidity / microstructure / macro / chat → Gemini Flash → gemini-3-flash-preview
"""

import base64
import mimetypes
import time
from typing import Dict, List, Optional, Tuple

from google import genai
from google.genai import types
from anthropic import Anthropic
from openai import OpenAI
from loguru import logger

import threading
from config.settings import settings


class CloudflareGenerator:
    """Cloudflare Workers AI — cloudflare_triage and cloudflare_rerank roles."""

    def __init__(self, model: Optional[str] = None):
        self._model = model  # e.g. "@cf/meta/llama-3-8b-instruct-awq"
        self._account_id = ""
        self._api_key = ""
        self._enabled = False
        self._session = None

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

    def generate(
        self, system_prompt: str, user_message: str, max_tokens: int = 1000,
        model: Optional[str] = None
    ) -> Optional[str]:
        self._init()
        if not self._enabled:
            return None
        import requests
        _model = model or self._model or settings.MODEL_CF_TRIAGE
        url = self._url(_model)
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
                self._session.headers.update({"Authorization": f"Bearer {self._api_key}"})
            resp = self._session.post(url, json=payload, timeout=10.0)
            resp.raise_for_status()
            return resp.json().get("result", {}).get("response", "")
        except Exception as e:
            logger.warning(f"CloudflareGenerator failed ({_model}): {e}")
            return None


class AIClient:
    def __init__(self):
        # ── Gemini (dual-project) ─────────────────────────────────────────────
        legacy_default_key = getattr(settings, "GEMINI_API_KEY", "")
        judge_key = getattr(settings, "GEMINI_API_KEY_JUDGE", "") or legacy_default_key
        vlm_key = getattr(settings, "GEMINI_API_KEY_VLM", "") or legacy_default_key
        # Final safety net for gemini_default path:
        # VLM key -> Judge key -> legacy default key -> Vertex AI auth.
        default_key = vlm_key or judge_key or legacy_default_key

        def _make_gemini(key: str):
            if key:
                return genai.Client(api_key=key)
            return genai.Client(
                vertexai=True,
                project=settings.PROJECT_ID,
                location=settings.VERTEX_REGION_GEMINI or "global",
            )

        self._gemini_default = _make_gemini(default_key)
        self._gemini_judge = _make_gemini(judge_key)
        self._gemini_vlm = _make_gemini(vlm_key)
        logger.info("Gemini clients initialized (default / judge / vlm)")

        # ── Anthropic (claude_standby — reserved) ────────────────────────────
        self._claude_client = None

        # ── Cerebras (meta_regime + risk_eval) ───────────────────────────────
        self._cerebras_client = None  # lazy init

        # ── Groq (rag_extraction / news_summarize / risk_eval_fallback) ──────
        self._groq_client = None

        # ── OpenRouter (monitor_hourly — free tier) ───────────────────────────
        self._openrouter_client = None

        # ── Cloudflare Workers AI ─────────────────────────────────────────────
        self._cf_generator = CloudflareGenerator()

        # backward-compat
        self.default_model_id = "gemini-3-flash-preview"  # chat/fallback only
        self.premium_model_id = settings.MODEL_JUDGE

        # ── Rate Limiting & Concurrency Control (V14.3) ──────────────────────
        self._global_api_lock = threading.Lock()
        self._last_call_timestamp = 0.0
        self._MIN_GAP = 0.5  # 500ms floor between ANY two AI calls


    # ── Lazy clients ──────────────────────────────────────────────────────────
    @property
    def claude_client(self):
        if self._claude_client is None:
            if settings.ANTHROPIC_API_KEY:
                self._claude_client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)
            else:
                logger.warning("ANTHROPIC_API_KEY not set — claude_standby unavailable")
                return None
        return self._claude_client

    @property
    def cerebras_client(self):
        """Cerebras via OpenAI-compatible endpoint."""
        if self._cerebras_client is None:
            key = getattr(settings, "CEREBRAS_API_KEY", "")
            if key:
                self._cerebras_client = OpenAI(
                    base_url="https://api.cerebras.ai/v1",
                    api_key=key,
                )
            else:
                logger.warning("CEREBRAS_API_KEY not set")
                return None
        return self._cerebras_client

    @property
    def groq_client(self):
        if self._groq_client is None:
            if settings.GROQ_API_KEY:
                self._groq_client = OpenAI(
                    base_url="https://api.groq.com/openai/v1",
                    api_key=settings.GROQ_API_KEY,
                )
            else:
                return None
        return self._groq_client

    @property
    def openrouter_client(self):
        if self._openrouter_client is None:
            key = (
                getattr(settings, "OPENROUTER_API_KEY", "")
            )
            if key:
                self._openrouter_client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=key,
                )
            else:
                logger.warning("OPENROUTER_API_KEY not set — monitor_hourly will fall back")
                return None
        return self._openrouter_client

    def _openai_client(self):
        if settings.OPENAI_API_KEY:
            return OpenAI(api_key=settings.OPENAI_API_KEY)
        return None

    # ── Role → (gemini_client, model_id, input_cap) ──────────────────────────
    def _get_route(self, role: str) -> Tuple[str, str, int]:
        """Returns (backend_tag, model_id, input_cap_chars).

        backend_tag: "gemini_default" | "gemini_judge" | "gemini_vlm" |
                     "cerebras" | "groq" | "openrouter" | "claude" | "cf"
        """
        role = (role or "general").lower()

        ROLE_MAP = {
            # judge / vlm
            "judge":              ("gemini_judge",   settings.MODEL_JUDGE,              settings.MAX_INPUT_CHARS_JUDGE),
            "self_correction":    ("gemini_judge",   settings.MODEL_SELF_CORRECTION,    settings.MAX_INPUT_CHARS_SELF_CORRECTION),
            "vlm_geometric":      ("gemini_vlm",     settings.MODEL_VLM_GEOMETRIC,      getattr(settings, "MAX_INPUT_CHARS_VLM_GEOMETRIC", 15000)),
            "vlm_analysis":       ("gemini_vlm",     settings.MODEL_VLM_GEOMETRIC,      15000),
            "vlm_telegram_chart": ("gemini_vlm",     settings.MODEL_VLM_TELEGRAM_CHART, settings.MAX_INPUT_CHARS_VLM_TELEGRAM_CHART),
            "rag_vision":         ("gemini_vlm",     settings.MODEL_VLM_GEOMETRIC,      15000),
            # meta_regime / risk_eval → Cerebras
            "meta_regime":        ("cerebras",       settings.MODEL_META_REGIME,        15000),
            "macro":              ("cerebras",       settings.MODEL_META_REGIME,        settings.MAX_INPUT_CHARS_MACRO),
            "risk_eval":          ("cerebras",       settings.MODEL_RISK_EVAL,          10000),
            "risk_eval_fallback": ("groq",           settings.MODEL_RISK_EVAL_FALLBACK, 10000),
            # Groq roles
            "rag_extraction":     ("groq",           settings.MODEL_RAG_EXTRACTION,     settings.MAX_INPUT_CHARS_RAG_EXTRACTION),
            "news_summarize":     ("groq",           settings.MODEL_NEWS_SUMMARIZE,     10000),
            "post_mortem":        ("groq",           settings.MODEL_RAG_EXTRACTION,     15000),
            "feedback":           ("groq",           settings.MODEL_RAG_EXTRACTION,     15000),
            # OpenRouter — monitor_hourly
            "monitor_hourly":     ("cerebras",       settings.MODEL_MONITOR_HOURLY,     8000),
            # Cloudflare
            "cloudflare_triage":  ("cf",             settings.MODEL_CF_TRIAGE,          5000),
            "cloudflare_rerank":  ("cf",             settings.MODEL_CF_RERANK,          5000),
            "triage":             ("cf",             settings.MODEL_CF_TRIAGE,          5000),
            # claude_standby — manual only
            "claude_standby":     ("claude",         settings.MODEL_CLAUDE_STANDBY,     20000),
            # Chat / UI only — NOT used in analysis pipeline
            "chat":               ("gemini_default", settings.MODEL_CHAT,               10000),
            # Chat / UI fallback (not used in analysis pipeline)
        }

        if role in ROLE_MAP:
            return ROLE_MAP[role]

        # Fallback: Groq general if available, else Gemini default
        if self.groq_client:
            return ("groq", "llama-3.3-70b-versatile", 15000)
        return ("gemini_default", self.default_model_id, 15000)

    def _trim_input(self, text: str, max_chars: int) -> str:
        if not text or len(text) <= max_chars:
            return text or ""
        return text[:max_chars] + f"\n\n[TRUNCATED {len(text)-max_chars} chars]"

    # ── Public interface ──────────────────────────────────────────────────────
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
        # ── Global Concurrency & Rate Limit Guard ────────────────────────────
        with self._global_api_lock:
            # Enforce minimum gap between calls
            now = time.time()
            elapsed = now - self._last_call_timestamp
            if elapsed < self._MIN_GAP:
                wait_time = self._MIN_GAP - elapsed
                time.sleep(wait_time)
            
            # Execute the routed call
            result = self._execute_routed_call(
                system_prompt, user_message, max_tokens, temperature, 
                chart_image_b64, use_premium, role
            )
            
            # Update timestamp AFTER call completes for a clean post-processing gap
            self._last_call_timestamp = time.time()
            return result

    def _execute_routed_call(
        self, system_prompt: str, user_message: str, max_tokens: int,
        temperature: float, chart_image_b64: Optional[str],
        use_premium: bool, role: str
    ) -> str:
        backend, model_id, cap = self._get_route(role)
        msg = self._trim_input(user_message, cap)


        if backend == "gemini_judge":
            return self._generate_gemini(self._gemini_judge, model_id, system_prompt, msg, max_tokens, temperature, chart_image_b64)
        if backend == "gemini_vlm":
            return self._generate_gemini(self._gemini_vlm, model_id, system_prompt, msg, max_tokens, temperature, chart_image_b64)
        if backend == "gemini_default":
            return self._generate_gemini(self._gemini_default, model_id, system_prompt, msg, max_tokens, temperature, chart_image_b64)
        if backend == "cerebras":
            result = self._generate_openai_compat(self.cerebras_client, model_id, system_prompt, msg, max_tokens, temperature, chart_image_b64, timeout=30.0, name="Cerebras")
            if result:
                return result
            # fallback → Groq risk_eval_fallback
            logger.warning(f"Cerebras failed for {model_id}, falling back to Groq")
            return self._generate_openai_compat(self.groq_client, settings.MODEL_RISK_EVAL_FALLBACK, system_prompt, msg, max_tokens, temperature, None, timeout=25.0, name="Groq(fallback)")
        if backend == "groq":
            result = self._generate_openai_compat(self.groq_client, model_id, system_prompt, msg, max_tokens, temperature, chart_image_b64, timeout=25.0, name="Groq")
            if result:
                return result
            logger.warning(f"Groq failed for {model_id}, falling back to Cloudflare")
            return self._cf_generator.generate(system_prompt, msg, max_tokens) or \
                   self._generate_gemini(self._gemini_default, self.default_model_id, system_prompt, msg, max_tokens, temperature, None)
        if backend == "openrouter":
            result = self._generate_openai_compat(self.openrouter_client, model_id, system_prompt, msg, max_tokens, temperature, None, timeout=30.0, name="OpenRouter")
            if result:
                return result
            logger.warning("OpenRouter failed, falling back to Groq")
            return self._generate_openai_compat(self.groq_client, "llama-3.3-70b-versatile", system_prompt, msg, max_tokens, temperature, None, timeout=25.0, name="Groq") or ""
        if backend == "cf":
            return self._cf_generator.generate(system_prompt, msg, max_tokens, model=model_id) or \
                   self._generate_gemini(self._gemini_default, self.default_model_id, system_prompt, msg, max_tokens, temperature, None)
        if backend == "claude":
            return self._generate_claude(model_id, system_prompt, msg, max_tokens, temperature, chart_image_b64)

        # Final fallback
        return self._generate_gemini(self._gemini_default, self.default_model_id, system_prompt, msg, max_tokens, temperature, chart_image_b64)

    # ── Backend implementations ───────────────────────────────────────────────
    def _generate_gemini(self, client, model_id: str, system_prompt: str, user_message: str,
                         max_tokens: int, temperature: float,
                         chart_image_b64: Optional[str] = None) -> str:
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
                if "gemini-3" in model_id.lower():
                    thinking_level = "HIGH" if "pro" in model_id.lower() else "LOW"
                    config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=thinking_level)

                config = types.GenerateContentConfig(**config_kwargs)
                response = client.models.generate_content(
                    model=model_id,
                    contents=[types.Content(role="user", parts=parts)],
                    config=config,
                )
                return response.text or ""
            except Exception as e:
                err = str(e).lower()
                if ("429" in err or "resource" in err or "503" in err or "exhausted" in err) and attempt < max_retries - 1:
                    sleep_t = base_delay * (2 ** attempt)
                    logger.warning(f"Gemini rate limit ({model_id}), retry in {sleep_t}s")
                    time.sleep(sleep_t)
                    continue
                logger.error(f"Gemini error ({model_id}): {e}")
                return ""
        return ""

    def _generate_openai_compat(
        self, client, model_id: str, system_prompt: str, user_message: str,
        max_tokens: int, temperature: float, chart_image_b64: Optional[str],
        timeout: float = 30.0, name: str = "OpenAI-compat"
    ) -> str:
        if client is None:
            return ""
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            user_content = []
            if chart_image_b64:
                mime = mimetypes.guess_type("chart.png")[0] or "image/png"
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{chart_image_b64}"}
                })
            user_content.append({"type": "text", "text": user_message})
            messages.append({"role": "user", "content": user_content})

            kwargs = {
                "model": model_id,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "timeout": timeout,
            }
            resp = client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"{name} error ({model_id}): {e}")
            return ""

    def _generate_claude(self, model_id: str, system_prompt: str, user_message: str,
                         max_tokens: int, temperature: float,
                         chart_image_b64: Optional[str] = None) -> str:
        try:
            if self.claude_client is None:
                logger.warning(f"Claude unavailable ({model_id}), falling back to Gemini")
                return self._generate_gemini(self._gemini_judge, self.premium_model_id, system_prompt, user_message, max_tokens, temperature, chart_image_b64)
            content = []
            if chart_image_b64:
                content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": chart_image_b64}})
            content.append({"type": "text", "text": user_message})
            response = self.claude_client.messages.create(
                model=model_id, max_tokens=max_tokens, temperature=temperature,
                system=system_prompt, messages=[{"role": "user", "content": content}], timeout=45.0,
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Claude error ({model_id}): {e}")
            return self._generate_gemini(self._gemini_judge, self.premium_model_id, system_prompt, user_message, max_tokens, temperature, chart_image_b64)

    # ── Legacy compat ─────────────────────────────────────────────────────────
    def _get_role_model_and_cap(self, role: str, use_premium: bool) -> Tuple[str, int]:
        """Backward-compat shim used by older agents."""
        backend, model_id, cap = self._get_route(role)
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
        flat_text = "\n".join([f"{m.get('role','user')}: {m.get('content','')}" for m in conversation_history])
        return self.generate_response(system_prompt=system_prompt, user_message=flat_text,
                                      max_tokens=max_tokens, temperature=temperature,
                                      use_premium=use_premium, role=role)


# Global singleton
ai_client = AIClient()
