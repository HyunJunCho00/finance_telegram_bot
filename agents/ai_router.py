"""Hybrid AI client with multi-LLM routing for 2026 SOTA models.

Key goals:
- Route requests to Gemini, Claude, or GPT based on the prefix.
- Keep Judge on strongest reasoning model (Claude/GPT).
- Use faster/cheaper models for high-frequency agents (Gemini Flash).
- Apply soft input caps per role to improve token efficiency.
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

from config.settings import settings


class CloudflareGenerator:
    """Cloudflare Workers AI Llama 3.1 70B — Zero-cost text generation.
    
    Used for secondary agents (Macro, Liquidity, Microstructure) to save Gemini costs.
    """
    _URL_TEMPLATE = "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/@cf/meta/llama-3.1-70b-instruct-fp8-fast"

    def __init__(self):
        self._account_id = ""
        self._api_key = ""
        self._enabled = False
        self._session = None

    def _init(self):
        if self._enabled: return
        try:
            from config.settings import settings
            self._account_id = getattr(settings, "CLOUDFLARE_ACCOUNT_ID", "")
            self._api_key = getattr(settings, "CLOUDFLARE_AI_API_KEY", "")
            self._enabled = bool(self._account_id and self._api_key)
        except: pass

    def generate(self, system_prompt: str, user_message: str, max_tokens: int = 1000) -> Optional[str]:
        self._init()
        if not self._enabled: return None

        import requests
        url = self._URL_TEMPLATE.format(account_id=self._account_id)
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            "max_tokens": max_tokens
        }
        
        try:
            if self._session is None:
                self._session = requests.Session()
                self._session.headers.update({"Authorization": f"Bearer {self._api_key}"})
            
            resp = self._session.post(url, json=payload, timeout=10.0)
            resp.raise_for_status()
            return resp.json().get("result", {}).get("response", "")
        except Exception as e:
            logger.warning(f"CloudflareGenerator failed: {e}")
            return None


class AIClient:
    def __init__(self):
        # Gemini Client (Vertex AI or Direct API)
        if settings.GEMINI_API_KEY:
            self._gemini_client = genai.Client(
                api_key=settings.GEMINI_API_KEY
            )
            logger.info("Gemini initialized with Direct API Key")
        else:
            self._gemini_client = genai.Client(
                vertexai=True,
                project=settings.PROJECT_ID,
                location=settings.VERTEX_REGION_GEMINI or "global",
            )
            logger.info(f"Gemini initialized with Vertex AI (Project: {settings.PROJECT_ID})")
        
        # Anthropic Client (Direct API only — NOT GCP Model Garden)
        self._claude_client = None
        
        # OpenAI Client
        self._openai_client = None

        # Groq Client (OpenAI compatible)
        self._groq_client = None

        # Cloudflare Text Generator
        self._cf_generator = CloudflareGenerator()

        # Backward-compatible defaults
        self.default_model_id = "gemini-2.5-flash"
        self.premium_model_id = settings.MODEL_JUDGE

    @property
    def claude_client(self):
        if self._claude_client is None:
            if settings.ANTHROPIC_API_KEY:
                self._claude_client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)
            else:
                logger.warning("ANTHROPIC_API_KEY not set — Claude models will fall back to Gemini")
                return None
        return self._claude_client

    @property
    def openai_client(self):
        if self._openai_client is None:
            if settings.OPENAI_API_KEY:
                self._openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
            else:
                # [FIX SILENT-2] Don't crash — return None, let _generate_openai handle fallback
                logger.warning("OPENAI_API_KEY not set — GPT models will fall back to Gemini")
                return None
        return self._openai_client

    @property
    def groq_client(self):
        if self._groq_client is None:
            if settings.GROQ_API_KEY:
                # Groq is OpenAI compatible
                self._groq_client = OpenAI(
                    base_url="https://api.groq.com/openai/v1",
                    api_key=settings.GROQ_API_KEY
                )
            else:
                return None
        return self._groq_client

    def _get_role_model_and_cap(self, role: str, use_premium: bool) -> Tuple[str, int]:
        role = (role or "general").lower()

        if role == "judge":
            return settings.MODEL_JUDGE, settings.MAX_INPUT_CHARS_JUDGE
        
        # --- FREE-FIRST ROUTING (2026 Tiered Optimization) ---
        if settings.GROQ_API_KEY:
            if role in ("chat", "routing", "triage"):
                # A. Initial Triage / Fast Mass Processing
                return "groq/llama-3.1-8b-instant", getattr(settings, f"MAX_INPUT_CHARS_{role.upper()}", 10000)
            elif role in ("rag_extraction", "self_correction", "feedback", "post_mortem"):
                # C. Reasoning / Evidence Verification Loop
                return "groq/openai/gpt-oss-20b", getattr(settings, f"MAX_INPUT_CHARS_{role.upper()}", 15000)
            elif role in ("structured_report", "comparison"):
                # D. Medium Precision & Structuring
                return "groq/qwen/qwen3-32b", 20000
            elif role in ("long_synthesis", "weekly_report", "monthly_report"):
                # E. Ultra-long Synthesis
                return "groq/moonshotai/kimi-k2-instruct-0905", 100000
            elif role in ("vlm_telegram_chart", "vlm_geometric", "vlm_analysis", "rag_vision"):
                # Vision Tasks — llama-3.2-90b-vision-preview decommissioned 2025-04-14
                return "groq/meta-llama/llama-4-scout-17b-16e-instruct", getattr(settings, f"MAX_INPUT_CHARS_{role.upper()}", 15000)
            else:
                # B. Main Standard Analysis (liquidity, microstructure, macro, general)
                return "groq/llama-3.3-70b-versatile", getattr(settings, f"MAX_INPUT_CHARS_{role.upper()}", 15000)

        # Fallback to Workers AI if Groq fails or is absent conceptually (handled in wrapper)
        self._cf_generator._init()
        if self._cf_generator._enabled:
             if role in ("liquidity", "microstructure", "macro", "chat", "routing", "rag_extraction"):
                 return "cf/llama-3.1-70b", getattr(settings, f"MAX_INPUT_CHARS_{role.upper()}", 15000)

        # Fail gracefully mechanism. If all else fails and it's not judge, don't use premium resources silently.
        return self.default_model_id, settings.MAX_INPUT_CHARS_LIQUIDITY

    def _trim_input(self, text: str, max_chars: int) -> str:
        if not text:
            return ""
        if len(text) <= max_chars:
            return text
        omitted = len(text) - max_chars
        return text[:max_chars] + f"\n\n[TRUNCATED {omitted} chars for token efficiency]"

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
        model_id, input_cap = self._get_role_model_and_cap(role, use_premium)
        trimmed_message = self._trim_input(user_message, input_cap)

        if model_id.startswith("claude"):
            return self._generate_claude(
                model_id=model_id,
                system_prompt=system_prompt,
                user_message=trimmed_message,
                max_tokens=max_tokens,
                temperature=temperature,
                chart_image_b64=chart_image_b64,
            )
        elif model_id.startswith("gpt-") or model_id.startswith("o1-") or model_id.startswith("o3-"):
            return self._generate_openai(
                model_id=model_id,
                system_prompt=system_prompt,
                user_message=trimmed_message,
                max_tokens=max_tokens,
                temperature=temperature,
                chart_image_b64=chart_image_b64,
            )

        if model_id.startswith("groq/"):
            return self._generate_openai(
                model_id=model_id.replace("groq/", ""),
                system_prompt=system_prompt,
                user_message=trimmed_message,
                max_tokens=max_tokens,
                temperature=temperature,
                chart_image_b64=chart_image_b64,
                client_override=self.groq_client
            )

        if model_id.startswith("cf/"):
            return self._cf_generator.generate(
                system_prompt=system_prompt,
                user_message=trimmed_message,
                max_tokens=max_tokens
            ) or self._generate_gemini(
                model_id=self.default_model_id,
                system_prompt=system_prompt,
                user_message=trimmed_message,
                max_tokens=max_tokens,
                temperature=temperature,
                chart_image_b64=chart_image_b64
            )

        return self._generate_gemini(
            model_id=model_id,
            system_prompt=system_prompt,
            user_message=trimmed_message,
            max_tokens=max_tokens,
            temperature=temperature,
            chart_image_b64=chart_image_b64,
        )

    def _generate_gemini(
        self,
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

                # Inject ThinkingConfig for Gemini 3.0 models
                if "gemini-3" in model_id.lower():
                    # Pro models get HIGH thinking for deep reasoning, Flash models get LOW for speed/cost
                    thinking_level = "HIGH" if "pro" in model_id.lower() else "LOW"
                    config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=thinking_level)
                    # Thinking models tend to perform better with temperature=0.0 depending on the task, but we'll leave it as configured

                config = types.GenerateContentConfig(**config_kwargs)

                response = self._gemini_client.models.generate_content(
                    model=model_id,
                    contents=[types.Content(role="user", parts=parts)],
                    config=config,
                )
                return response.text or ""

            except Exception as e:
                err_str = str(e).lower()
                if "429" in err_str or "resource" in err_str or "503" in err_str or "exhausted" in err_str:
                    if attempt < max_retries - 1:
                        sleep_time = base_delay * (2 ** attempt)
                        logger.warning(f"Gemini API rate limit ({model_id}), retrying in {sleep_time}s...")
                        time.sleep(sleep_time)
                        continue
                logger.error(f"Gemini API error ({model_id}): {e}")
                return ""

    def _generate_claude(
        self,
        model_id: str,
        system_prompt: str,
        user_message: str,
        max_tokens: int,
        temperature: float,
        chart_image_b64: Optional[str] = None,
    ) -> str:
        try:
            # Guard: if no API key, fall back to Gemini immediately
            if self.claude_client is None:
                logger.warning(f"Claude unavailable for {model_id}, falling back to Gemini")
                return self._generate_gemini(
                    model_id=self.default_model_id,
                    system_prompt=system_prompt,
                    user_message=user_message,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    chart_image_b64=chart_image_b64,
                )

            content = []

            if chart_image_b64:
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": chart_image_b64,
                    },
                })

            content.append({"type": "text", "text": user_message})

            response = self.claude_client.messages.create(
                model=model_id,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": content}],
                timeout=45.0,
            )

            return response.content[0].text

        except Exception as e:
            logger.error(f"Claude API error ({model_id}): {e}")
            logger.warning("Falling back to Gemini default model...")
            return self._generate_gemini(
                model_id=self.default_model_id,
                system_prompt=system_prompt,
                user_message=user_message,
                max_tokens=max_tokens,
                temperature=temperature,
                chart_image_b64=chart_image_b64,
            )

    def _generate_openai(
        self,
        model_id: str,
        system_prompt: str,
        user_message: str,
        max_tokens: int,
        temperature: float,
        chart_image_b64: Optional[str] = None,
        client_override: Optional[OpenAI] = None,
    ) -> str:
        is_groq = (client_override is not None and getattr(self, "_groq_client", None) == client_override) or "groq" in str(client_override)
        try:
            client = client_override or self.openai_client
            # [FIX SILENT-2] Guard: if no API key, fall back immediately
            if client is None:
                if is_groq:
                    logger.warning(f"Groq API unavailable for {model_id}, failing gracefully")
                    return ""
                logger.warning(f"AI Provider unavailable for {model_id}, falling back to Gemini")
                return self._generate_gemini(
                    model_id=self.default_model_id,
                    system_prompt=system_prompt,
                    user_message=user_message,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    chart_image_b64=chart_image_b64,
                )

            messages = []
            
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
                
            user_content = []
            if chart_image_b64:
                # OpenAI vision format
                mime = mimetypes.guess_type("chart.png")[0] or "image/png"
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime};base64,{chart_image_b64}"
                    }
                })
            
            user_content.append({"type": "text", "text": user_message})
            messages.append({"role": "user", "content": user_content})

            kwargs = {
                "model": model_id,
                "messages": messages,
            }
            if not model_id.startswith("o"):
                # "o1-" or "o3-" models don't support temperature/max_tokens normally, 
                # but Groq o-series proxies (like gpt-oss-20b) might technically accept it depending on the proxy. 
                # Safest is to try and pass it for Groq since it acts like standard inference.
                kwargs["temperature"] = temperature
                kwargs["max_tokens"] = max_tokens
            # Ensure timeout isn't outrageously long for quick fallback
            kwargs["timeout"] = 25.0 if is_groq else 45.0

            response = client.chat.completions.create(**kwargs)

            return response.choices[0].message.content or ""

        except Exception as e:
            logger.error(f"OpenAI/Groq API error ({model_id}): {e}")
            if is_groq:
                # Fallback to Cloudflare for text, else fail gracefully to prevent premium token usage.
                if chart_image_b64:
                    logger.warning("Groq Vision failed (rate limit or down). Failing gracefully to avoid premium tokens.")
                    return ""
                
                logger.warning("Falling back to Cloudflare Workers AI due to Groq error...")
                cf_result = self._cf_generator.generate(
                    system_prompt=system_prompt,
                    user_message=user_message,
                    max_tokens=max_tokens
                )
                if cf_result:
                    return cf_result
                logger.warning("Cloudflare also failed. Failing gracefully to avoid premium tokens.")
                return ""

            logger.warning("Falling back to Gemini default model...")
            return self._generate_gemini(
                model_id=self.default_model_id,
                system_prompt=system_prompt,
                user_message=user_message,
                max_tokens=max_tokens,
                temperature=temperature,
                chart_image_b64=chart_image_b64,
            )

    def generate_with_context(
        self,
        system_prompt: str,
        conversation_history: List[Dict],
        max_tokens: int = 4000,
        temperature: float = 0.7,
        use_premium: bool = False,
        role: str = "general",
    ) -> str:
        try:
            flat_text = "\n".join([
                f"{msg.get('role', 'user')}: {msg.get('content', '')}"
                for msg in conversation_history
            ])
            return self.generate_response(
                system_prompt=system_prompt,
                user_message=flat_text,
                max_tokens=max_tokens,
                temperature=temperature,
                use_premium=use_premium,
                role=role,
            )
        except Exception as e:
            logger.error(f"AI API error: {e}")
            return ""

# The global singleton instance
ai_client = AIClient()
