"""Hybrid AI client with role-based model routing.

Key goals:
- Keep Judge on strongest reasoning model.
- Use faster/cheaper models for high-frequency agents.
- Apply soft input caps per role to improve token efficiency.
"""

import base64
from typing import Dict, List, Optional, Tuple

import vertexai
from anthropic import AnthropicVertex
from loguru import logger
from vertexai.generative_models import GenerativeModel, Image, Part

from config.settings import settings


class AIClient:
    def __init__(self):
        vertexai.init(project=settings.PROJECT_ID, location=settings.REGION)

        self._gemini_models: Dict[str, GenerativeModel] = {}
        self._claude_client = None

        # Backward-compatible defaults
        self.default_model_id = "gemini-2.5-flash"
        self.premium_model_id = settings.MODEL_JUDGE

    @property
    def claude_client(self) -> AnthropicVertex:
        if self._claude_client is None:
            self._claude_client = AnthropicVertex(
                region=settings.REGION,
                project_id=settings.PROJECT_ID,
            )
        return self._claude_client

    def _get_gemini_model(self, model_id: str) -> GenerativeModel:
        if model_id not in self._gemini_models:
            self._gemini_models[model_id] = GenerativeModel(model_id)
        return self._gemini_models[model_id]

    def _get_role_model_and_cap(self, role: str, use_premium: bool) -> Tuple[str, int]:
        role = (role or "general").lower()

        if use_premium or role == "judge":
            return settings.MODEL_JUDGE, settings.MAX_INPUT_CHARS_JUDGE
        if role == "bullish":
            return settings.MODEL_BULLISH, settings.MAX_INPUT_CHARS_BULLISH
        if role == "bearish":
            return settings.MODEL_BEARISH, settings.MAX_INPUT_CHARS_BEARISH
        if role == "risk":
            return settings.MODEL_RISK, settings.MAX_INPUT_CHARS_RISK
        if role in ("self_correction", "feedback"):
            return settings.MODEL_SELF_CORRECTION, settings.MAX_INPUT_CHARS_SELF_CORRECTION
        if role in ("rag", "rag_extraction"):
            return settings.MODEL_RAG_EXTRACTION, settings.MAX_INPUT_CHARS_RAG_EXTRACTION
        return self.default_model_id, settings.MAX_INPUT_CHARS_BULLISH

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
        try:
            model = self._get_gemini_model(model_id)
            parts = []

            if chart_image_b64:
                image_bytes = base64.b64decode(chart_image_b64)
                parts.append(Part.from_image(Image.from_bytes(image_bytes)))

            parts.append(Part.from_text(user_message))

            response = model.generate_content(
                contents=parts,
                generation_config={
                    "max_output_tokens": max_tokens,
                    "temperature": temperature,
                },
                system_instruction=system_prompt,
            )
            return response.text

        except Exception as e:
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


claude_client = AIClient()
