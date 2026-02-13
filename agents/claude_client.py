"""Unified AI Client using GCP Vertex AI.

Cost strategy ($30/month budget):
- Bull/Bear/Risk agents -> Gemini 2.0 Flash ($0.15/$0.60 per M tokens) - fast, cheap
- Judge agent -> Gemini 2.5 Pro ($1.25/$10 per M tokens) - deep analysis
- Chart image -> Judge only, SWING mode only (512x512 ~1024 tokens)

Monthly estimate: ~180 cycles x ~$0.02/cycle = ~$3.6/month for AI
+ VM e2-small $12 + Perplexity $5 = ~$21/month total (well under $30)

Architecture: All through Vertex AI Gemini. Can switch to Claude by changing model IDs.
"""

import vertexai
from vertexai.generative_models import GenerativeModel, Part, Image
from config.settings import settings
from typing import List, Dict, Optional
from loguru import logger
import base64


class AIClient:
    def __init__(self):
        vertexai.init(
            project=settings.PROJECT_ID,
            location=settings.REGION
        )

        # Cost-efficient for bull/bear/risk agents
        self.default_model_id = "gemini-2.0-flash-001"
        # Best analysis for judge decisions
        self.premium_model_id = "gemini-2.5-pro-preview-05-06"

        self._default_model = None
        self._premium_model = None

    @property
    def default_model(self) -> GenerativeModel:
        if self._default_model is None:
            self._default_model = GenerativeModel(self.default_model_id)
        return self._default_model

    @property
    def premium_model(self) -> GenerativeModel:
        if self._premium_model is None:
            self._premium_model = GenerativeModel(self.premium_model_id)
        return self._premium_model

    def generate_response(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: int = 4000,
        temperature: float = 0.7,
        chart_image_b64: Optional[str] = None,
        use_premium: bool = False,
    ) -> str:
        try:
            model = self.premium_model if use_premium else self.default_model

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
            model_name = self.premium_model_id if use_premium else self.default_model_id
            logger.error(f"AI API error ({model_name}): {e}")
            return ""

    def generate_with_context(
        self,
        system_prompt: str,
        conversation_history: List[Dict],
        max_tokens: int = 4000,
        temperature: float = 0.7,
        use_premium: bool = False,
    ) -> str:
        try:
            model = self.premium_model if use_premium else self.default_model

            # Convert conversation history to Gemini format
            contents = []
            for msg in conversation_history:
                role = "user" if msg["role"] == "user" else "model"
                if isinstance(msg["content"], str):
                    contents.append({"role": role, "parts": [{"text": msg["content"]}]})
                elif isinstance(msg["content"], list):
                    parts = []
                    for item in msg["content"]:
                        if item.get("type") == "text":
                            parts.append({"text": item["text"]})
                    contents.append({"role": role, "parts": parts})

            response = model.generate_content(
                contents=contents,
                generation_config={
                    "max_output_tokens": max_tokens,
                    "temperature": temperature,
                },
                system_instruction=system_prompt,
            )

            return response.text

        except Exception as e:
            logger.error(f"AI API error: {e}")
            return ""


# Keep the old variable name for backward compatibility with all agents
claude_client = AIClient()
