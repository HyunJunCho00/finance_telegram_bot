"""Hybrid AI Client: Gemini Flash (agents) + Claude Opus 4.6 (Judge).

Cost strategy ($30/month budget):
- Bull/Bear/Risk agents -> Gemini 2.0 Flash ($0.15/$0.60 per M tokens)
- Judge agent -> Claude Opus 4.6 via Vertex AI Model Garden
  - Best reasoning model for final trading decisions
  - ~$15/$75 per M tokens (estimated, Vertex pricing)
- Chart image -> Judge only, SWING mode only (512x512 ~1024 tokens)

Monthly estimate:
  Flash agents: 180 cycles x 3 agents x ~3K tokens = ~1.6M tokens -> ~$1.2/month
  Claude Judge: 180 cycles x ~5K tokens = ~900K tokens -> ~$15-20/month (input heavy)
  VM e2-small: $12/month
  Perplexity: $5/month
  Total: ~$25-30/month

Architecture:
  - Gemini via vertexai SDK (native)
  - Claude via anthropic SDK with AnthropicVertex (Model Garden)
"""

import vertexai
from vertexai.generative_models import GenerativeModel, Part, Image
from anthropic import AnthropicVertex
from config.settings import settings
from typing import List, Dict, Optional
from loguru import logger
import base64
import json


class AIClient:
    def __init__(self):
        vertexai.init(
            project=settings.PROJECT_ID,
            location=settings.REGION
        )

        # ── Gemini Flash: cost-efficient for bull/bear/risk agents ──
        self.default_model_id = "gemini-2.0-flash-001"
        self._default_model = None

        # ── Claude Opus 4.6: best reasoning for Judge decisions ──
        self.premium_model_id = "claude-opus-4-20250918"
        self._claude_client = None

    @property
    def default_model(self) -> GenerativeModel:
        if self._default_model is None:
            self._default_model = GenerativeModel(self.default_model_id)
        return self._default_model

    @property
    def claude_client(self) -> AnthropicVertex:
        if self._claude_client is None:
            self._claude_client = AnthropicVertex(
                region=settings.REGION,
                project_id=settings.PROJECT_ID,
            )
        return self._claude_client

    def generate_response(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: int = 4000,
        temperature: float = 0.7,
        chart_image_b64: Optional[str] = None,
        use_premium: bool = False,
    ) -> str:
        if use_premium:
            return self._generate_claude(
                system_prompt, user_message, max_tokens,
                temperature, chart_image_b64
            )
        else:
            return self._generate_gemini(
                system_prompt, user_message, max_tokens,
                temperature, chart_image_b64
            )

    def _generate_gemini(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: int,
        temperature: float,
        chart_image_b64: Optional[str] = None,
    ) -> str:
        try:
            parts = []

            if chart_image_b64:
                image_bytes = base64.b64decode(chart_image_b64)
                parts.append(Part.from_image(Image.from_bytes(image_bytes)))

            parts.append(Part.from_text(user_message))

            response = self.default_model.generate_content(
                contents=parts,
                generation_config={
                    "max_output_tokens": max_tokens,
                    "temperature": temperature,
                },
                system_instruction=system_prompt,
            )

            return response.text

        except Exception as e:
            logger.error(f"Gemini API error ({self.default_model_id}): {e}")
            return ""

    def _generate_claude(
        self,
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
                    }
                })

            content.append({
                "type": "text",
                "text": user_message,
            })

            response = self.claude_client.messages.create(
                model=self.premium_model_id,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": content}],
            )

            return response.content[0].text

        except Exception as e:
            logger.error(f"Claude API error ({self.premium_model_id}): {e}")
            # Fallback to Gemini Pro if Claude fails
            logger.warning("Falling back to Gemini for Judge...")
            try:
                return self._generate_gemini(
                    system_prompt, user_message, max_tokens,
                    temperature, chart_image_b64
                )
            except Exception as e2:
                logger.error(f"Gemini fallback also failed: {e2}")
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
            if use_premium:
                # Claude format
                messages = []
                for msg in conversation_history:
                    role = msg["role"]
                    if role not in ("user", "assistant"):
                        role = "user"
                    if isinstance(msg["content"], str):
                        messages.append({"role": role, "content": msg["content"]})
                    elif isinstance(msg["content"], list):
                        messages.append({"role": role, "content": msg["content"]})

                response = self.claude_client.messages.create(
                    model=self.premium_model_id,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system_prompt,
                    messages=messages,
                )
                return response.content[0].text
            else:
                # Gemini format
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

                response = self.default_model.generate_content(
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
