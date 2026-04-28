import json
import logging
from typing import Type, TypeVar, Optional
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

class Guardrail:
    """
    A unified LLM Guardrail that enforces Pydantic schemas on AI outputs.
    If the LLM outputs invalid formats or breaks business rules (e.g., leverage > 3),
    this guardrail catches the ValidationError and auto-prompts the AI to fix it.
    """
    
    @staticmethod
    def validate_with_retry(
        ai_client_func,
        system_prompt: str,
        user_message: str,
        response_model: Type[T],
        max_retries: int = 3,
        **kwargs
    ) -> Optional[T]:
        """
        Executes an AI call and validates the output strictly against the response_model.
        Provides feedback to the LLM if validation fails.
        """
        schema_json = json.dumps(response_model.model_json_schema(), ensure_ascii=False, indent=2)
        
        # Inject the schema constraint into the prompt
        enforced_prompt = (
            f"{system_prompt}\n\n"
            f"--- STRICT GUARDRAIL RULES ---\n"
            f"You MUST output raw JSON that exactly satisfies this Pydantic schema:\n"
            f"{schema_json}\n\n"
            f"Do not include markdown blocks, explanations, or text outside the JSON."
        )
        
        current_user_msg = user_message
        
        for attempt in range(1, max_retries + 1):
            try:
                # 1. Call the underlying AI model (e.g., Gemini, Groq)
                raw_response = ai_client_func(
                    system_prompt=enforced_prompt,
                    user_message=current_user_msg,
                    **kwargs
                )
                
                # 2. Pre-process the output (strip markdown fences)
                cleaned = raw_response.strip()
                if cleaned.startswith("```"):
                    cleaned = "\n".join(cleaned.split("\n")[1:])
                if cleaned.endswith("```"):
                    cleaned = "\n".join(cleaned.split("\n")[:-1])
                cleaned = cleaned.strip()
                
                # 3. Pydantic Strict Validation
                data_dict = json.loads(cleaned)
                validated_object = response_model.model_validate(data_dict)
                
                if attempt > 1:
                    logger.success(f"🛡️ Guardrail fixed the LLM hallucination on attempt {attempt}!")
                    
                return validated_object
                
            except (json.JSONDecodeError, ValidationError) as e:
                logger.warning(f"🛡️ Guardrail blocked invalid AI output (Attempt {attempt}/{max_retries}): {e}")
                
                if attempt == max_retries:
                    logger.error("❌ Guardrail max retries exceeded. AI failed to comply with schema.")
                    return None
                    
                # 4. Auto-correction Prompting
                current_user_msg = (
                    f"{user_message}\n\n"
                    f"--- VALIDATION ERROR IN PREVIOUS RESPONSE ---\n"
                    f"Your last output failed schema validation with the following error:\n{e}\n"
                    f"Please fix the data types, constraints, and JSON syntax and try again."
                )
            except Exception as e:
                logger.error(f"Unexpected error in guardrail: {e}")
                return None
                
        return None
