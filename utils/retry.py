import time
import functools
from loguru import logger


def api_retry(max_attempts=3, delay_seconds=2, backoff=2):
    """
    Retry decorator for ALL API calls (Anthropic, Gemini, OpenAI, Supabase, etc.).

    [FIX] Previously caught only requests.exceptions.RequestException and ValueError,
    which missed SDK-specific exceptions:
      - anthropic.APIError / anthropic.RateLimitError
      - google.api_core.exceptions.GoogleAPIError
      - openai.APIError / openai.RateLimitError
    Now catches any Exception so all SDK transient errors are properly retried.

    On final failure returns None so the pipeline continues with fallback data
    rather than crashing the whole analysis run.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            current_delay = delay_seconds

            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    if attempts == max_attempts:
                        logger.error(
                            f"[retry] {func.__name__} failed after {max_attempts} "
                            f"attempts. Last error ({type(e).__name__}): {e}"
                        )
                        return None

                    logger.warning(
                        f"[retry] {func.__name__} error ({type(e).__name__}): {e}. "
                        f"Retrying {attempts}/{max_attempts} in {current_delay}s..."
                    )
                    time.sleep(current_delay)
                    current_delay *= backoff

        return wrapper
    return decorator
