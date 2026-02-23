import time
import functools
import requests
from loguru import logger

def api_retry(max_attempts=3, delay_seconds=2, backoff=2, exceptions=(requests.exceptions.RequestException, ValueError)):
    """
    A robust retry decorator designed for API calls to prevent pipeline failures 
    due to rate limits (HTTP 429) or transient network issues.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            current_delay = delay_seconds
            
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    attempts += 1
                    if attempts == max_attempts:
                        logger.error(f"API Retry failed after {max_attempts} attempts for {func.__name__}. Error: {e}")
                        return None  # Or raise if pipeline must halt
                    
                    logger.warning(f"API Error in {func.__name__}: {e}. Retrying {attempts}/{max_attempts} in {current_delay}s...")
                    time.sleep(current_delay)
                    current_delay *= backoff
        return wrapper
    return decorator
