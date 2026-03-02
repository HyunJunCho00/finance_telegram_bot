import requests
from typing import Dict, List, Any, Optional
from config.settings import settings
from loguru import logger
import json

class ExaCollector:
    """Exa Neural Search collector for Deep Alpha Discovery (Credits: $20).
    
    Exa is used for semantic/similarity search. It's great for finding:
    - Technical research papers/PDFs
    - Deep Twitter/Substack threads
    - Niche developer docs
    """
    BASE_URL = "https://api.exa.ai/search"

    def __init__(self):
        self.api_key = settings.EXA_API_KEY

    def search(self, query: str, num_results: int = 3, type: str = "neural", use_autoprompt: bool = True) -> Dict[str, Any]:
        """Execute a neural or keyword search via Exa.
        
        Cost Strategy (2026):
        - Neural ($): $5 per 1-25 results. Use for conceptual queries.
        - Keyword ($$): $2.5 per 1-25 results. Use for specific names/IDs.
        - num_results: Keep <= 5 to avoid higher tier pricing.
        """
        if not self.api_key:
            logger.warning("EXA_API_KEY not set, skipping Exa search")
            return {"results": [], "status": "no_key"}

        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json"
        }
        
        # Guardrail: Never waste credits on large result sets
        safe_num = min(num_results, 10)

        payload = {
            "query": query,
            "type": type, # neural | keyword
            "use_autoprompt": use_autoprompt,
            "num_results": safe_num,
            "highlights": {"numSentences": 2}, # Efficient snippets
        }

        try:
            response = requests.post(self.BASE_URL, headers=headers, json=payload, timeout=20)
            response.raise_for_status()
            data = response.json()
            
            results = data.get("results", [])
            formatted = []
            for r in results:
                formatted.append({
                    "title": r.get("title"),
                    "url": r.get("url"),
                    "score": r.get("score"),
                    "highlight": r.get("highlights", [""])[0] if r.get("highlights") else ""
                })

            return {
                "results": formatted,
                "status": "ok",
                "request_id": data.get("request_id")
            }
        except Exception as e:
            logger.error(f"Exa search error: {e}")
            return {"results": [], "status": "error", "error": str(e)}

    def find_similar(self, url: str) -> Dict[str, Any]:
        """Find contents similar to a specific high-alpha URL."""
        if not self.api_key: return {"results": [], "status": "no_key"}
        
        endpoint = "https://api.exa.ai/findSimilar"
        headers = {"x-api-key": self.api_key, "Content-Type": "application/json"}
        payload = {"url": url, "num_results": 5}
        
        try:
            response = requests.post(endpoint, headers=headers, json=payload, timeout=20)
            response.raise_for_status()
            data = response.json()
            return {"results": data.get("results", []), "status": "ok"}
        except Exception as e:
            logger.error(f"Exa find_similar error: {e}")
            return {"results": [], "status": "error"}

exa_collector = ExaCollector()
