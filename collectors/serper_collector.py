import requests
from typing import Dict, List, Any, Optional
from config.settings import settings
from loguru import logger
import json

class SerperCollector:
    """Google SERP collector for precise link verification (Free: 2,500 queries).
    
    Used as a sniper for verifying specific Telegram rumors or official news.
    """
    BASE_URL = "https://google.serper.dev/search"

    def __init__(self):
        self.api_key = settings.SERPER_API_KEY

    def search(self, query: str, num_results: int = 5) -> Dict[str, Any]:
        """Execute a Google Search via Serper.dev."""
        if not self.api_key:
            logger.warning("SERPER_API_KEY not set, skipping Google search")
            return {"results": [], "status": "no_key"}

        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json"
        }
        payload = json.dumps({
            "q": query,
            "num": num_results
        })

        try:
            response = requests.post(self.BASE_URL, headers=headers, data=payload, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            # Organic results mapping
            organic = data.get("organic", [])
            results = []
            for item in organic:
                results.append({
                    "title": item.get("title"),
                    "link": item.get("link"),
                    "snippet": item.get("snippet"),
                    "position": item.get("position")
                })

            return {
                "results": results,
                "status": "ok",
                "metadata": {
                    "search_parameters": data.get("searchParameters"),
                    "credits_spent": 1
                }
            }
        except Exception as e:
            logger.error(f"Serper search error: {e}")
            return {"results": [], "status": "error", "error": str(e)}

    def verify_news(self, news_text: str) -> Dict[str, Any]:
        """Verify a specific piece of news by looking for official sources."""
        # Simple extraction logic: check for keywords
        query = f"official news: {news_text}"
        return self.search(query, num_results=3)

serper_collector = SerperCollector()
