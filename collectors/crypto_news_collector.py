import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import List, Dict
from loguru import logger
from datetime import datetime, timezone
from processors.light_rag import light_rag

class CryptoNewsCollector:
    def __init__(self):
        self.base_url = "https://cryptocurrency.cv/api/news"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        self.session = requests.Session()
        
        # Robust retry logic for API fetch
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
    
    def fetch_news(self, categories: List[str] = ["bitcoin", "ethereum", "macro"], limit: int = 10, lang: str = "en") -> List[Dict]:
        results = []
        for cat in categories:
            try:
                r = self.session.get(
                    self.base_url,
                    params={"category": cat, "limit": limit, "lang": lang},
                    headers=self.headers,
                    timeout=15.0
                )
                r.raise_for_status()
                data = r.json()
                articles = data.get("articles", [])
                results.extend(articles)
            except Exception as e:
                logger.warning(f"CryptoNewsCollector error fetching {cat}: {e}")
            time.sleep(1) # Be polite to the free API
        return results

    def fetch_and_ingest(self, categories: List[str] = ["bitcoin", "ethereum", "macro", "trading", "institutional", "layer2", "defi"]):
        """Fetches news and ingests it into LightRAG."""
        logger.info(f"Fetching Free Crypto News API for categories: {categories}")
        articles = self.fetch_news(categories=categories, limit=10)
        
        if not articles:
            logger.info("No news collected from Crypto News API.")
            return

        # Deduplicate
        seen_links = set()
        unique_articles = []
        for a in articles:
            link = a.get("link", "")
            if link and link not in seen_links:
                seen_links.add(link)
                unique_articles.append(a)

        # Format into a text block
        formatted_news = []
        for a in unique_articles:
            source = a.get("source", "NewsAPI")
            cat = a.get("category", "general")
            title = a.get("title", "")
            desc = a.get("description", "")
            formatted_news.append(f"[{source}|{cat}] {title}\n{desc}")

        full_text = "\n\n---\n\n".join(formatted_news)
        
        if full_text.strip():
            logger.info(f"Ingesting {len(unique_articles)} articles from Crypto News API into LightRAG.")
            try:
                light_rag.ingest_message(
                    text=full_text,
                    channel="CRYPTO_NEWS_API",
                    timestamp=datetime.now(timezone.utc).isoformat()
                )
            except Exception as e:
                logger.error(f"Error ingesting Crypto News into RAG: {e}")

collector = CryptoNewsCollector()
