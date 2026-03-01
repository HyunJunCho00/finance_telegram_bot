import requests
from typing import Dict, List, Any, Optional
from config.settings import settings
from loguru import logger
import json

class TavilyCollector:
    BASE_URL = "https://api.tavily.com/search"

    def __init__(self):
        self.api_key = settings.TAVILY_API_KEY

    def search(self, query: str, search_depth: str = "basic", max_results: int = 5, include_domains: Optional[List[str]] = None) -> Dict[str, Any]:
        """Execute a Tavily search and return standardized results."""
        if not self.api_key:
            logger.warning("TAVILY_API_KEY not set, returning empty search results")
            return {"results": [], "answer": "", "status": "no_key"}

        payload = {
            "api_key": self.api_key,
            "query": query,
            "search_depth": search_depth,
            "include_answer": True,
            "max_results": max_results,
        }
        if include_domains:
            payload["include_domains"] = include_domains

        try:
            response = requests.post(self.BASE_URL, json=payload, timeout=20)
            response.raise_for_status()
            data = response.json()
            
            # Standardize output for RAG triangulation
            return {
                "results": data.get("results", []),
                "answer": data.get("answer", ""),
                "status": "ok"
            }
        except Exception as e:
            logger.error(f"Tavily search error: {e}")
            return {"results": [], "answer": "", "status": "error", "error": str(e)}

    def search_targeted_compat(self, entity: str, context: str = "", search_depth: str = "advanced") -> Dict[str, Any]:
        """Compatibility wrapper for Perplexity's search_targeted logic."""
        query = f"{entity} crypto news verification. {context}"
        
        # If advanced (high priority), restrict to trusted domains
        include_domains = settings.TRUSTED_NEWS_DOMAINS if search_depth == "advanced" else None
        
        data = self.search(query, search_depth=search_depth, max_results=5, include_domains=include_domains)
        
        if data["status"] != "ok":
            return {
                "entity": entity,
                "status": data["status"],
                "summary": "Tavily search failed.",
                "btc_eth_impact": "",
                "key_facts": [],
                "market_relevance": "",
                "sources": []
            }

        # Transform Tavily results to targeted search schema
        results = data["results"]
        sources = [r.get("url") for r in results if r.get("url")][:5]
        
        # We use the 'answer' as the summary
        summary = data.get("answer") or "No direct answer found, check sources."
        
        # Extract snippets as key facts
        key_facts = []
        for r in results:
            content = r.get("content", "")
            if content:
                # Basic cleaning of snippets
                clean_fact = content[:300].replace("\n", " ").strip()
                key_facts.append(clean_fact)

        # Calculate trust_score: 0-100
        # 1. Source Authority (up to 70 pts)
        trusted_count = 0
        for s in sources:
            if any(domain in s for domain in settings.TRUSTED_NEWS_DOMAINS):
                trusted_count += 1
        
        source_score = min(70, trusted_count * 20) # 3-4 trusted sources = max score
        
        # 2. Fact Consistency (up to 30 pts)
        fact_score = 30 if len(key_facts) >= 3 else (len(key_facts) * 10)
        
        trust_score = source_score + fact_score

        return {
            "entity": entity,
            "status": "ok",
            "summary": summary,
            "btc_eth_impact": "See summary and sources for impact details.",
            "key_facts": key_facts[:5],
            "market_relevance": "Detected via Tavily triangulation.",
            "sources": sources,
            "trust_score": trust_score
        }

tavily_collector = TavilyCollector()
