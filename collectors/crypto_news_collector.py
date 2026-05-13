import feedparser
import sqlite3
import hashlib
import json
import time
import re
import html
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional
from loguru import logger
from processors.light_rag import light_rag

# 공신력 순 정렬: 전통 금융 권위 > 크립토 기관급 > 데이터/리서치 > 커뮤니티
RSS_FEEDS = {
    "Reuters Crypto":   "https://feeds.reuters.com/reuters/technologyNews",   # 전통 금융 최고 권위
    "CoinDesk":         "https://www.coindesk.com/arc/outboundfeeds/rss/?outputType=xml",  # 크립토 최고참
    "The Block":        "https://www.theblock.co/rss.xml",                    # 기관급 크립토
    "Blockworks":       "https://blockworks.co/feed",                         # 기관 투자자 타겟
    "Messari":          "https://messari.io/rss",                             # 데이터 기반 리서치
    "DL News":          "https://www.dlnews.com/rss/",                        # 심층 취재
    "Decrypt":          "https://decrypt.co/feed",                            # 광범위 커버리지
    "Bitcoin Magazine": "https://bitcoinmagazine.com/.rss/full/",             # BTC 특화
    "The Defiant":      "https://thedefiant.io/api/feed",                     # DeFi 특화
    "Protos":           "https://protos.com/feed/",                           # 비판적 저널리즘
}

# 2. 분류 키워드
TAG_KEYWORDS = {
    "bitcoin": ["bitcoin", "btc", "strategic reserve", "microstrategy", "mstr", "etf flow", "bitvm", "runes", "l2", "sovereign wealth"],
    # 26년 메타: RWA(실물자산 토큰화), BlackRock BUIDL, 금융 인프라망 편입, 페트라 업그레이드, 리스테이킹
    "ethereum": ["ethereum", "eth", "pectra", "restaking", "eigenlayer", "etf staking", "blob", "eip", "lrt", "symbiotic", "rwa", "tokenization", "buidl", "infrastructure", "smart contract"],
    # 26년 메타: Clarity Act(명확성 법안), FIT21, SAB 121 폐지, 스테이블코인 법안, 금리 인하 사이클, 관세 전쟁
    "macro": ["sec", "cftc", "fed", "powell", "rate cut", "inflation", "cpi", "pce", "tariff", "blackrock", "treasury", "election", "regulation", "clarity act", "fit21", "sab 121", "stablecoin bill"],
    "institutional": ["institutional", "pension fund", "grayscale", "fidelity", "state wisconsin", "investment board", "etf"],
    "layer2": ["layer 2", "l2", "arbitrum", "optimism", "base", "zksync", "starknet", "scroll", "linea"],
    "defi": ["defi", "dex", "yield", "tvl", "uniswap", "aave", "maker", "ena", "ethena", "pendle", "jup"],
    "hack": ["hack", "exploit", "breach", "stolen", "vulnerability", "rugpull", "scam", "fraud"]
}

class CryptoNewsCollector:
    def __init__(self):
        # 봇의 데이터 폴더 내에 SQLite DB 저장
        self.db_path = Path("data/crypto_news.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS articles (
                id           TEXT PRIMARY KEY,
                title        TEXT NOT NULL,
                url          TEXT UNIQUE NOT NULL,
                source       TEXT,
                summary      TEXT,
                published_at TEXT,
                collected_at TEXT,
                tags         TEXT,
                raw_json     TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_published ON articles(published_at DESC)")
        conn.commit()
        conn.close()

    def _clean_text(self, raw_text: str) -> str:
        """HTML 태그 제거 및 특수문자 디코딩을 수행하여 깔끔한 문장으로 만듭니다."""
        if not raw_text:
            return ""
        # 1. HTML 엔티티 디코딩 (&quot; -> ", &#8216; -> ')
        text = html.unescape(raw_text)
        # 2. 모든 HTML 태그(<p>, <a>, <img> 등) 완전 삭제
        text = re.sub(r'<[^>]+>', '', text)
        # 3. 연속된 공백이나 줄바꿈을 띄어쓰기 하나로 압축
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _tag_article(self, title: str, summary: str) -> List[str]:
        text = (title + " " + summary).lower()
        return [tag for tag, kws in TAG_KEYWORDS.items() if any(kw in text for kw in kws)]

    def fetch_rss_feeds(self) -> List[Dict]:
        """RSS를 파싱하고 DB에 저장한 뒤, '새로 추가된' 기사들만 반환합니다."""
        conn = sqlite3.connect(str(self.db_path))
        new_articles = []

        for source, url in RSS_FEEDS.items():
            try:
                feed = feedparser.parse(url, agent="CryptoNewsCollector/2.0")
                
                for entry in feed.entries:
                    link = entry.get("link", "")
                    if not link:
                        continue
                        
                    title = self._clean_text(entry.get("title", ""))
                    summary_raw = entry.get("summary", entry.get("description", ""))
                    summary = self._clean_text(summary_raw) # 길이 제한 해제 (원본 그대로 보존)
                    pub = entry.get("published", entry.get("updated", ""))
                    tags = self._tag_article(title, summary)
                    
                    article_id = hashlib.md5(link.encode()).hexdigest()
                    
                    # DB 저장 (INSERT OR IGNORE)
                    try:
                        cursor = conn.execute("""
                            INSERT OR IGNORE INTO articles
                                (id, title, url, source, summary, published_at, collected_at, tags, raw_json)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            article_id, title, link, source, summary, pub,
                            datetime.now(timezone.utc).isoformat(),
                            json.dumps(tags, ensure_ascii=False),
                            json.dumps({"title": title, "link": link}, ensure_ascii=False)
                        ))
                        
                        # 변화가 있다면(새로운 기사라면) 리스트에 추가
                        if cursor.rowcount > 0:
                            new_articles.append({
                                "source": source,
                                "title": title,
                                "url": link,
                                "summary": summary,
                                "tags": tags
                            })
                    except sqlite3.Error as e:
                        logger.warning(f"DB insert error: {e}")
                        
            except Exception as e:
                logger.warning(f"[{source}] feed parse error: {e}")
                
            time.sleep(0.5)  # 소스 간 딜레이

        conn.commit()
        conn.close()
        return new_articles

    def fetch_news(self, categories: List[str] = None, limit: int = 10, lang: str = "en") -> List[Dict]:
        """Query recent articles from SQLite with per-source cap for diversity."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            rows = conn.execute(
                "SELECT title, url, source, summary, tags FROM articles ORDER BY published_at DESC LIMIT 500"
            ).fetchall()
        finally:
            conn.close()

        MAX_PER_SOURCE = max(2, limit // len(RSS_FEEDS) + 1)
        source_counts: dict = {}
        results = []
        for title, url, source, summary, tags_json in rows:
            if len(results) >= limit:
                break
            if source_counts.get(source, 0) >= MAX_PER_SOURCE:
                continue
            if categories:
                try:
                    tags = json.loads(tags_json or "[]")
                except Exception:
                    tags = []
                if not any(t in categories for t in tags):
                    continue
            results.append({"link": url, "source": source, "title": title, "description": summary or ""})
            source_counts[source] = source_counts.get(source, 0) + 1
        return results

    def fetch_and_ingest(self, categories: List[str] = None):
        """기존 API를 완벽히 대체하는 메인 함수. cloud_jobs에서 1시간마다 호출됩니다."""
        logger.info("Fetching Crypto News via RSS Feeds...")
        
        # 1. RSS에서 뉴스 긁어오고 SQLite에 영구 보존
        new_articles = self.fetch_rss_feeds()
        
        if not new_articles:
            logger.info("No new articles from RSS since last run.")
            return

        # 2. LightRAG 주입을 위한 포매팅
        formatted_news = []
        for a in new_articles:
            if categories and not any(tag in categories for tag in a["tags"]):
                continue
                
            tags_str = ",".join(a["tags"]) if a["tags"] else "general"
            formatted_news.append(f"[{a['source']}|{tags_str}] {a['title']}\n{a['summary']}")

        if not formatted_news:
            logger.info("No articles matched the required categories.")
            return

        full_text = "\n\n---\n\n".join(formatted_news)
        
        # 3. LightRAG 지식 그래프에 뉴스 밀어넣기
        logger.info(f"Ingesting {len(formatted_news)} new RSS articles into LightRAG.")
        try:
            light_rag.ingest_message(
                text=full_text,
                channel="CRYPTO_NEWS_RSS",
                timestamp=datetime.now(timezone.utc).isoformat()
            )
        except Exception as e:
            logger.error(f"Error ingesting Crypto News into RAG: {e}")

        # 새로 수집된 기사 반환 (entrypoint에서 텔레그램 발송에 사용)
        return new_articles

# 전역 인스턴스 노출 (entrypoint.py 에서 import collector 로 사용)
collector = CryptoNewsCollector()


