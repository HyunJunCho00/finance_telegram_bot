"""LightRAG: 정석 Graph RAG implementation.

Architecture (LightRAG 정석):
1. Graph Extraction (LLM 기반): Gemini Flash로 Entity + Relationship 추출
2. Dual-level Indexing:
   - Low-level: 구체적 entity + 직접 관계 (Local retrieval)
   - High-level: 요약된 topic + community (Global retrieval)
3. Storage:
   - Neo4j Aura (free tier): Knowledge Graph 저장
   - Zilliz Cloud (free tier): Vector embeddings 저장
   - Supabase (PostgreSQL): 원본 텍스트 + 메타데이터
4. Incremental Update: 새 데이터는 기존 그래프에 병합 (전체 리빌드 불필요)

Cost:
  - Neo4j Aura Free: 200K nodes, 400K relationships (충분)
  - Zilliz Cloud Free: 1M vectors, 2 collections (충분)
  - Gemini Flash (extraction): ~$0.5/month
  - Gemini Embedding: ~$0.05/month
"""

import re
import hashlib
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
from loguru import logger
import json

# Gemini embedding (google-genai SDK)
try:
    from google import genai
    from google.genai import types
    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False
    logger.warning("google-genai embedding not available")

# Neo4j
try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    logger.warning("neo4j driver not available, using in-memory graph fallback")

# Milvus (Zilliz Cloud)
try:
    from pymilvus import MilvusClient
    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False
    logger.warning("pymilvus not available, using in-memory vector fallback")


class Neo4jGraph:
    """Neo4j Aura Free tier knowledge graph.

    Stores entities and relationships extracted by LLM.
    Supports:
    - Entity upsert with description merging
    - Relationship upsert with weight accumulation
    - Local retrieval (entity neighbors, 2-hop BFS)
    - Global retrieval (community detection via entity importance)
    """

    def __init__(self, uri: str = "", user: str = "neo4j", password: str = ""):
        self._driver = None
        self._uri = uri
        self._user = user
        self._password = password

    @property
    def driver(self):
        if self._driver is None and NEO4J_AVAILABLE and self._uri:
            try:
                self._driver = GraphDatabase.driver(
                    self._uri, auth=(self._user, self._password)
                )
                self._driver.verify_connectivity()
                self._ensure_indexes()
                logger.info("Neo4j connected")
            except Exception as e:
                logger.error(f"Neo4j connection failed: {e}")
                self._driver = None
        return self._driver

    def _ensure_indexes(self):
        """Create indexes for fast lookups."""
        with self.driver.session() as session:
            session.run("CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)")
            session.run("CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)")

    def upsert_entity(self, name: str, entity_type: str, description: str,
                      source_id: str = "", timestamp: str = ""):
        """Upsert entity node. Merges descriptions on conflict."""
        if not self.driver:
            return
        try:
            with self.driver.session() as session:
                session.run("""
                    MERGE (e:Entity {name: $name})
                    ON CREATE SET
                        e.type = $type,
                        e.description = $description,
                        e.mention_count = 1,
                        e.first_seen = $timestamp,
                        e.last_seen = $timestamp
                    ON MATCH SET
                        e.description = CASE
                            WHEN size(e.description) < 500
                            THEN e.description + ' | ' + $description
                            ELSE $description
                        END,
                        e.mention_count = e.mention_count + 1,
                        e.last_seen = $timestamp
                """, name=name.lower(), type=entity_type, description=description[:300],
                     timestamp=timestamp)
        except Exception as e:
            logger.error(f"Neo4j upsert_entity error: {e}")

    def upsert_relationship(self, source: str, target: str, rel_type: str,
                            description: str = "", weight: float = 1.0,
                            timestamp: str = "", source_id: str = ""):
        """Upsert relationship edge. Accumulates weight on conflict."""
        if not self.driver:
            return
        try:
            with self.driver.session() as session:
                session.run("""
                    MERGE (s:Entity {name: $source})
                    MERGE (t:Entity {name: $target})
                    MERGE (s)-[r:RELATES_TO {type: $rel_type}]->(t)
                    ON CREATE SET
                        r.description = $description,
                        r.weight = $weight,
                        r.first_seen = $timestamp,
                        r.last_seen = $timestamp,
                        r.count = 1
                    ON MATCH SET
                        r.description = $description,
                        r.weight = r.weight + $weight,
                        r.last_seen = $timestamp,
                        r.count = r.count + 1
                """, source=source.lower(), target=target.lower(),
                     rel_type=rel_type, description=description[:200],
                     weight=weight, timestamp=timestamp)
        except Exception as e:
            logger.error(f"Neo4j upsert_relationship error: {e}")

    def get_local_context(self, entity: str, max_depth: int = 2, limit: int = 20) -> Dict:
        """Low-level retrieval: BFS from entity, up to max_depth hops."""
        if not self.driver:
            return {"entity": entity, "neighbors": [], "paths": []}
        try:
            with self.driver.session() as session:
                # Direct neighbors (depth 1)
                result = session.run("""
                    MATCH (e:Entity {name: $name})-[r:RELATES_TO]-(n:Entity)
                    RETURN n.name AS neighbor, n.type AS type,
                           r.type AS rel_type, r.description AS rel_desc,
                           r.weight AS weight, r.last_seen AS last_seen,
                           n.description AS desc
                    ORDER BY r.weight DESC, r.last_seen DESC
                    LIMIT $limit
                """, name=entity.lower(), limit=limit)

                neighbors = []
                for record in result:
                    neighbors.append({
                        "name": record["neighbor"],
                        "type": record["type"],
                        "rel_type": record["rel_type"],
                        "rel_desc": record["rel_desc"],
                        "weight": record["weight"],
                        "last_seen": record["last_seen"],
                        "description": record["desc"],
                    })

                # 2-hop paths if needed
                paths = []
                if max_depth >= 2:
                    result2 = session.run("""
                        MATCH (e:Entity {name: $name})-[r1:RELATES_TO]-(n1:Entity)
                              -[r2:RELATES_TO]-(n2:Entity)
                        WHERE n2.name <> $name
                        RETURN n1.name AS via, r1.type AS r1_type,
                               n2.name AS target, r2.type AS r2_type,
                               n2.description AS desc
                        ORDER BY r1.weight + r2.weight DESC
                        LIMIT 10
                    """, name=entity.lower())
                    for record in result2:
                        paths.append({
                            "via": record["via"],
                            "r1_type": record["r1_type"],
                            "target": record["target"],
                            "r2_type": record["r2_type"],
                            "description": record["desc"],
                        })

                return {
                    "entity": entity,
                    "neighbors": neighbors,
                    "paths": paths,
                }
        except Exception as e:
            logger.error(f"Neo4j local context error: {e}")
            return {"entity": entity, "neighbors": [], "paths": []}

    def get_global_context(self, top_k: int = 10) -> Dict:
        """High-level retrieval: most important entities + communities."""
        if not self.driver:
            return {"important_entities": [], "recent_relationships": []}
        try:
            with self.driver.session() as session:
                # Top entities by mention count (PageRank approximation)
                result = session.run("""
                    MATCH (e:Entity)
                    RETURN e.name AS name, e.type AS type,
                           e.mention_count AS mentions, e.description AS desc,
                           e.last_seen AS last_seen
                    ORDER BY e.mention_count DESC
                    LIMIT $limit
                """, limit=top_k)

                important = []
                for record in result:
                    important.append({
                        "name": record["name"],
                        "type": record["type"],
                        "mentions": record["mentions"],
                        "description": record["desc"],
                        "last_seen": record["last_seen"],
                    })

                # Recent high-weight relationships
                result2 = session.run("""
                    MATCH (s:Entity)-[r:RELATES_TO]->(t:Entity)
                    RETURN s.name AS source, r.type AS rel_type,
                           t.name AS target, r.description AS desc,
                           r.weight AS weight, r.last_seen AS last_seen
                    ORDER BY r.last_seen DESC, r.weight DESC
                    LIMIT 15
                """)

                recent_rels = []
                for record in result2:
                    recent_rels.append({
                        "source": record["source"],
                        "rel_type": record["rel_type"],
                        "target": record["target"],
                        "description": record["desc"],
                        "weight": record["weight"],
                        "last_seen": record["last_seen"],
                    })

                return {
                    "important_entities": important,
                    "recent_relationships": recent_rels,
                }
        except Exception as e:
            logger.error(f"Neo4j global context error: {e}")
            return {"important_entities": [], "recent_relationships": []}

    def cleanup_old(self, days: int = 90):
        """Remove entities/relationships older than N days."""
        if not self.driver:
            return
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        try:
            with self.driver.session() as session:
                # Delete old relationships
                session.run("""
                    MATCH ()-[r:RELATES_TO]->()
                    WHERE r.last_seen < $cutoff
                    DELETE r
                """, cutoff=cutoff)
                # Delete orphan entities
                session.run("""
                    MATCH (e:Entity)
                    WHERE e.last_seen < $cutoff
                      AND NOT (e)--()
                    DELETE e
                """, cutoff=cutoff)
            logger.info("Neo4j cleanup completed")
        except Exception as e:
            logger.error(f"Neo4j cleanup error: {e}")

    def get_stats(self) -> Dict:
        if not self.driver:
            return {"nodes": 0, "relationships": 0, "connected": False}
        try:
            with self.driver.session() as session:
                nodes = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
                rels = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
                return {"nodes": nodes, "relationships": rels, "connected": True}
        except Exception:
            return {"nodes": 0, "relationships": 0, "connected": False}


class MilvusVectorStore:
    """Zilliz Cloud (Milvus) free tier vector store.

    Stores document embeddings for semantic search.
    Free tier: 1M vectors, 2 collections.
    """

    COLLECTION_NAME = "crypto_news"
    DIMENSION = 768  # text-embedding-005 dimension

    def __init__(self, uri: str = "", token: str = ""):
        self._client = None
        self._uri = uri
        self._token = token

    @property
    def client(self):
        if self._client is None and MILVUS_AVAILABLE and self._uri:
            try:
                self._client = MilvusClient(uri=self._uri, token=self._token)
                self._ensure_collection()
                logger.info("Milvus connected")
            except Exception as e:
                logger.error(f"Milvus connection failed: {e}")
                self._client = None
        return self._client

    def _ensure_collection(self):
        """Create collection if not exists."""
        if not self._client:
            return
        try:
            if not self._client.has_collection(self.COLLECTION_NAME):
                from pymilvus import CollectionSchema, FieldSchema, DataType
                fields = [
                    FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
                    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.DIMENSION),
                    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2000),
                    FieldSchema(name="channel", dtype=DataType.VARCHAR, max_length=100),
                    FieldSchema(name="timestamp", dtype=DataType.VARCHAR, max_length=50),
                    FieldSchema(name="entities", dtype=DataType.VARCHAR, max_length=500),
                ]
                schema = CollectionSchema(fields=fields, enable_dynamic_field=True)
                self._client.create_collection(
                    collection_name=self.COLLECTION_NAME,
                    schema=schema,
                )
                # Create vector index
                self._client.create_index(
                    collection_name=self.COLLECTION_NAME,
                    field_name="vector",
                    index_params={"metric_type": "COSINE", "index_type": "AUTOINDEX"},
                )
                logger.info(f"Milvus collection '{self.COLLECTION_NAME}' created")
        except Exception as e:
            logger.error(f"Milvus collection setup error: {e}")

    def upsert(self, doc_id: str, embedding: np.ndarray, text: str,
               channel: str = "", timestamp: str = "", entities: List[str] = None):
        """Upsert a document embedding."""
        if not self.client:
            return
        try:
            data = [{
                "id": doc_id,
                "vector": embedding.tolist(),
                "text": text[:2000],
                "channel": channel,
                "timestamp": timestamp,
                "entities": json.dumps(entities or [])[:500],
            }]
            self.client.upsert(collection_name=self.COLLECTION_NAME, data=data)
        except Exception as e:
            logger.error(f"Milvus upsert error: {e}")

    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Dict]:
        """Search for similar documents by embedding."""
        if not self.client:
            return []
        try:
            results = self.client.search(
                collection_name=self.COLLECTION_NAME,
                data=[query_embedding.tolist()],
                limit=top_k,
                output_fields=["text", "channel", "timestamp", "entities"],
            )
            docs = []
            for hit in results[0]:
                entity = hit.get("entity", {})
                docs.append({
                    "id": hit.get("id", ""),
                    "score": hit.get("distance", 0),
                    "text": entity.get("text", ""),
                    "channel": entity.get("channel", ""),
                    "timestamp": entity.get("timestamp", ""),
                    "entities": json.loads(entity.get("entities", "[]")),
                })
            return docs
        except Exception as e:
            logger.error(f"Milvus search error: {e}")
            return []

    def get_stats(self) -> Dict:
        if not self.client:
            return {"vectors": 0, "connected": False}
        try:
            stats = self.client.get_collection_stats(self.COLLECTION_NAME)
            return {"vectors": stats.get("row_count", 0), "connected": True}
        except Exception:
            return {"vectors": 0, "connected": False}


class InMemoryFallback:
    """In-memory fallback when Neo4j/Milvus unavailable (local dev)."""

    def __init__(self):
        self.edges = defaultdict(list)
        self.entity_meta = {}
        self.edge_counts = defaultdict(int)
        self.documents = {}

    def upsert_entity(self, name, entity_type, description, **kwargs):
        ts = kwargs.get("timestamp", "")
        name = name.lower()
        if name not in self.entity_meta:
            self.entity_meta[name] = {"type": entity_type, "desc": description, "count": 0, "first": ts}
        self.entity_meta[name]["count"] += 1
        self.entity_meta[name]["last"] = ts
        self.edge_counts[name] += 1

    def upsert_relationship(self, source, target, rel_type, **kwargs):
        source, target = source.lower(), target.lower()
        self.edges[source].append({
            "target": target, "type": rel_type,
            "desc": kwargs.get("description", ""),
            "timestamp": kwargs.get("timestamp", ""),
        })
        self.edge_counts[source] += 1
        self.edge_counts[target] += 1

    def get_local_context(self, entity, max_depth=2, limit=20):
        entity = entity.lower()
        neighbors = sorted(
            self.edges.get(entity, []),
            key=lambda x: x.get("timestamp", ""), reverse=True
        )[:limit]
        paths = []
        if max_depth >= 2:
            for n in neighbors[:5]:
                for e2 in self.edges.get(n["target"], [])[:3]:
                    paths.append({"via": n["target"], "r1_type": n["type"],
                                  "target": e2["target"], "r2_type": e2["type"],
                                  "description": e2.get("desc", "")})
        return {"entity": entity, "neighbors": neighbors, "paths": paths}

    def get_global_context(self, top_k=10):
        important = sorted(self.entity_meta.items(), key=lambda x: x[1]["count"], reverse=True)[:top_k]
        return {
            "important_entities": [
                {"name": n, "type": m["type"], "mentions": m["count"], "description": m["desc"]}
                for n, m in important
            ],
            "recent_relationships": [],
        }

    def upsert_vector(self, doc_id, embedding, text, channel="", timestamp="", entities=None):
        self.documents[doc_id] = {
            "text": text[:1000], "embedding": embedding, "channel": channel,
            "timestamp": timestamp, "entities": entities or [],
        }

    def search_vectors(self, query_embedding, top_k=10):
        if query_embedding is None:
            return []
        scores = []
        for doc_id, doc in self.documents.items():
            emb = doc.get("embedding")
            if emb is not None:
                dot = np.dot(query_embedding, emb)
                norm = np.linalg.norm(query_embedding) * np.linalg.norm(emb)
                sim = float(dot / norm) if norm > 0 else 0.0
                scores.append((doc_id, sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        results = []
        for doc_id, score in scores[:top_k]:
            doc = self.documents[doc_id]
            results.append({
                "id": doc_id, "score": score, "text": doc["text"],
                "channel": doc["channel"], "timestamp": doc["timestamp"],
                "entities": doc["entities"],
            })
        return results

    def get_stats(self):
        return {
            "nodes": len(self.entity_meta),
            "relationships": sum(len(v) for v in self.edges.values()),
            "vectors": len(self.documents),
            "connected": False,
        }

    def cleanup_old(self, days=90):
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        for entity in list(self.edges.keys()):
            self.edges[entity] = [e for e in self.edges[entity]
                                  if not e.get("timestamp") or e["timestamp"] >= cutoff]
            if not self.edges[entity]:
                del self.edges[entity]
        old_docs = [k for k, v in self.documents.items()
                    if v.get("timestamp") and v["timestamp"] < cutoff]
        for k in old_docs:
            del self.documents[k]


class LightRAGEngine:
    """LightRAG 정석 구현.

    Pipeline:
    1. Ingest: text -> LLM entity/relationship extraction -> Neo4j + Milvus
    2. Query:
       - Local mode: entity neighbors (specific facts)
       - Global mode: community/topic summaries (abstract questions)
       - Hybrid mode: both combined (recommended)

    LLM-based extraction uses Gemini Flash (cheap, fast).
    """

    # Entity extraction prompt (LLM 기반)
    EXTRACTION_PROMPT = """Extract entities and relationships from this crypto news text.

TEXT: {text}

Return JSON only (no markdown):
{{
  "entities": [
    {{"name": "entity_name", "type": "person|org|coin|event|indicator|exchange", "description": "brief description"}}
  ],
  "relationships": [
    {{"source": "entity1", "target": "entity2", "type": "relationship_type", "description": "brief explanation"}}
  ]
}}

Rules:
- Entity names should be lowercase, canonical (e.g. "bitcoin" not "BTC/USDT")
- Common entity types: coin, exchange, person, org, event, indicator, regulation
- Common relationship types: affects, trades_on, regulates, invested_in, partnered_with, listed_on, caused, leads_to, correlated_with
- Only extract clearly stated facts, not speculation
- Maximum 8 entities and 6 relationships per text
- If no clear entities/relationships, return empty arrays"""

    def __init__(self):
        # Embedding model (lazy init)
        self._embed_client = None

        # Storage backends (configured from settings)
        self._graph = None
        self._vector_store = None
        self._fallback = None

        # LLM client for extraction (lazy init)
        self._ai_client = None

        # Dedup cache
        self._ingested_ids: Set[str] = set()

    def _get_ai_client(self):
        """Lazy-load AI client for LLM-based extraction."""
        if self._ai_client is None:
            try:
                from agents.claude_client import claude_client
                self._ai_client = claude_client
            except Exception as e:
                logger.error(f"AI client init for RAG failed: {e}")
        return self._ai_client

    @property
    def graph(self):
        """Get graph backend (Neo4j or in-memory fallback)."""
        if self._graph is not None:
            return self._graph
        try:
            from config.settings import settings
            neo4j_uri = getattr(settings, 'NEO4J_URI', '')
            neo4j_password = getattr(settings, 'NEO4J_PASSWORD', '')
            if neo4j_uri and NEO4J_AVAILABLE:
                self._graph = Neo4jGraph(
                    uri=neo4j_uri, user="neo4j", password=neo4j_password
                )
                if self._graph.driver:
                    return self._graph
        except Exception as e:
            logger.warning(f"Neo4j init failed: {e}")

        # Fallback to in-memory
        if self._fallback is None:
            self._fallback = InMemoryFallback()
        self._graph = self._fallback
        return self._graph

    @property
    def vector_store(self):
        """Get vector backend (Milvus/Zilliz or in-memory fallback)."""
        if self._vector_store is not None:
            return self._vector_store
        try:
            from config.settings import settings
            milvus_uri = getattr(settings, 'MILVUS_URI', '')
            milvus_token = getattr(settings, 'MILVUS_TOKEN', '')
            if milvus_uri and MILVUS_AVAILABLE:
                store = MilvusVectorStore(uri=milvus_uri, token=milvus_token)
                if store.client:
                    self._vector_store = store
                    return self._vector_store
        except Exception as e:
            logger.warning(f"Milvus init failed: {e}")

        # Fallback uses same InMemoryFallback
        if self._fallback is None:
            self._fallback = InMemoryFallback()
        self._vector_store = self._fallback
        return self._vector_store

    @property
    def embed_client(self):
        if self._embed_client is None and EMBEDDING_AVAILABLE:
            try:
                from config.settings import settings
                self._embed_client = genai.Client(
                    vertexai=True,
                    project=settings.PROJECT_ID,
                    location=settings.vertex_region,
                )
            except Exception as e:
                logger.warning(f"Embedding client init failed: {e}")
        return self._embed_client

    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding from Gemini text-embedding-005."""
        if self.embed_client is None:
            return None
        try:
            response = self.embed_client.models.embed_content(
                model="text-embedding-005",
                contents=[text[:512]],
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
            )
            embeddings = getattr(response, "embeddings", None) or []
            if embeddings and getattr(embeddings[0], "values", None):
                return np.array(embeddings[0].values, dtype=np.float32)
        except Exception as e:
            logger.warning(f"Embedding error: {e}")
        return None

    def _extract_with_llm(self, text: str) -> Dict:
        """LLM-based entity and relationship extraction (Gemini Flash).

        This is the core of LightRAG 정석 - LLM understands context,
        unlike regex which only matches patterns.
        """
        ai = self._get_ai_client()
        if not ai:
            return {"entities": [], "relationships": []}

        prompt = self.EXTRACTION_PROMPT.format(text=text[:800])
        try:
            response = ai.generate_response(
                system_prompt="You are a knowledge graph extraction engine. Return only valid JSON.",
                user_message=prompt,
                temperature=0.1,
                max_tokens=800,
                use_premium=False,  # cost-efficient model for extraction
                role="rag_extraction",
            )

            # Parse JSON
            text_clean = response.strip()
            if text_clean.startswith("```"):
                lines = text_clean.split('\n')
                lines = [l for l in lines if not l.strip().startswith("```")]
                text_clean = '\n'.join(lines)

            start = text_clean.find('{')
            end = text_clean.rfind('}') + 1
            if start >= 0 and end > start:
                return json.loads(text_clean[start:end])

            return {"entities": [], "relationships": []}
        except Exception as e:
            logger.warning(f"LLM extraction error: {e}")
            return {"entities": [], "relationships": []}

    def _extract_with_regex_fallback(self, text: str) -> Dict:
        """Regex fallback when LLM is unavailable."""
        entities = []
        known = {
            'btc', 'bitcoin', 'eth', 'ethereum', 'sol', 'solana', 'xrp',
            'bnb', 'doge', 'ada', 'avax', 'link', 'dot', 'matic', 'uni',
            'binance', 'coinbase', 'bybit', 'okx', 'upbit', 'kraken',
            'sec', 'cftc', 'fed', 'blackrock', 'grayscale', 'microstrategy',
            'tether', 'usdc', 'usdt', 'etf', 'halving', 'defi',
            'whale', 'liquidation', 'hack', 'exploit',
            'cpi', 'fomc', 'inflation', 'rate',
        }
        text_lower = text.lower()
        found = set()
        for entity in known:
            if entity in text_lower:
                found.add(entity)
                etype = "coin" if entity in {'btc','bitcoin','eth','ethereum','sol','xrp','bnb','doge','ada','avax','link','dot','matic','uni'} else \
                         "exchange" if entity in {'binance','coinbase','bybit','okx','upbit','kraken'} else \
                         "org" if entity in {'sec','cftc','fed','blackrock','grayscale','microstrategy','tether'} else "event"
                entities.append({"name": entity, "type": etype, "description": ""})

        # Co-occurrence relationships
        relationships = []
        found_list = list(found)
        for i in range(len(found_list)):
            for j in range(i + 1, min(len(found_list), i + 4)):
                relationships.append({
                    "source": found_list[i], "target": found_list[j],
                    "type": "co_mentioned", "description": "mentioned together"
                })

        return {"entities": entities, "relationships": relationships}

    def ingest_message(self, text: str, channel: str = "", timestamp: str = "",
                       message_id: str = "") -> Dict:
        """Ingest a single message through the full LightRAG pipeline.

        1. Dedup check
        2. LLM extraction (entities + relationships)
        3. Store entities/relationships in Neo4j
        4. Embed text, store in Milvus
        """
        if not text or len(text.strip()) < 15:
            return {}

        doc_id = message_id or hashlib.md5(
            f"{channel}:{text[:100]}:{timestamp}".encode()
        ).hexdigest()

        if doc_id in self._ingested_ids:
            return {"status": "duplicate_id", "doc_id": doc_id}

        # Step 0: Semantic Deduplication using Vector Store
        embedding = self._get_embedding(text[:512])
        if embedding is not None:
            # Search for highly similar past messages
            similar_docs = []
            if isinstance(self.vector_store, InMemoryFallback):
                similar_docs = self.vector_store.search_vectors(embedding, top_k=1)
            else:
                similar_docs = self.vector_store.search(embedding, top_k=1)
                
            if similar_docs and similar_docs[0].get("score", 0) > 0.92:
                logger.info(f"RAG Dedup: Skipping '{text[:50]}...' (Score: {similar_docs[0].get('score', 0):.2f})")
                self._ingested_ids.add(doc_id)
                return {"status": "duplicate_semantic", "doc_id": doc_id}

        # Step 1: LLM-based extraction (or regex fallback)
        ai = self._get_ai_client()
        if ai:
            extracted = self._extract_with_llm(text)
        else:
            extracted = self._extract_with_regex_fallback(text)

        entities = extracted.get("entities", [])
        relationships = extracted.get("relationships", [])

        # Step 2: Store entities in graph
        entity_names = []
        for ent in entities:
            name = ent.get("name", "").lower().strip()
            if not name:
                continue
            self.graph.upsert_entity(
                name=name,
                entity_type=ent.get("type", "unknown"),
                description=ent.get("description", ""),
                source_id=doc_id,
                timestamp=timestamp,
            )
            entity_names.append(name)

        # Step 3: Store relationships in graph
        for rel in relationships:
            source = rel.get("source", "").lower().strip()
            target = rel.get("target", "").lower().strip()
            if not source or not target:
                continue
            self.graph.upsert_relationship(
                source=source, target=target,
                rel_type=rel.get("type", "related_to"),
                description=rel.get("description", ""),
                weight=1.0, timestamp=timestamp, source_id=doc_id,
            )

        # Step 4: Embed and store in vector store
        if embedding is not None:
            if isinstance(self.vector_store, InMemoryFallback):
                self.vector_store.upsert_vector(
                    doc_id, embedding, text[:1000], channel, timestamp, entity_names
                )
            else:
                self.vector_store.upsert(
                    doc_id, embedding, text[:1000], channel, timestamp, entity_names
                )

        self._ingested_ids.add(doc_id)

        return {
            "status": "ingested",
            "doc_id": doc_id,
            "entities": entity_names,
            "relationships": len(relationships),
        }

    def ingest_batch(self, messages: List[Dict]) -> int:
        """Ingest a batch of telegram messages."""
        count = 0
        for msg in messages:
            result = self.ingest_message(
                text=msg.get("text", ""),
                channel=msg.get("channel", ""),
                timestamp=msg.get("timestamp", msg.get("created_at", "")),
                message_id=str(msg.get("message_id", msg.get("id", ""))),
            )
            if result.get("status") == "ingested":
                count += 1
        stats = self.get_stats()
        logger.info(f"LightRAG ingested {count}/{len(messages)} messages (stats: {stats})")
        return count

    def query(self, query_text: str, mode: str = "hybrid", top_k: int = 10) -> Dict:
        """Query the Graph RAG.

        Modes (LightRAG 정석 Dual-level Retrieval):
        - "local": entity neighbors, specific facts (Low-level)
        - "global": community summaries, abstract topics (High-level)
        - "hybrid": both combined (recommended)
        """
        # Extract entities from query
        query_entities = []
        known_coins = {'btc', 'bitcoin', 'eth', 'ethereum', 'sol', 'xrp', 'bnb'}
        known_orgs = {'sec', 'fed', 'binance', 'blackrock', 'grayscale'}
        query_lower = query_text.lower()
        for entity in known_coins | known_orgs:
            if entity in query_lower:
                query_entities.append(entity)

        result = {
            "local_context": {},
            "global_context": {},
            "semantic_results": [],
            "entities_found": query_entities,
            "mode": mode,
        }

        # ── Local retrieval (Low-level: entity neighbors) ──
        if mode in ("local", "hybrid"):
            for entity in query_entities:
                ctx = self.graph.get_local_context(entity, max_depth=2)
                if ctx.get("neighbors") or ctx.get("paths"):
                    result["local_context"][entity] = ctx

        # ── Global retrieval (High-level: important entities + topics) ──
        if mode in ("global", "hybrid"):
            result["global_context"] = self.graph.get_global_context(top_k=10)

        # ── Semantic search (vector similarity) ──
        query_embedding = self._get_embedding(query_text)
        if query_embedding is not None:
            if isinstance(self.vector_store, InMemoryFallback):
                result["semantic_results"] = self.vector_store.search_vectors(
                    query_embedding, top_k=top_k
                )
            else:
                result["semantic_results"] = self.vector_store.search(
                    query_embedding, top_k=top_k
                )

        return result

    def format_context_for_agents(self, query_result: Dict, max_length: int = 2000) -> str:
        """Format Graph RAG query result as compact text for agent consumption."""
        lines = []

        # Local context (specific facts about queried entities)
        local_ctx = query_result.get("local_context", {})
        if local_ctx:
            lines.append("[KNOWLEDGE GRAPH - Local]")
            for entity, ctx in local_ctx.items():
                for n in ctx.get("neighbors", [])[:4]:
                    rel_type = n.get("rel_type", n.get("type", "?"))
                    target = n.get("name", n.get("target", "?"))
                    lines.append(f"  ({entity}) --[{rel_type}]--> ({target})")
                for p in ctx.get("paths", [])[:2]:
                    lines.append(
                        f"  ({entity}) -> ({p.get('via','?')}) --[{p.get('r2_type','?')}]--> "
                        f"({p.get('target','?')}) [2-hop]"
                    )

        # Global context (market-wide important entities)
        global_ctx = query_result.get("global_context", {})
        important = global_ctx.get("important_entities", [])
        if important:
            ent_str = ", ".join([f"{e['name']}({e.get('mentions',0)})" for e in important[:5]])
            lines.append(f"[KEY ENTITIES] {ent_str}")

        recent_rels = global_ctx.get("recent_relationships", [])
        if recent_rels:
            lines.append("[RECENT EVENTS]")
            for r in recent_rels[:5]:
                lines.append(
                    f"  ({r['source']}) --[{r['rel_type']}]--> ({r['target']}): "
                    f"{r.get('description', '')[:80]}"
                )

        # Semantic search results (relevant news)
        semantic = query_result.get("semantic_results", [])
        if semantic:
            lines.append("[RELEVANT NEWS]")
            for doc in semantic[:5]:
                score_str = f"({doc.get('score', 0):.2f})" if doc.get("score") else ""
                lines.append(
                    f"  [{doc.get('channel', '?')}] {doc.get('text', '')[:120]} {score_str}"
                )

        result = '\n'.join(lines)
        return result[:max_length] if result else "RAG Context: No data available"

    def get_stats(self) -> Dict:
        """Return current RAG stats."""
        graph_stats = self.graph.get_stats() if hasattr(self.graph, 'get_stats') else {}
        vector_stats = self.vector_store.get_stats() if hasattr(self.vector_store, 'get_stats') else {}
        return {
            "graph": graph_stats,
            "vectors": vector_stats,
            "ingested_docs": len(self._ingested_ids),
        }

    def cleanup_old(self, days: int = 90):
        """Remove old data from all backends."""
        try:
            if hasattr(self.graph, 'cleanup_old'):
                self.graph.cleanup_old(days=days)
            logger.info("LightRAG cleanup completed")
        except Exception as e:
            logger.error(f"LightRAG cleanup error: {e}")


# Singleton instance
light_rag = LightRAGEngine()
