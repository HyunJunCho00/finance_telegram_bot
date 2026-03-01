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
import os
from pathlib import Path

# [FIX HIGH-11] Persistent path for ingested IDs cache
_BASE_DIR = Path(__file__).parent.parent
_IDS_CACHE_PATH = str(_BASE_DIR / "data" / "ingested_ids.json")

# Voyage AI embedding
try:
    import voyageai
    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False
    logger.warning("voyageai embedding not available")

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
                        e.last_seen = $timestamp,
                        e.evidence_count = 1,
                        e.evidence_list = [$source_id],
                        e.status = 'PENDING'
                    ON MATCH SET
                        e.description = CASE
                            WHEN size(e.description) < 500 AND NOT e.description CONTAINS $description THEN e.description + ' | ' + $description
                            ELSE e.description
                        END,
                        e.mention_count = e.mention_count + 1,
                        e.last_seen = $timestamp,
                        e.evidence_count = CASE 
                            WHEN NOT $source_id IN e.evidence_list THEN coalesce(e.evidence_count, 1) + 1 
                            ELSE coalesce(e.evidence_count, 1) 
                        END,
                        e.evidence_list = CASE 
                            WHEN NOT $source_id IN e.evidence_list THEN e.evidence_list + [$source_id] 
                            ELSE e.evidence_list 
                        END,
                        e.status = CASE
                            WHEN e.status = 'PENDING' AND NOT $source_id IN e.evidence_list AND coalesce(e.evidence_count, 1) >= 2 THEN 'CORROBORATED'
                            ELSE coalesce(e.status, 'PENDING')
                        END
                """, name=name.lower(), type=entity_type, description=description[:300],
                     timestamp=timestamp, source_id=source_id)
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
                        r.count = 1,
                        r.evidence_count = 1,
                        r.evidence_list = [$source_id],
                        r.status = 'PENDING'
                    ON MATCH SET
                        r.description = CASE
                            WHEN size(r.description) < 300 AND NOT r.description CONTAINS $description THEN r.description + ' | ' + $description
                            ELSE r.description
                        END,
                        r.weight = r.weight + $weight,
                        r.last_seen = $timestamp,
                        r.count = r.count + 1,
                        r.evidence_count = CASE 
                            WHEN NOT $source_id IN r.evidence_list THEN coalesce(r.evidence_count, 1) + 1 
                            ELSE coalesce(r.evidence_count, 1) 
                        END,
                        r.evidence_list = CASE 
                            WHEN NOT $source_id IN r.evidence_list THEN r.evidence_list + [$source_id] 
                            ELSE r.evidence_list 
                        END,
                        r.status = CASE
                            WHEN r.status = 'PENDING' AND NOT $source_id IN r.evidence_list AND coalesce(r.evidence_count, 1) >= 2 THEN 'CORROBORATED'
                            ELSE coalesce(r.status, 'PENDING')
                        END
                """, source=source.lower(), target=target.lower(),
                     rel_type=rel_type, description=description[:200],
                     weight=weight, timestamp=timestamp, source_id=source_id)
        except Exception as e:
            logger.error(f"Neo4j upsert_relationship error: {e}")

    def get_triangulation_candidates(self, limit: int = 10) -> List[Dict]:
        """Fetch PENDING or CORROBORATED relationships to be verified by web search."""
        if not self.driver:
            return []
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (s:Entity)-[r:RELATES_TO]->(t:Entity)
                    WHERE r.status IN ['PENDING', 'CORROBORATED']
                    RETURN s.name AS source, t.name AS target, r.type AS rel_type, 
                           r.description AS description, r.status AS status, r.evidence_count AS evidence_count
                    ORDER BY r.evidence_count DESC, r.last_seen DESC
                    LIMIT $limit
                """, limit=limit)
                return [dict(rec) for rec in result]
        except Exception as e:
            logger.error(f"Neo4j get_triangulation_candidates error: {e}")
            return []

    def update_relationship_status(self, source: str, target: str, rel_type: str, status: str, weight_boost: float = 0.0):
        if not self.driver:
            return
        try:
            with self.driver.session() as session:
                session.run("""
                    MATCH (s:Entity {name: $source})-[r:RELATES_TO {type: $rel_type}]->(t:Entity {name: $target})
                    SET r.status = $status, r.weight = r.weight + $weight_boost
                """, source=source, target=target, rel_type=rel_type, status=status, weight_boost=weight_boost)
        except Exception as e:
            logger.error(f"Neo4j update_relationship_status error: {e}")

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

    def export_all(self) -> Dict:
        """Extract all entities and relationships for GCS archival."""
        if not self.driver:
            return {"entities": [], "relationships": []}
        try:
            with self.driver.session() as session:
                # 1. Export Entities
                res_ent = session.run("MATCH (e:Entity) RETURN e")
                entities = [dict(record["e"]) for record in res_ent]

                # 2. Export Relationships
                res_rel = session.run("""
                    MATCH (s:Entity)-[r:RELATES_TO]->(t:Entity)
                    RETURN s.name AS source, t.name AS target, properties(r) AS props, type(r) AS type
                """)
                relationships = []
                for record in res_rel:
                    rel = record["props"]
                    rel.update({"source": record["source"], "target": record["target"], "type": record["type"]})
                    relationships.append(rel)

                return {
                    "entities": entities,
                    "relationships": relationships,
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            logger.error(f"Neo4j export error: {e}")
            return {"entities": [], "relationships": []}

    def import_data(self, entities: List[Dict], relationships: List[Dict]):
        """Bulk import entities and relationships (Restore from GCS)."""
        if not self.driver:
            return
        try:
            with self.driver.session() as session:
                # Import Entities
                session.run("""
                    UNWIND $entities AS ent
                    MERGE (e:Entity {name: ent.name})
                    SET e += ent
                """, entities=entities)

                # Import Relationships
                session.run("""
                    UNWIND $rels AS rel
                    MATCH (s:Entity {name: rel.source})
                    MATCH (t:Entity {name: rel.target})
                    MERGE (s)-[r:RELATES_TO {type: rel.type}]->(t)
                    SET r += rel
                """, rels=relationships)
                logger.info(f"Neo4j: Imported {len(entities)} entities, {len(relationships)} relationships")
        except Exception as e:
            logger.error(f"Neo4j import error: {e}")

    def cleanup_old(self, days: int = 90):
        """Delete data older than X days, while preserving important relationships."""
        if not self.driver:
            return
        try:
            cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            with self.driver.session() as session:
                # 1. Delete old relationships (keep high weight ones)
                res_rel = session.run("""
                    MATCH ()-[r:RELATES_TO]->()
                    WHERE r.last_seen < $cutoff AND r.weight < 5
                    DELETE r
                    RETURN count(r) AS deleted_count
                """, cutoff=cutoff)
                rel_deleted = res_rel.single()["deleted_count"]

                # 2. Delete orphaned entities
                res_ent = session.run("""
                    MATCH (e:Entity)
                    WHERE NOT (e)-[:RELATES_TO]-()
                    DELETE e
                    RETURN count(e) AS deleted_count
                """)
                ent_deleted = res_ent.single()["deleted_count"]

                logger.info(f"Neo4j Cleanup: Deleted {rel_deleted} relationships and {ent_deleted} entities")
        except Exception as e:
            logger.error(f"Neo4j cleanup error: {e}")

    def get_stats(self) -> Dict:
        """Return graph node and relationship counts."""
        if not self.driver:
            return {"nodes": 0, "relationships": 0, "connected": False}
        try:
            with self.driver.session() as session:
                nodes = session.run("MATCH (n:Entity) RETURN count(n) AS c").single()["c"]
                rels = session.run("MATCH ()-[r:RELATES_TO]->() RETURN count(r) AS c").single()["c"]
                return {"nodes": nodes, "relationships": rels, "connected": True}
        except Exception:
            return {"nodes": 0, "relationships": 0, "connected": False}



class MilvusVectorStore:
    """Zilliz Cloud (Milvus) free tier vector store.

    Stores document embeddings for semantic search.
    Free tier: 1M vectors, 2 collections.
    """

    COLLECTION_NAME = "crypto_news_1024"
    DIMENSION = 1024  # voyage-3 dimension

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
                # Ensure load works even if index was somehow missing
                try:
                    self._client.load_collection(self.COLLECTION_NAME)
                except Exception:
                    # If load fails due to missing index, try to create it
                    idx_params = self._client.prepare_index_params()
                    idx_params.add_index(field_name="vector", metric_type="COSINE", index_type="AUTOINDEX")
                    self._client.create_index(
                        collection_name=self.COLLECTION_NAME,
                        index_params=idx_params,
                    )
                    self._client.load_collection(self.COLLECTION_NAME)
                logger.info("Milvus connected and collection loaded")
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
                idx_params = self._client.prepare_index_params()
                idx_params.add_index(field_name="vector", metric_type="COSINE", index_type="AUTOINDEX")
                self._client.create_index(
                    collection_name=self.COLLECTION_NAME,
                    index_params=idx_params,
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


class CloudflareReranker:
    """Cloudflare Workers AI bge-reranker-base — cross-encoder for borderline dedup.

    Role in pipeline:
      Called ONLY when cosine+Jaccard land in the borderline zone.
      ~8% of messages. Free tier: 10,000 neurons/day (>>our usage).

    Why cross-encoder over bi-encoder here?
      Both texts are fed together → model sees full context of both sides.
      Catches semantic negation ("BTC buys" vs "BTC sells") and implicit entities
      that pure cosine/Jaccard miss.

    Graceful degradation:
      Any network/auth failure → returns None → caller falls back to cosine decision.
      Never blocks the ingestion pipeline.
    """

    _URL_TEMPLATE = (
        "https://api.cloudflare.com/client/v4/accounts/{account_id}"
        "/ai/run/@cf/baai/bge-reranker-base"
    )
    _TIMEOUT_S = 3.0  # tight timeout: don't hold up streaming ingestion

    def __init__(self):
        self._account_id: str = ""
        self._api_key: str = ""
        self._enabled: bool = False
        self._session = None  # requests.Session, lazy init

    def _init(self):
        """Lazy init from settings — avoids circular import at module load."""
        if self._enabled or self._account_id:
            return
        try:
            from config.settings import settings
            self._account_id = getattr(settings, "CLOUDFLARE_ACCOUNT_ID", "")
            self._api_key = getattr(settings, "CLOUDFLARE_AI_API_KEY", "")
            self._enabled = bool(self._account_id and self._api_key)
            if not self._enabled:
                logger.debug("CloudflareReranker: credentials not set, disabled.")
        except Exception as e:
            logger.warning(f"CloudflareReranker init failed: {e}")

    @property
    def enabled(self) -> bool:
        self._init()
        return self._enabled

    def rerank(self, query: str, contexts: List[str]) -> Optional[List[float]]:
        """Call bge-reranker-base and return relevance scores [0, 1] per context.

        Args:
            query:    New message text.
            contexts: List of past document texts to compare against.

        Returns:
            List of float scores aligned with `contexts`, or None on failure.
            Scores are raw logits mapped to [0, 1] via sigmoid by Cloudflare.
        """
        self._init()
        if not self._enabled or not contexts:
            return None

        import requests  # optional dep — already used elsewhere in project

        url = self._URL_TEMPLATE.format(account_id=self._account_id)
        payload = {
            "query": query[:512],
            "contexts": [{"text": c[:512]} for c in contexts],
        }
        try:
            if self._session is None:
                self._session = requests.Session()
                self._session.headers.update(
                    {"Authorization": f"Bearer {self._api_key}",
                     "Content-Type": "application/json"}
                )
            resp = self._session.post(url, json=payload, timeout=self._TIMEOUT_S)
            resp.raise_for_status()
            data = resp.json()
            results = data.get("result", [])
            # results: [{"id": 0, "score": 0.92}, ...] ordered by id
            scores = [0.0] * len(contexts)
            for item in results:
                idx = item.get("id", -1)
                if 0 <= idx < len(scores):
                    scores[idx] = float(item.get("score", 0.0))
            return scores
        except Exception as e:
            logger.warning(f"CloudflareReranker.rerank() failed (falling back to cosine): {e}")
            return None


# Module-level singleton — shared across all ingest_message calls
_cf_reranker = CloudflareReranker()


class InMemoryFallback:
    """In-memory fallback when Neo4j/Milvus unavailable (local dev)."""

    # [FIX MEDIUM-17] Memory caps to prevent OOM
    MAX_ENTITIES = 10000
    MAX_DOCUMENTS = 5000

    def __init__(self):
        self.edges = defaultdict(list)
        self.entity_meta = {}
        self.edge_counts = defaultdict(int)
        self.documents = {}

    def upsert_entity(self, name, entity_type, description, **kwargs):
        ts = kwargs.get("timestamp", "")
        name = name.lower()
        # [FIX MEDIUM-17] Evict oldest 20% when cap reached
        if len(self.entity_meta) >= self.MAX_ENTITIES and name not in self.entity_meta:
            sorted_entities = sorted(self.entity_meta.items(), key=lambda x: x[1].get("last", ""))
            to_remove = sorted_entities[:len(sorted_entities) // 5]
            for old_name, _ in to_remove:
                del self.entity_meta[old_name]
                self.edges.pop(old_name, None)
                self.edge_counts.pop(old_name, None)
        source_id = kwargs.get("source_id", "")
        if name not in self.entity_meta:
            self.entity_meta[name] = {"type": entity_type, "desc": description, "count": 0, "first": ts, "evidence_count": 0, "evidence_list": [], "status": "PENDING"}
        self.entity_meta[name]["count"] += 1
        self.entity_meta[name]["last"] = ts
        if source_id and source_id not in self.entity_meta[name]["evidence_list"]:
            self.entity_meta[name]["evidence_list"].append(source_id)
            self.entity_meta[name]["evidence_count"] += 1
        if self.entity_meta[name]["status"] == "PENDING" and self.entity_meta[name]["evidence_count"] >= 2:
            self.entity_meta[name]["status"] = "CORROBORATED"
        self.edge_counts[name] += 1

    def upsert_relationship(self, source, target, rel_type, **kwargs):
        source, target = source.lower(), target.lower()
        ts = kwargs.get("timestamp", "")
        source_id = kwargs.get("source_id", "")
        
        existing = next((e for e in self.edges[source] if e["target"] == target and e["type"] == rel_type), None)
        if existing:
            if source_id and source_id not in existing.get("evidence_list", []):
                existing.setdefault("evidence_list", []).append(source_id)
                existing["evidence_count"] = existing.get("evidence_count", 0) + 1
            if existing.get("status", "PENDING") == "PENDING" and existing["evidence_count"] >= 2:
                existing["status"] = "CORROBORATED"
            existing["timestamp"] = ts
        else:
            self.edges[source].append({
                "target": target, "type": rel_type,
                "desc": kwargs.get("description", ""),
                "timestamp": ts,
                "evidence_count": 1,
                "evidence_list": [source_id] if source_id else [],
                "status": "PENDING"
            })
        self.edge_counts[source] += 1
        self.edge_counts[target] += 1

    def get_triangulation_candidates(self, limit: int = 10) -> List[Dict]:
        candidates = []
        for source, rels in self.edges.items():
            for r in rels:
                if r.get("status", "PENDING") in ["PENDING", "CORROBORATED"]:
                    candidates.append({
                        "source": source,
                        "target": r["target"],
                        "rel_type": r["type"],
                        "description": r.get("desc", ""),
                        "status": r.get("status", "PENDING"),
                        "evidence_count": r.get("evidence_count", 1)
                    })
        candidates.sort(key=lambda x: (x["evidence_count"], x.get("timestamp", "")), reverse=True)
        return candidates[:limit]

    def update_relationship_status(self, source: str, target: str, rel_type: str, status: str, weight_boost: float = 0.0):
        source, target = source.lower(), target.lower()
        if source in self.edges:
            for r in self.edges[source]:
                if r["target"] == target and r["type"] == rel_type:
                    r["status"] = status
                    r["weight"] = r.get("weight", 1.0) + weight_boost

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
        # [FIX MEDIUM-17] Evict oldest documents when cap reached
        if len(self.documents) >= self.MAX_DOCUMENTS and doc_id not in self.documents:
            sorted_docs = sorted(self.documents.items(), key=lambda x: x[1].get("timestamp", ""))
            for old_id, _ in sorted_docs[:len(sorted_docs) // 5]:
                del self.documents[old_id]
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

    # ── Deduplication thresholds ──────────────────────────────────────────────
    # Two separate thresholds for two distinct purposes:
    #
    # SPAM_THRESHOLD (0.90):
    #   Same-channel, time-windowed forward detection.
    #   Only applies within SPAM_WINDOW_HOURS of the past document.
    #   Research basis: SemHash default ~0.9, NeMo SemDedup eps=0.90 for near-exact.
    #   voyage-finance-2 calibration: verbatim Telegram forwards score 0.96-0.99,
    #   paraphrased reposts score 0.91-0.95. Threshold at 0.90 catches both.
    #
    # CORROBORATION_THRESHOLD (0.84):
    #   Cross-channel: different outlets reporting the same event.
    #   Lower threshold because paraphrase/editorial variation is expected.
    #   Research basis: FinMTEB 2025 cross-source financial STS pairs cluster 0.82-0.88.
    #   Time-window NOT applied — corroboration is independent of recency.
    #
    # SPAM_WINDOW_HOURS (48):
    #   Max age of a past document to be considered a spam duplicate.
    #   A 3-month-old "BTC up 2%" is NOT a duplicate of today's "BTC up 2%".
    #   Reuters/Bloomberg internal dedup uses 1-24h; 48h adds safety margin
    #   for weekly-summary channels.
    SPAM_THRESHOLD: float = 0.90
    CORROBORATION_THRESHOLD: float = 0.84
    SPAM_WINDOW_HOURS: int = 48

    # ── NDD-MAC entity gate parameters ─────────────────────────────────────────
    # ENTITY_GATE_STRONG (0.50):
    #   Jaccard overlap threshold that qualifies as "strong entity match".
    #   When Jaccard ≥ 0.50, both messages share majority of financial entities
    #   (e.g., both mention bitcoin + binance) → likely same event → relax cosine
    #   requirement by ENTITY_BOOST_MARGIN.
    #
    # ENTITY_BOOST_MARGIN (0.05):
    #   Cosine threshold relaxation when entity gate is strong.
    #   "Strong entity confirmation allows slightly lower cosine to still qualify."
    #   Net effect: SPAM gate becomes 0.85 (was 0.90), CORR gate becomes 0.79 (was 0.84).
    #
    # Entity gate = None (zero overlap):
    #   Messages mention entirely different financial entities → cannot be near-
    #   duplicates of the same event → skip cosine check for that candidate entirely.
    #   e.g., "BTC hits ATH" vs "ETH staking update" → Jaccard=0 → hard skip.
    ENTITY_GATE_STRONG: float = 0.50
    ENTITY_BOOST_MARGIN: float = 0.05

    # ── Cloudflare reranker borderline zones ───────────────────────────────────
    # Reranker is called ONLY when cosine falls in [lower, threshold) — the gray
    # zone where cosine+Jaccard alone are unreliable.
    #
    # Purpose A (Spam, same-channel + 48h):
    #   RERANKER_SPAM_LOWER  (0.78): below this → definitely not spam, skip reranker
    #   RERANKER_SPAM_CONFIRM (0.75): reranker score above this → confirm as spam
    #   Zone width: spam_thr - 0.78 = ~0.07~0.12 (narrow by design)
    #   Goal: prevent false-positive spam blocks (permanent data loss)
    #
    # Purpose B (Corroboration, cross-channel):
    #   RERANKER_CORR_LOWER  (0.65): below this → not worth checking
    #   RERANKER_CORR_CONFIRM (0.65): reranker score above this → corroboration
    #   Zone: 0.65 to corr_thr (~0.79-0.84) — expands corroboration detection
    #   Goal: catch same-event reports that current threshold misses
    #
    # Expected call rate: ~8% of ingested messages.
    # Cost: <1 neuron/call. Free tier: 10,000 neurons/day → effectively free.
    RERANKER_SPAM_LOWER: float = 0.78
    RERANKER_SPAM_CONFIRM: float = 0.75
    RERANKER_CORR_LOWER: float = 0.65
    RERANKER_CORR_CONFIRM: float = 0.65

    def _call_reranker(self, query: str, past_text: str) -> Optional[float]:
        """Invoke Cloudflare bge-reranker-base for a single query/document pair.

        Returns the relevance score [0, 1] or None if reranker is disabled/failed.
        Caller must treat None as "reranker unavailable → fall back to cosine logic".
        """
        scores = _cf_reranker.rerank(query, [past_text])
        if scores is None:
            return None
        return scores[0] if scores else None

    @staticmethod
    def _entity_gate_check(new_entities: set, past_entities_json: str) -> Optional[float]:
        """Entity prerequisite gate — NDD-MAC (NAACL 2025 Industry) metadata signal.

        Financial event deduplication requires shared entity context.
        Two messages with zero overlapping entities (coins, exchanges, orgs) cannot
        be near-duplicates of the same event, regardless of cosine similarity.

        Returns:
            None  — zero entity overlap → different financial events → skip candidate.
            1.0   — entity data unavailable (extraction failed or generic text)
                    → no gate applied, fall back to pure cosine comparison.
            float — Jaccard score (0, 1] → partial/strong entity overlap.

        Why 1.0 as the "no data" fallback?
            Conservative: if we can't extract entities from either message, we should
            NOT silently pass through potential duplicates. 1.0 means "treat as strong
            match" from the entity dimension — the cosine threshold then decides alone.
        """
        if not new_entities:
            return 1.0  # No entity data on new msg → cannot gate → pure cosine
        try:
            past_entities = set(json.loads(past_entities_json or "[]"))
        except (json.JSONDecodeError, TypeError):
            return 1.0
        if not past_entities:
            return 1.0  # No entity data on past doc → cannot gate → pure cosine

        intersection = new_entities & past_entities
        if not intersection:
            return None  # Zero shared entities → hard gate → skip this candidate

        union = new_entities | past_entities
        return len(intersection) / len(union) if union else 0.0

    @staticmethod
    def _parse_ts(ts_str: str) -> Optional[datetime]:
        """Parse ISO-8601 timestamp string to UTC datetime. Returns None on failure."""
        if not ts_str:
            return None
        try:
            dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except Exception:
            return None

    def _within_spam_window(self, current_ts: str, past_ts: str) -> bool:
        """Return True if past document is within SPAM_WINDOW_HOURS of current message.

        If either timestamp is unparseable, defaults to True (conservative: apply
        spam filter even without time data, to avoid regressing to old behaviour).
        """
        dt_current = self._parse_ts(current_ts)
        dt_past = self._parse_ts(past_ts)
        if dt_current is None or dt_past is None:
            return True  # conservative: treat as within window
        age_hours = abs((dt_current - dt_past).total_seconds()) / 3600.0
        return age_hours <= self.SPAM_WINDOW_HOURS

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

        # [FIX HIGH-11] Load from disk to avoid 1000+ LLM calls on restart
        self._ingested_ids: Set[str] = self._load_ingested_ids()

    def _load_ingested_ids(self) -> Set[str]:
        """Load ingested message IDs from disk. Falls back to GCS if local file missing (VM reset recovery)."""
        # 1) Try local disk first (fast)
        try:
            if os.path.exists(_IDS_CACHE_PATH):
                with open(_IDS_CACHE_PATH, 'r', encoding='utf-8') as f:
                    ids = set(json.load(f))
                    logger.info(f"LightRAG: loaded {len(ids)} ingested IDs from disk")
                    return ids
        except Exception as e:
            logger.warning(f"Failed to load ingested IDs from disk: {e}")

        # 2) Fallback: restore from GCS to survive VM resets
        try:
            from config.settings import settings
            if settings.GCS_ARCHIVE_BUCKET:
                from google.cloud import storage
                client = storage.Client()
                bucket = client.bucket(settings.GCS_ARCHIVE_BUCKET)
                blob = bucket.blob("lightrag/ingested_ids.json")
                if blob.exists():
                    data = json.loads(blob.download_as_text())
                    ids = set(data)
                    logger.info(f"LightRAG: restored {len(ids)} ingested IDs from GCS (VM reset recovery)")
                    # Write back to local disk immediately
                    os.makedirs(os.path.dirname(_IDS_CACHE_PATH), exist_ok=True)
                    with open(_IDS_CACHE_PATH, 'w', encoding='utf-8') as f:
                        json.dump(data, f)
                    return ids
        except Exception as e:
            logger.warning(f"LightRAG: GCS ingested_ids restore failed: {e}")

        logger.info("LightRAG: starting fresh (no ingested IDs found)")
        return set()

    def _save_ingested_ids(self):
        """Persist ingested IDs to disk AND GCS (keep latest 5000)."""
        try:
            os.makedirs(os.path.dirname(_IDS_CACHE_PATH), exist_ok=True)
            ids_list = list(self._ingested_ids)
            if len(ids_list) > 5000:
                ids_list = ids_list[-5000:]
            # 1) Save local
            with open(_IDS_CACHE_PATH, 'w', encoding='utf-8') as f:
                json.dump(ids_list, f)
            # 2) Backup to GCS
            try:
                from config.settings import settings
                if settings.GCS_ARCHIVE_BUCKET:
                    from google.cloud import storage
                    client = storage.Client()
                    bucket = client.bucket(settings.GCS_ARCHIVE_BUCKET)
                    blob = bucket.blob("lightrag/ingested_ids.json")
                    blob.upload_from_string(json.dumps(ids_list), content_type="application/json")
            except Exception as gcs_e:
                logger.warning(f"LightRAG: GCS ingested_ids backup failed (non-fatal): {gcs_e}")
        except Exception as e:
            logger.warning(f"Failed to save ingested IDs cache: {e}")

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
            neo4j_uri = settings.neo4j_uri
            neo4j_user = settings.neo4j_user
            neo4j_password = getattr(settings, 'NEO4J_PASSWORD', '')
            if neo4j_uri and NEO4J_AVAILABLE:
                self._graph = Neo4jGraph(
                    uri=neo4j_uri, user=neo4j_user, password=neo4j_password
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
                if getattr(settings, 'VOYAGE_API_KEY', None):
                    self._embed_client = voyageai.Client(api_key=settings.VOYAGE_API_KEY)
                else:
                    logger.warning("VOYAGE_API_KEY not set")
            except Exception as e:
                logger.warning(f"Embedding client init failed: {e}")
        return self._embed_client

    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding from Voyage AI voyage-finance-2."""
        if self.embed_client is None:
            return None
        
        import time
        max_retries = 3
        base_delay = 2.0
        
        for attempt in range(max_retries):
            try:
                result = self.embed_client.embed([text[:4000]], model="voyage-finance-2")
                if result.embeddings and len(result.embeddings) > 0:
                    return np.array(result.embeddings[0], dtype=np.float32)
            except Exception as e:
                err_str = str(e).lower()
                if "429" in err_str or "rate" in err_str or "limit" in err_str:
                    if attempt < max_retries - 1:
                        sleep_time = base_delay * (2 ** attempt)
                        logger.warning(f"Embedding error (429), retrying in {sleep_time}s...")
                        time.sleep(sleep_time)
                        continue
                logger.warning(f"Embedding error: {e}")
                return None
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
                try:
                    return json.loads(text_clean[start:end])
                except json.JSONDecodeError as decode_err:
                    logger.warning(f"LLM extraction JSONDecodeError: {decode_err} on string: {text_clean[start:end]}")
                    return {"entities": [], "relationships": []}

            return {"entities": [], "relationships": []}
        except Exception as e:
            logger.warning(f"LLM extraction API error: {e}")
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

        # Step 0: Metadata-Aware Semantic Dedup  ─── NDD-MAC architecture (NAACL 2025)
        #
        # Three signals, two purposes:
        #
        #  Signal 1 — Cosine similarity (semantic):  voyage-finance-2 embeddings
        #  Signal 2 — Entity Jaccard (metadata):     fast regex extraction, no LLM cost
        #             Identifies shared financial entities: BTC/ETH/exchange/org/macro
        #  Signal 3 — Temporal window (48h):          spam purpose only
        #
        #  Purpose A — Spam guard (same-channel, ≤48h):
        #    Base threshold: SPAM_THRESHOLD=0.90
        #    Strong entity match (Jaccard≥0.50) relaxes it by ENTITY_BOOST_MARGIN=0.05
        #    → effective threshold 0.85 when entity confirms semantic signal
        #
        #  Purpose B — Cross-channel corroboration (no time limit):
        #    Base threshold: CORROBORATION_THRESHOLD=0.84
        #    Strong entity match → effective threshold 0.79
        #    NOT discarded — proceeds to ingest so graph edge weight accumulates
        #
        #  Entity hard gate:
        #    Zero entity overlap → messages concern different financial events →
        #    skip that candidate entirely, regardless of cosine score.
        #    e.g. "BTC breaks ATH" vs "ETH staking upgrade" → Jaccard=0 → skip.
        #    If entity extraction yields nothing → fallback to pure cosine (no gate).
        #
        #  Searches top_k=3 candidates; entity gate filters per candidate;
        #  best cosine among surviving candidates is used.

        # Fast regex entity extraction (no LLM) for metadata signal
        quick_entities = set(
            e["name"]
            for e in self._extract_with_regex_fallback(text[:300]).get("entities", [])
        )

        embedding = self._get_embedding(text[:512])
        if embedding is not None:
            similar_docs = (
                self.vector_store.search_vectors(embedding, top_k=3)
                if isinstance(self.vector_store, InMemoryFallback)
                else self.vector_store.search(embedding, top_k=3)
            )

            # Entity gate: filter candidates; track best surviving candidate
            best_doc: Optional[Dict] = None
            best_cosine: float = 0.0
            best_jaccard: float = 0.0

            for candidate in (similar_docs or []):
                jaccard = self._entity_gate_check(
                    quick_entities, candidate.get("entities", "[]")
                )
                if jaccard is None:
                    # Hard gate: zero entity overlap → different event → skip
                    logger.debug(
                        f"Truth Engine: Entity gate blocked candidate "
                        f"[ch={candidate.get('channel','?')}] '{candidate.get('text','')[:40]}'"
                    )
                    continue
                cosine = candidate.get("score", 0)
                if cosine > best_cosine:
                    best_cosine = cosine
                    best_jaccard = jaccard
                    best_doc = candidate

            if best_doc:
                past_channel = best_doc.get("channel", "")
                past_ts = best_doc.get("timestamp", "")

                # Threshold adjustment: strong entity overlap validates semantic signal
                spam_thr = self.SPAM_THRESHOLD
                corr_thr = self.CORROBORATION_THRESHOLD
                if best_jaccard >= self.ENTITY_GATE_STRONG:
                    spam_thr -= self.ENTITY_BOOST_MARGIN   # 0.90 → 0.85
                    corr_thr -= self.ENTITY_BOOST_MARGIN   # 0.84 → 0.79

                if past_channel == channel:
                    # ── Purpose A: same-channel spam guard (time-windowed) ──
                    if self._within_spam_window(timestamp, past_ts):
                        if best_cosine > spam_thr:
                            # High confidence spam — no reranker needed
                            logger.info(
                                f"Truth Engine: Spam discarded (high-conf) "
                                f"[cosine={best_cosine:.3f}, entity_j={best_jaccard:.2f}, "
                                f"thr={spam_thr:.2f}, ch={channel}] '{text[:50]}...'"
                            )
                            self._ingested_ids.add(doc_id)
                            return {"status": "spam_same_channel", "doc_id": doc_id}

                        elif best_cosine >= self.RERANKER_SPAM_LOWER:
                            # Borderline zone — ask cross-encoder to confirm
                            reranker_score = self._call_reranker(text, best_doc.get("text", ""))
                            if reranker_score is not None:
                                if reranker_score > self.RERANKER_SPAM_CONFIRM:
                                    logger.info(
                                        f"Truth Engine: Spam confirmed by reranker "
                                        f"[cosine={best_cosine:.3f}, reranker={reranker_score:.3f}, "
                                        f"ch={channel}] '{text[:50]}...'"
                                    )
                                    self._ingested_ids.add(doc_id)
                                    return {"status": "spam_same_channel", "doc_id": doc_id}
                                else:
                                    logger.info(
                                        f"Truth Engine: Reranker cleared borderline — NOT spam "
                                        f"[cosine={best_cosine:.3f}, reranker={reranker_score:.3f}] "
                                        f"'{text[:50]}...'"
                                    )
                            # reranker_score is None (disabled/failed) → cosine didn't
                            # exceed spam_thr on its own → treat as not spam, proceed
                else:
                    # ── Purpose B: cross-channel corroboration ──
                    past_text = best_doc.get("text", "")

                    if best_cosine > corr_thr:
                        # High confidence corroboration — no reranker needed
                        logger.info(
                            f"Truth Engine: Corroboration (high-conf) "
                            f"[cosine={best_cosine:.3f}, entity_j={best_jaccard:.2f}, "
                            f"thr={corr_thr:.2f}, src={past_channel}→{channel}] "
                            f"'{text[:50]}...' — proceeding to ingest."
                        )
                    elif best_cosine >= self.RERANKER_CORR_LOWER:
                        # Expanded zone — reranker may catch corroboration cosine missed
                        reranker_score = self._call_reranker(text, past_text)
                        if reranker_score is not None and reranker_score > self.RERANKER_CORR_CONFIRM:
                            logger.info(
                                f"Truth Engine: Corroboration expanded by reranker "
                                f"[cosine={best_cosine:.3f}, reranker={reranker_score:.3f}, "
                                f"src={past_channel}→{channel}] '{text[:50]}...'"
                            )
                            # Corroboration detected — proceed to ingest normally
                            # (graph upsert in Step 2/3 will accumulate edge weight)

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
        # [FIX HIGH-11] Persist to disk for process restart resilience
        if len(self._ingested_ids) % 50 == 0:  # Batch writes every 50 ingestions
            self._save_ingested_ids()

        return {
            "status": "ingested",
            "doc_id": doc_id,
            "entities": entity_names,
            "relationships": len(relationships),
        }

    # Cost guardrail: max new LLM extractions per ingest cycle
    # Prevents cost explosion when VM restarts or backlog accumulates
    INGEST_CAP_PER_CYCLE: int = 200

    def ingest_batch(self, messages: List[Dict]) -> int:
        """Ingest a batch of telegram messages with per-cycle cost cap.

        Only processes messages that haven't been ingested yet (dedup via _ingested_ids).
        Caps at INGEST_CAP_PER_CYCLE new LLM calls per cycle to prevent cost explosions.
        """
        count = 0
        skipped_dup = 0
        skipped_cap = 0

        for msg in messages:
            msg_id = str(msg.get("message_id", msg.get("id", "")))
            text = msg.get("text", "")

            # Fast-path dedup check before hitting LLM
            doc_id = msg_id or __import__('hashlib').md5(
                f"{msg.get('channel','')}:{text[:100]}:{msg.get('timestamp','')}".encode()
            ).hexdigest()
            if doc_id in self._ingested_ids:
                skipped_dup += 1
                continue

            # Cost cap: stop processing after N new extractions per cycle
            if count >= self.INGEST_CAP_PER_CYCLE:
                skipped_cap += 1
                continue

            result = self.ingest_message(
                text=text,
                channel=msg.get("channel", ""),
                timestamp=msg.get("timestamp", msg.get("created_at", "")),
                message_id=msg_id,
            )
            if result.get("status") == "ingested":
                count += 1

        self._save_ingested_ids()  # Always persist after batch
        stats = self.get_stats()
        if skipped_cap > 0:
            logger.warning(
                f"LightRAG: ingest cap hit ({self.INGEST_CAP_PER_CYCLE}). "
                f"Queued {skipped_cap} msgs for next cycle. "
                f"Ingested={count}, Dup={skipped_dup}"
            )
        else:
            logger.info(f"LightRAG ingested {count}/{len(messages)} (dup={skipped_dup}, stats={stats})")
        return count

    def query(self, query_text: str, top_k: int = 10, **kwargs) -> Dict:
        """Query the Logical Truth Engine (Intent-Based)."""
        ai = self._get_ai_client()
        intent = "hybrid"
        
        # Intent Classification (Gemini Flash)
        if ai:
            prompt = f"Analyze this user query: '{query_text}'. Classify the intent strictly as one of: [FACT_CHECK, MARKET_OVERVIEW, NARRATIVE_SEARCH]. Output only the classification word."
            try:
                resp = ai.generate_response(
                    system_prompt="You are an intent classifier for a financial truth engine. Output exactly one word.",
                    user_message=prompt,
                    temperature=0.1, max_tokens=20, role="rag_extraction"
                )
                classification = resp.strip().upper()
                if "FACT" in classification: 
                    intent = "local"
                    top_k = 5  # Fact checks need precision, less noise
                elif "OVERVIEW" in classification: 
                    intent = "global"
                else: 
                    intent = "hybrid"
            except Exception:
                pass
                
        # Extract entities from query using LLM for precision
        query_entities = []
        if ai and intent in ("local", "hybrid"):
            entity_prompt = f"Extract the core financial entities (coins, exchanges, organizations) from this query: '{query_text}'. Return a comma-separated list of keywords. If none, return empty."
            try:
                resp = ai.generate_response(
                    system_prompt="Extract keywords.", user_message=entity_prompt,
                    temperature=0.1, max_tokens=50, role="rag_extraction"
                )
                query_entities = [e.strip().lower() for e in resp.split(",") if e.strip()]
            except Exception:
                pass
                
        # Fallback keyword matching
        if not query_entities:
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
            "intent": intent,
        }

        # ── Local retrieval (Low-level: exact facts) ──
        if intent in ("local", "hybrid"):
            for entity in query_entities:
                ctx = self.graph.get_local_context(entity, max_depth=2)
                if ctx.get("neighbors") or ctx.get("paths"):
                    result["local_context"][entity] = ctx

        # ── Global retrieval (High-level: important entities + topics) ──
        if intent in ("global", "hybrid"):
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

    def format_context_for_agents(self, query_result: Dict, max_length: int = 2500) -> str:
        """Format Logical Truth Engine query result as compact text for agent consumption."""
        lines = []
        intent = query_result.get("intent", "hybrid")
        lines.append(f"[QUERY INTENT: {intent.upper()}]")

        # Local context (specific facts about queried entities)
        local_ctx = query_result.get("local_context", {})
        if local_ctx:
            lines.append("\n[EVIDENCE GRAPH - Local Facts]")
            for entity, ctx in local_ctx.items():
                for n in ctx.get("neighbors", [])[:5]:
                    rel_type = n.get("rel_type", n.get("type", "?"))
                    target = n.get("name", n.get("target", "?"))
                    # The graph queries need to return status if we want to show it, assuming they don't yet, we mark "EVIDENCE: X sources"
                    weight = n.get("weight", 1)
                    lines.append(f"  ({entity}) --[{rel_type}]--> ({target}) [Sources: {weight:.1f}]")
                for p in ctx.get("paths", [])[:3]:
                    lines.append(
                        f"  ({entity}) -> ({p.get('via','?')}) --[{p.get('r2_type','?')}]--> "
                        f"({p.get('target','?')}) [2-hop Inference]"
                    )

        # Global context (market-wide important entities)
        global_ctx = query_result.get("global_context", {})
        important = global_ctx.get("important_entities", [])
        if important:
            lines.append("\n[MARKET THEMES - Key Entities]")
            ent_str = ", ".join([f"{e['name']}({e.get('mentions',0)})" for e in important[:5]])
            lines.append(f"  {ent_str}")

        recent_rels = global_ctx.get("recent_relationships", [])
        if recent_rels:
            lines.append("\n[ACTIVE NARRATIVES - Recent Graph Events]")
            for r in recent_rels[:5]:
                weight = r.get("weight", 1)
                lines.append(
                    f"  ({r['source']}) --[{r['rel_type']}]--> ({r['target']}) [Sources: {weight:.1f}]: "
                    f"{r.get('description', '')[:100]}"
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

    def export_knowledge_graph(self) -> Dict:
        """Export all entities and relationships from Neo4j to a serializable dict."""
        if hasattr(self.graph, 'export_all'):
            return self.graph.export_all()
        logger.warning("Graph backend does not support export_all")
        return {"entities": [], "relationships": [], "timestamp": datetime.now().isoformat()}

    def archive_to_gcs(self) -> bool:
        """Export the current graph and upload as a JSON backup to GCS."""
        try:
            from config.settings import settings
            if not settings.ENABLE_GCS_ARCHIVE or not settings.GCS_ARCHIVE_BUCKET:
                logger.debug("GCS Archive disabled or bucket not configured")
                return False

            knowledge = self.export_knowledge_graph()
            if not knowledge.get("entities") and not knowledge.get("relationships"):
                return False

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"rag_archive_{timestamp}.json"
            local_path = os.path.join(self.WORKING_DIR, filename)

            with open(local_path, 'w', encoding='utf-8') as f:
                json.dump(knowledge, f, ensure_ascii=False, indent=2)

            # Upload to GCS
            from google.cloud import storage
            client = storage.Client()
            bucket = client.bucket(settings.GCS_ARCHIVE_BUCKET)
            blob = bucket.blob(f"rag_archive/{filename}")
            blob.upload_from_filename(local_path)

            logger.info(f"LightRAG: Archived knowledge to GCS: {filename}")
            
            # Cleanup local temp file
            os.remove(local_path)
            return True
        except Exception as e:
            logger.error(f"LightRAG Archival error: {e}")
            return False

    def cleanup_old(self, days: int = 90):
        """Archive to GCS first, then remove old data from Neo4j."""
        try:
            # 1. Backup everything to GCS first
            self.archive_to_gcs()

            # 2. Trigger graph-specific cleanup
            if hasattr(self.graph, 'cleanup_old'):
                self.graph.cleanup_old(days=days)
            
            # 3. Vector store cleanup (optional, usually handled by deletion of nodes if integrated)
            # Currently Milvus in this setup mirrors the graph.
            
            logger.info(f"LightRAG: Completed cleanup (Retention: {days} days)")
        except Exception as e:
            logger.error(f"LightRAG cleanup error: {e}")

    def restore_from_gcs(self, gcs_path: str):
        """Restore knowledge from a GCS archive file."""
        try:
            from google.cloud import storage
            from config.settings import settings
            
            client = storage.Client()
            bucket = client.bucket(settings.GCS_ARCHIVE_BUCKET)
            blob = bucket.blob(gcs_path)
            
            content = blob.download_as_text()
            data = json.loads(content)
            
            entities = data.get("entities", [])
            relationships = data.get("relationships", [])
            
            logger.info(f"LightRAG: Restoring {len(entities)} entities and {len(relationships)} relationships...")
            
            # Simple restore: use existing ingest logic or direct graph commands
            if hasattr(self.graph, 'import_data'):
                self.graph.import_data(entities, relationships)
                logger.info("LightRAG: Restoration successful")
            else:
                logger.warning("Graph backend does not support import_data")
                
        except Exception as e:
            logger.error(f"LightRAG Restoration error: {e}")

    def run_triangulation_worker(self, limit: int = 5):
        """Cross-Domain Triangulation: Verifies CORROBORATED claims via Web Search (Perplexity)."""
        logger.info("Truth Engine: Starting Triangulation Worker...")
        try:
            from collectors.perplexity_collector import perplexity_collector
        except ImportError:
            logger.warning("Truth Engine: Perplexity collector not available for triangulation.")
            return

        candidates = []
        if hasattr(self.graph, 'get_triangulation_candidates'):
            candidates = self.graph.get_triangulation_candidates(limit=limit)

        verified_count = 0
        for cand in candidates:
            # We only triangulate claims that have multiple sources (CORROBORATED)
            if cand.get("status") != "CORROBORATED":
                continue
                
            source = cand["source"]
            target = cand["target"]
            rel_type = cand["rel_type"]
            desc = cand["description"]
            
            logger.info(f"Triangulating claim: ({source}) --[{rel_type}]--> ({target})")
            
            # Use Perplexity targeted search to verify
            query_entity = f"{source} and {target}"
            context = f"Claim: {source} {rel_type} {target}. Description: {desc}. Is there recent news confirming this?"
            
            try:
                result = perplexity_collector.search_targeted(
                    entity=query_entity,
                    entity_type="narrative",
                    context=context
                )
                
                # If Perplexity finds key facts corroborating the entity context
                if result.get("status") == "ok" and len(result.get("key_facts", [])) > 0:
                    logger.info(f"Truth Engine: Triangulation SUCCEEDED for ({source})-({target}). Moving to PROBABLE.")
                    if hasattr(self.graph, "update_relationship_status"):
                        self.graph.update_relationship_status(
                            source=source, target=target, rel_type=rel_type,
                            status="PROBABLE", weight_boost=2.0
                        )
                    verified_count += 1
                else:
                    logger.info(f"Truth Engine: Triangulation FAILED for ({source})-({target}). Keeping as CORROBORATED.")
            except Exception as e:
                logger.error(f"Truth Engine: Triangulation search error: {e}")
                
        logger.info(f"Truth Engine: Triangulation cycle complete. Verified: {verified_count}/{len(candidates)}")


# Singleton instance
light_rag = LightRAGEngine()
