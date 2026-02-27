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
                     weight=weight, timestamp=timestamp)
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

        # Step 0: Semantic Bot Spam Check using Vector Store (Independence Check)
        embedding = self._get_embedding(text[:512])
        if embedding is not None:
            # Search for highly similar past messages to detect verbatim spam
            similar_docs = []
            if isinstance(self.vector_store, InMemoryFallback):
                similar_docs = self.vector_store.search_vectors(embedding, top_k=1)
            else:
                similar_docs = self.vector_store.search(embedding, top_k=1)
                
            if similar_docs and similar_docs[0].get("score", 0) > 0.96:
                past_doc = similar_docs[0]
                past_channel = past_doc.get("channel", "")
                if past_channel == channel:
                    # Ignore exact duplicates from the same channel (Spam)
                    logger.info(f"Truth Engine: Discarding exact forward from same channel '{text[:50]}...'")
                    self._ingested_ids.add(doc_id)
                    return {"status": "spam_same_channel", "doc_id": doc_id}
                else:
                    logger.info(f"Truth Engine: High similarity cross-channel message '{text[:50]}...'. Will extract triplets for corroboration.")

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

    def query(self, query_text: str, top_k: int = 10) -> Dict:
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
