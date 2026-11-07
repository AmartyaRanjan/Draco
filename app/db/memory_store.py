"""Per-user long-term memory storage: Qdrant (dense) + OpenSearch (BM25).

Memory lives in its own Qdrant collection and OpenSearch index, never in the document-chunk ones,
and every read, write and delete carries a `user_id` filter: there is no code path that queries
memory without an owner. Point/document IDs are derived from (user_id, memory_id), so one user can
never overwrite another user's item even with a colliding memory_id.
"""
import os
from uuid import UUID, uuid5

from qdrant_client import models

from app.db import opensearch as os_db
from app.db import qdrant as qd_db

MEMORY_COLLECTION = os.getenv("MEMORY_QDRANT_COLLECTION", "rag_user_memory")
MEMORY_INDEX = os.getenv("MEMORY_OPENSEARCH_INDEX", "rag_user_memory")
_NAMESPACE = UUID("5b0f3f58-2f43-4b8e-9a63-6f1f0c7f6a21")


class MemoryStoreConfigError(ValueError):
    """Memory would share a collection/index with document chunks."""


def _require_owner(user_id) -> str:
    if not isinstance(user_id, str) or not user_id.strip():
        raise PermissionError("memory access requires an authenticated user_id")
    return user_id


def item_key(user_id: str, memory_id: str) -> str:
    """Store-level ID: owner-bound, deterministic, a valid Qdrant UUID."""
    return str(uuid5(_NAMESPACE, f"{user_id}\x1f{memory_id}"))


class MemoryIndex:
    def __init__(self, qdrant=None, opensearch=None, collection: str | None = None, index: str | None = None,
                 vector_size: int | None = None):
        self._qdrant, self._opensearch = qdrant, opensearch
        self.collection = collection or MEMORY_COLLECTION
        self.index = index or MEMORY_INDEX
        self.vector_size = vector_size or qd_db.VECTOR_SIZE
        if self.collection == qd_db.COLLECTION_NAME:
            raise MemoryStoreConfigError("MEMORY_QDRANT_COLLECTION must differ from QDRANT_COLLECTION.")
        if self.index == os_db.INDEX_NAME:
            raise MemoryStoreConfigError("MEMORY_OPENSEARCH_INDEX must differ from OPENSEARCH_INDEX.")
        self._ready = False

    @property
    def qdrant(self):
        return self._qdrant or qd_db.get_qdrant_client()

    @property
    def opensearch(self):
        return self._opensearch or os_db.get_opensearch_client()

    # ---- schema ------------------------------------------------------------------------
    def ensure(self) -> None:
        if self._ready:
            return
        q = self.qdrant
        if not q.collection_exists(self.collection):
            q.create_collection(collection_name=self.collection, vectors_config=models.VectorParams(
                size=self.vector_size, distance=models.Distance.COSINE))
        for field_name in ("user_id", "memory_id"):
            try:
                q.create_payload_index(collection_name=self.collection, field_name=field_name,
                                       field_schema=models.PayloadSchemaType.KEYWORD)
            except Exception as exc:
                if "exist" not in str(exc).lower() and "already" not in str(exc).lower():
                    raise
        o = self.opensearch
        if not o.indices.exists(index=self.index):
            o.indices.create(index=self.index, body={"mappings": {"properties": {
                "memory_id": {"type": "keyword"}, "user_id": {"type": "keyword"},
                "text": {"type": "text"}, "kind": {"type": "keyword"}, "created_at": {"type": "date"}}}})
        self._ready = True

    # ---- writes ------------------------------------------------------------------------
    def upsert(self, user_id: str, memory_id: str, text: str, vector: list[float], *, kind: str,
               created_at: str) -> None:
        user_id = _require_owner(user_id)
        if len(vector) != self.vector_size:
            raise ValueError(f"Expected {self.vector_size} dimensions; got {len(vector)}.")
        self.ensure()
        payload = {"memory_id": memory_id, "user_id": user_id, "text": text, "kind": kind, "created_at": created_at}
        key = item_key(user_id, memory_id)
        self.qdrant.upsert(collection_name=self.collection,
                           points=[models.PointStruct(id=key, vector=vector, payload=payload)], wait=True)
        self.opensearch.index(index=self.index, id=key, body=payload, refresh=True)

    def delete(self, user_id: str, memory_id: str) -> None:
        user_id = _require_owner(user_id)
        self.ensure()
        self.qdrant.delete(collection_name=self.collection, points_selector=models.FilterSelector(
            filter=self._owner_filter(user_id, memory_id)), wait=True)
        self.opensearch.delete_by_query(index=self.index, body={"query": {"bool": {"filter": [
            {"term": {"user_id": user_id}}, {"term": {"memory_id": memory_id}}]}}},
            refresh=True, conflicts="proceed")

    def delete_user(self, user_id: str) -> None:
        """Erase everything one user stored (right-to-erasure)."""
        user_id = _require_owner(user_id)
        self.ensure()
        self.qdrant.delete(collection_name=self.collection, points_selector=models.FilterSelector(
            filter=self._owner_filter(user_id)), wait=True)
        self.opensearch.delete_by_query(index=self.index, body={"query": {"bool": {"filter": [
            {"term": {"user_id": user_id}}]}}}, refresh=True, conflicts="proceed")

    # ---- reads -------------------------------------------------------------------------
    @staticmethod
    def _owner_filter(user_id: str, memory_id: str | None = None) -> models.Filter:
        must = [models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))]
        if memory_id is not None:
            must.append(models.FieldCondition(key="memory_id", match=models.MatchValue(value=memory_id)))
        return models.Filter(must=must)

    def dense_search(self, user_id: str, vector: list[float], limit: int, timeout_s: float = 5) -> list[dict]:
        user_id = _require_owner(user_id)
        self.ensure()
        res = self.qdrant.query_points(collection_name=self.collection, query=vector, limit=limit,
                                       query_filter=self._owner_filter(user_id), with_payload=True,
                                       timeout=max(1, int(timeout_s)))
        # Defence in depth: drop anything the store returned for another owner.
        return [{**p.payload, "score": float(p.score)} for p in res.points
                if (p.payload or {}).get("user_id") == user_id]

    def lexical_search(self, user_id: str, query: str, limit: int, timeout_s: float = 5) -> list[dict]:
        user_id = _require_owner(user_id)
        self.ensure()
        body = {"size": limit, "_source": ["memory_id", "user_id", "text", "kind", "created_at"],
                "query": {"bool": {"must": [{"match": {"text": query}}],
                                   "filter": [{"term": {"user_id": user_id}}]}}}
        res = self.opensearch.search(index=self.index, body=body, request_timeout=timeout_s)
        return [{**h["_source"], "score": float(h["_score"])} for h in res["hits"]["hits"]
                if h["_source"].get("user_id") == user_id]

    def count(self, user_id: str) -> int:
        user_id = _require_owner(user_id)
        self.ensure()
        return int(self.qdrant.count(collection_name=self.collection, count_filter=self._owner_filter(user_id),
                                     exact=True).count)

    def list_items(self, user_id: str, limit: int = 100) -> list[dict]:
        user_id = _require_owner(user_id)
        self.ensure()
        points, _ = self.qdrant.scroll(collection_name=self.collection, scroll_filter=self._owner_filter(user_id),
                                       limit=limit, with_payload=True)
        return [dict(p.payload) for p in points if (p.payload or {}).get("user_id") == user_id]
