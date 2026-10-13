
import os
from uuid import UUID, uuid5

from qdrant_client import QdrantClient, models


COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "rag_chunks")
VECTOR_SIZE = int(os.getenv("EMBEDDING_DIMENSIONS", "1024"))
NAMESPACE = UUID("e746b09d-4e6b-46f5-a66a-0df64fbacdd0")

_client = None


def get_qdrant_client() -> QdrantClient:
    """Return one reusable Qdrant client per process."""
    global _client

    if _client is None:
        url = os.getenv("QDRANT_URL", "http://localhost:6333")
        api_key = os.getenv("QDRANT_API_KEY") or None

        _client = QdrantClient(
            url=url,
            api_key=api_key,
            timeout=30,
        )

    return _client


def ensure_collection() -> None:
    """Create the collection and document-scoped payload indexes if missing."""
    client = get_qdrant_client()

    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE,
            ),
        )

    # Metadata indexes support document-scoped filtering. Collections created
    # before the indexes were added still need them, so ensure both fields on
    # every call. Re-creating an existing index is a no-op.
    for field_name in ("document_id", "document_version"):
        try:
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field_name,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        except Exception as exc:
            message = str(exc).lower()
            if "exists" in message or "already" in message:
                continue
            raise


def point_id_for_chunk(chunk_id: str) -> str:
    """Convert a stable chunk ID to a deterministic Qdrant UUID."""
    return str(uuid5(NAMESPACE, chunk_id))


def upsert_chunks(chunks: list[dict], vectors: list[list[float]]) -> None:
    """Store one batch of chunks and their dense vectors."""
    if len(chunks) != len(vectors):
        raise ValueError("Each chunk must have exactly one vector.")

    if not chunks:
        return

    ensure_collection()

    points = []

    for chunk, vector in zip(chunks, vectors):
        if len(vector) != VECTOR_SIZE:
            raise ValueError(
                f"Expected {VECTOR_SIZE} dimensions; got {len(vector)}."
            )

        payload = dict(chunk)

        points.append(
            models.PointStruct(
                id=point_id_for_chunk(chunk["chunk_id"]),
                vector=vector,
                payload=payload,
            )
        )

    get_qdrant_client().upsert(
        collection_name=COLLECTION_NAME,
        points=points,
        wait=True,
    )


def delete_stale_versions(document_id: str, keep_version: str) -> None:
    """Remove chunks of this document written under any other version."""
    get_qdrant_client().delete(
        collection_name=COLLECTION_NAME,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document_id),
                    )
                ],
                must_not=[
                    models.FieldCondition(
                        key="document_version",
                        match=models.MatchValue(value=keep_version),
                    )
                ],
            )
        ),
        wait=True,
    )
