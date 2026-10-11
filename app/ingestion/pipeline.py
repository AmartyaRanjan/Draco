
import os
from collections.abc import Iterator

from app.db.opensearch import delete_stale_versions as delete_stale_text
from app.db.opensearch import index_chunks
from app.db.qdrant import delete_stale_versions as delete_stale_vectors
from app.db.qdrant import upsert_chunks
from app.ingestion.chunk import iter_chunks
from app.ingestion.embed import embed_chunks
from app.ingestion.schema import ChunkRecord


CHUNK_BATCH_SIZE = int(
    os.getenv("INGESTION_CHUNK_BATCH_SIZE", "16")
)


def _batches(
    chunks: Iterator[ChunkRecord],
    batch_size: int,
) -> Iterator[list[ChunkRecord]]:
    """Yield bounded batches from a chunk iterator."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    batch = []

    for chunk in chunks:
        batch.append(chunk)

        if len(batch) >= batch_size:
            yield batch
            batch = []

    if batch:
        yield batch


def ingest_converted_document(
    document_id: str,
    document,
    document_version: str = "1",
) -> dict:
    """
    Chunk a converted document and index each batch in both stores.

    Completion is the worker's responsibility. This function returns
    only after all batches have been written successfully.
    """
    chunks = iter_chunks(
        document=document,
        document_id=document_id,
        document_version=document_version,
    )

    total_chunks = 0

    for batch in _batches(chunks, CHUNK_BATCH_SIZE):
        records = [chunk.to_payload() for chunk in batch]

        # Generate dense embeddings for only the current batch.
        vectors = embed_chunks(batch)

        if len(vectors) != len(records):
            raise RuntimeError(
                "Embedding count does not match chunk count."
            )

        # Each backend uses the same chunk identity.
        # If one write fails, the worker fails and a retry can
        # overwrite the same IDs in the successful backend.
        upsert_chunks(records, vectors)
        index_chunks(records)

        total_chunks += len(records)

    if total_chunks == 0:
        raise ValueError(
            "No text chunks were produced from the document."
        )

    # Chunk IDs include the version, so a new version never overwrites an
    # old one. Drop the superseded chunks only now that every batch of the
    # new version is written; a failed run leaves the old version searchable.
    delete_stale_vectors(document_id, document_version)
    delete_stale_text(document_id, document_version)

    return {
        "document_id": document_id,
        "chunk_count": total_chunks,
        "document_version": document_version,
    }
