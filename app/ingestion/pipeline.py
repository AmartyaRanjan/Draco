import os
from collections.abc import Iterable, Iterator

from app.db.opensearch import delete_stale_versions as delete_stale_text
from app.db.opensearch import index_chunks
from app.db.qdrant import delete_stale_versions as delete_stale_vectors
from app.db.qdrant import upsert_chunks
from app.ingestion.embed import embed_chunks
from app.ingestion.headers import apply_context_headers
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


def _search_record(chunk: ChunkRecord, record: dict) -> dict:
    """OpenSearch document: the payload plus `search_text` (header + text) when a header is set.
    `text` stays the displayed/cited text in both stores."""
    if not chunk.context_header:
        return record
    return {**record, "search_text": chunk.index_text()}


def ingest_chunks(
    document_id: str,
    chunks: Iterable[ChunkRecord],
    document_version: str = "1",
    title: str | None = None,
) -> dict:
    """Index a chunk stream in both stores, batch by batch, then drop superseded versions."""
    total_chunks = 0

    for batch in _batches(apply_context_headers(chunks, title), CHUNK_BATCH_SIZE):
        records = [chunk.to_payload() for chunk in batch]

        # Generate dense embeddings for only the current batch
        # (header + text when contextual headers are on).
        vectors = embed_chunks(batch)

        if len(vectors) != len(records):
            raise RuntimeError(
                "Embedding count does not match chunk count."
            )

        # Each backend uses the same chunk identity.
        # If one write fails, the worker fails and a retry can
        # overwrite the same IDs in the successful backend.
        upsert_chunks(records, vectors)
        index_chunks([_search_record(c, r) for c, r in zip(batch, records, strict=True)])

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


def ingest_converted_document(
    document_id: str,
    document,
    document_version: str = "1",
    title: str | None = None,
) -> dict:
    """
    Chunk a converted (Docling) document and index each batch in both stores.

    Completion is the worker's responsibility. This function returns
    only after all batches have been written successfully.
    """
    # Imported here so the Markdown/text path never needs Docling.
    from app.ingestion.chunk import iter_chunks

    chunks = iter_chunks(
        document=document,
        document_id=document_id,
        document_version=document_version,
    )
    return ingest_chunks(document_id, chunks, document_version, title)


def ingest_text_document(
    document_id: str,
    text: str,
    fmt: str,
    document_version: str = "1",
    title: str | None = None,
) -> dict:
    """Chunk Markdown (`fmt="markdown"`) or plain text (`fmt="text"`) without Docling and index it."""
    from app.ingestion.text_chunk import iter_text_chunks

    chunks = iter_text_chunks(text, document_id, document_version, fmt=fmt)
    return ingest_chunks(document_id, chunks, document_version, title)
