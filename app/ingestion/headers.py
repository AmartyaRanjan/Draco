"""Contextual chunk headers.

A chunk such as "CGPA: 8.7" is unfindable once it is separated from the document and section it
belongs to (production.md section 6, item 3). Before embedding and BM25 indexing, each chunk gets a
short header naming its document and section path:

    Document: Annual report 2025
    Section: Results > Revenue

The header is stored separately (`context_header`); `text`, which is what is shown and cited,
is unchanged. OpenSearch indexes `search_text` = header + text next to `text`; Qdrant embeds the
same header + text. Existing indexes do not change until their documents are re-ingested
(`scripts/reindex.py`).

INGESTION_CONTEXT_HEADERS=true (default) turns it on for new ingests. The chunker reserves
INGESTION_CONTEXT_HEADER_RESERVE_TOKENS of the embedding window for the header, so header + chunk
still fits the embedder's max length; headers are cut to INGESTION_CONTEXT_HEADER_MAX_CHARS.
"""
import os
import re
from collections.abc import Iterable, Iterator
from pathlib import Path

from app.ingestion.schema import ChunkRecord


def headers_enabled() -> bool:
    return os.getenv("INGESTION_CONTEXT_HEADERS", "true").lower() == "true"


HEADER_RESERVE_TOKENS = max(0, int(os.getenv("INGESTION_CONTEXT_HEADER_RESERVE_TOKENS", "64")))
HEADER_MAX_CHARS = max(40, int(os.getenv("INGESTION_CONTEXT_HEADER_MAX_CHARS", "200")))


def _clip(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def title_from_filename(filename: str | None) -> str | None:
    """'2DOF_research-roadmap_v2.pdf' -> '2DOF research-roadmap v2'."""
    if not filename:
        return None
    stem = Path(str(filename)).stem
    stem = re.sub(r"[_]+", " ", stem)
    stem = re.sub(r"\s+", " ", stem).strip()
    return stem or None


def build_context_header(title: str | None, section_path: str | None, max_chars: int = HEADER_MAX_CHARS) -> str | None:
    parts = []
    if title and title.strip():
        parts.append(f"Document: {_clip(title, max_chars // 2)}")
    if section_path and section_path.strip():
        room = max(20, max_chars - sum(len(p) for p in parts) - 10)
        parts.append(f"Section: {_clip(section_path, room)}")
    if not parts:
        return None
    return "\n".join(parts)


def apply_context_headers(chunks: Iterable[ChunkRecord], title: str | None,
                          enabled: bool | None = None) -> Iterator[ChunkRecord]:
    """Set `context_header` on each chunk (lazily; the chunk stream stays bounded)."""
    on = headers_enabled() if enabled is None else enabled
    for chunk in chunks:
        if on:
            chunk.context_header = build_context_header(title, chunk.section_path)
        yield chunk


def chunk_token_budget(max_tokens: int, enabled: bool | None = None) -> int:
    """Chunk size that leaves room for the header inside the embedder's window."""
    on = headers_enabled() if enabled is None else enabled
    if not on:
        return max_tokens
    return max(64, max_tokens - HEADER_RESERVE_TOKENS)
