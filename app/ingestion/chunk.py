
import hashlib
import os
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import lru_cache

from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import (
    HuggingFaceTokenizer,
)

from app.ingestion.schema import ChunkRecord


MODEL_ID = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
MAX_TOKENS = int(os.getenv("EMBEDDING_MAX_LENGTH", "512"))
# Chunks with less real text than this are merged into a neighbour.
MIN_CHUNK_CHARS = int(os.getenv("INGESTION_MIN_CHUNK_CHARS", "50"))

# Docling's marker for formulas it could not decode; it carries no content.
_PLACEHOLDER = re.compile(r"<!--\s*formula-not-decoded\s*-->")


@dataclass
class _RawChunk:
    text: str
    page_start: int | None
    page_end: int | None
    section_path: str | None


@lru_cache(maxsize=1)
def build_chunker() -> HybridChunker:
    # Loading the tokenizer costs seconds, so build one chunker per process.
    tokenizer = HuggingFaceTokenizer.from_pretrained(
        MODEL_ID,
        max_tokens=MAX_TOKENS,
    )

    return HybridChunker(
        tokenizer=tokenizer,
        max_tokens=MAX_TOKENS,
        merge_peers=True,
    )


def _page_range(chunk) -> tuple[int | None, int | None]:
    """Extract page provenance from Docling chunk metadata."""
    page_numbers = []

    for item in getattr(chunk.meta, "doc_items", []) or []:
        for provenance in getattr(item, "prov", []) or []:
            page_no = getattr(provenance, "page_no", None)
            if page_no is not None:
                page_numbers.append(int(page_no))

    if not page_numbers:
        return None, None

    return min(page_numbers), max(page_numbers)


def _section_path(chunk) -> str | None:
    headings = getattr(chunk.meta, "headings", None)

    if not headings:
        return None

    return " > ".join(str(heading) for heading in headings)


def _content(text: str) -> str:
    """The chunk text without Docling placeholders, which hold no content."""
    return _PLACEHOLDER.sub("", text).strip()


def _is_small(chunk: _RawChunk) -> bool:
    return len(_content(chunk.text)) < MIN_CHUNK_CHARS


def _join(first: _RawChunk, second: _RawChunk) -> _RawChunk:
    """Append `second` to `first`, dropping placeholders from the added text."""
    pages = [
        page
        for page in (first.page_start, first.page_end, second.page_start, second.page_end)
        if page is not None
    ]
    added = _content(second.text)
    return _RawChunk(
        text=f"{first.text}\n{added}" if added else first.text,
        page_start=min(pages) if pages else None,
        page_end=max(pages) if pages else None,
        section_path=first.section_path,
    )


def merge_small_chunks(
    chunks: Iterator[_RawChunk],
    count_tokens: Callable[[str], int],
    max_tokens: int = MAX_TOKENS,
) -> Iterator[_RawChunk]:
    """Fold chunks with almost no text into the previous chunk.

    A small chunk at the very start joins the chunk after it instead. A merge
    that would exceed `max_tokens` is skipped, because the embedder truncates
    longer text and the tail would silently go unsearchable; that small chunk
    stays on its own.
    """
    current = None
    is_first = True

    for chunk in chunks:
        if current is None:
            current = chunk
            continue

        if _is_small(chunk):
            merged = _join(current, chunk)
        elif is_first and _is_small(current):
            merged = _join(current, chunk)
            merged.section_path = chunk.section_path
        else:
            merged = None

        if merged is not None and count_tokens(merged.text) <= max_tokens:
            current = merged
        else:
            yield current
            is_first = False
            current = chunk

    if current is not None:
        yield current


def _raw_chunks(document, chunker) -> Iterator[_RawChunk]:
    for chunk in chunker.chunk(document):
        text = chunker.contextualize(chunk=chunk).strip()

        if not text:
            continue

        page_start, page_end = _page_range(chunk)
        yield _RawChunk(text, page_start, page_end, _section_path(chunk))


def iter_chunks(
    document,
    document_id: str,
    document_version: str = "1",
) -> Iterator[ChunkRecord]:
    """
    Yield normalized chunks from a Docling document.

    Chunks with almost no text are merged into a neighbour first. Chunk IDs
    remain stable for the same document version, chunking configuration, and
    ordered chunk content.
    """
    chunker = build_chunker()

    def count_tokens(text: str) -> int:
        return chunker.tokenizer.count_tokens(text=text)

    merged = merge_small_chunks(_raw_chunks(document, chunker), count_tokens)

    for index, chunk in enumerate(merged):
        identity = (
            f"{document_id}|{document_version}|{index}|{chunk.text}"
        )
        chunk_id = hashlib.sha256(
            identity.encode("utf-8")
        ).hexdigest()

        yield ChunkRecord(
            chunk_id=chunk_id,
            document_id=document_id,
            document_version=document_version,
            chunk_index=index,
            text=chunk.text,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            section_path=chunk.section_path,
        )
