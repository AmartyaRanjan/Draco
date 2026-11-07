"""Lightweight chunking for Markdown and plain text (no Docling).

Markdown: ATX (`#`) and setext (`===` / `---`) headings build the section path, fenced code blocks
are kept intact and never parsed for headings, YAML front matter is dropped. Plain text: paragraphs
only. Paragraphs are packed greedily into chunks of at most `max_tokens` (the embedder's window,
minus the contextual-header reserve), a chunk never spans two sections unless it is too small to
stand alone, and an oversized paragraph is split on sentences, then lines, then words.

Like Docling's `contextualize()` for PDFs, each chunk's text starts with its heading path, so a
Markdown chunk reads the same way a PDF chunk does. No page numbers exist for these formats.
"""
import hashlib
import os
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass

from app.ingestion.headers import chunk_token_budget
from app.ingestion.schema import ChunkRecord

MODEL_ID = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
MAX_TOKENS = int(os.getenv("EMBEDDING_MAX_LENGTH", "512"))
MIN_CHUNK_CHARS = int(os.getenv("INGESTION_MIN_CHUNK_CHARS", "50"))

_ATX = re.compile(r"^ {0,3}(#{1,6})(?:[ \t]+(.*?))?(?:[ \t]+#+)?[ \t]*$")
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_SETEXT = re.compile(r"^ {0,3}(=+|-+)[ \t]*$")
_SENTENCE = re.compile(r"(?<=[.!?。！？।])\s+")


@dataclass
class _Block:
    section: tuple[str, ...]
    text: str


def _strip_front_matter(text: str) -> str:
    if text.startswith("---\n") or text.startswith("---\r\n"):
        end = text.find("\n---", 4)
        if 0 < end < 8192:
            nl = text.find("\n", end + 4)
            return text[nl + 1:] if nl >= 0 else ""
    return text


def _clean_heading(text: str) -> str:
    text = re.sub(r"[*_`]+", "", text or "")
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)  # [label](url) -> label
    return re.sub(r"\s+", " ", text).strip()


def markdown_blocks(text: str) -> Iterator[_Block]:
    """Paragraph blocks with the heading path in force at each one."""
    lines = _strip_front_matter(text.replace("\r\n", "\n").replace("\r", "\n")).split("\n")
    path: list[tuple[int, str]] = []
    para: list[str] = []
    fence: str | None = None

    def section() -> tuple[str, ...]:
        return tuple(h for _, h in path)

    def flush():
        body = "\n".join(para).strip()
        para.clear()
        if body:
            yield _Block(section(), body)

    def set_heading(level: int, title: str):
        while path and path[-1][0] >= level:
            path.pop()
        if title:
            path.append((level, title))

    for line in lines:
        if fence is not None:
            para.append(line)
            if line.strip().startswith(fence):
                fence = None
            continue
        m = _FENCE.match(line)
        if m:
            fence = m.group(1)[0] * 3
            para.append(line)
            continue
        m = _ATX.match(line)
        if m:
            yield from flush()
            set_heading(len(m.group(1)), _clean_heading(m.group(2) or ""))
            continue
        m = _SETEXT.match(line)
        if m and len(para) == 1 and para[0].strip():
            title = _clean_heading(para.pop())
            set_heading(1 if m.group(1)[0] == "=" else 2, title)
            continue
        if not line.strip():
            yield from flush()
            continue
        para.append(line)
    yield from flush()


def text_blocks(text: str) -> Iterator[_Block]:
    for para in re.split(r"\n\s*\n", text.replace("\r\n", "\n").replace("\r", "\n")):
        body = para.strip()
        if body:
            yield _Block((), body)


def _split_oversized(text: str, count: Callable[[str], int], limit: int) -> list[str]:
    """Split one paragraph that exceeds `limit` on sentences, then lines, then words."""
    if count(text) <= limit:
        return [text]
    for splitter in (_SENTENCE, re.compile(r"\n+"), re.compile(r"\s+")):
        parts = [p for p in splitter.split(text) if p.strip()]
        if len(parts) > 1:
            break
    else:
        # One unbreakable token run: cut by characters.
        step = max(1, len(text) * limit // max(1, count(text)))
        return [text[i:i + step] for i in range(0, len(text), step)]
    out, cur = [], ""
    for part in parts:
        cand = f"{cur} {part}".strip() if cur else part
        if count(cand) <= limit:
            cur = cand
            continue
        if cur:
            out.append(cur)
        if count(part) > limit:
            out.extend(_split_oversized(part, count, limit))
            cur = ""
        else:
            cur = part
    if cur:
        out.append(cur)
    return out


@dataclass
class _Piece:
    section: tuple[str, ...]
    body: str


def _render(piece: _Piece) -> str:
    """Heading path first, as Docling's contextualize() does for PDF chunks."""
    return "\n".join([*piece.section, piece.body]) if piece.section else piece.body


def pack_blocks(blocks: Iterator[_Block], count: Callable[[str], int], max_tokens: int,
                min_chars: int = MIN_CHUNK_CHARS) -> Iterator[_Piece]:
    cur: _Piece | None = None
    pending: _Piece | None = None  # last full piece, held so a tiny tail can still join it

    def emit(piece: _Piece):
        nonlocal pending
        if pending is not None:
            yield pending
        pending = piece

    for block in blocks:
        heading_cost = count("\n".join(block.section)) + 1 if block.section else 0
        limit = max(16, max_tokens - heading_cost)
        for part in _split_oversized(block.text, count, limit):
            if cur is None:
                cur = _Piece(block.section, part)
                continue
            same_section = cur.section == block.section
            small = len(cur.body) < min_chars
            if not same_section and not small:
                yield from emit(cur)
                cur = _Piece(block.section, part)
                continue
            joined = _Piece(block.section if not same_section else cur.section, f"{cur.body}\n\n{part}")
            if count(_render(joined)) <= max_tokens:
                cur = joined
            else:
                yield from emit(cur)
                cur = _Piece(block.section, part)
    if cur is not None:
        if pending is not None and len(cur.body) < min_chars and cur.section == pending.section:
            joined = _Piece(pending.section, f"{pending.body}\n\n{cur.body}")
            if count(_render(joined)) <= max_tokens:
                pending, cur = joined, None
        if pending is not None:
            yield pending
        if cur is not None:
            yield cur
    elif pending is not None:
        yield pending


_COUNTER = None


def default_token_counter() -> Callable[[str], int]:
    """BGE-M3 tokenizer (CPU) when available, else the conservative char estimate."""
    global _COUNTER
    if _COUNTER is None:
        from app.retrieval.context import TokenCounter
        _COUNTER = TokenCounter(MODEL_ID)
    return _COUNTER.count


def iter_text_chunks(text: str, document_id: str, document_version: str = "1", fmt: str = "markdown",
                     count_tokens: Callable[[str], int] | None = None,
                     max_tokens: int | None = None) -> Iterator[ChunkRecord]:
    """ChunkRecords for Markdown (`fmt="markdown"`) or plain text (`fmt="text"`); same ID scheme as PDFs."""
    if fmt not in ("markdown", "text"):
        raise ValueError(f"unsupported text format {fmt!r}")
    count = count_tokens or default_token_counter()
    budget = max_tokens or chunk_token_budget(MAX_TOKENS)
    blocks = markdown_blocks(text) if fmt == "markdown" else text_blocks(text)
    for index, piece in enumerate(pack_blocks(blocks, count, budget)):
        body = _render(piece)
        chunk_id = hashlib.sha256(f"{document_id}|{document_version}|{index}|{body}".encode()).hexdigest()
        yield ChunkRecord(
            chunk_id=chunk_id, document_id=document_id, document_version=document_version,
            chunk_index=index, text=body, page_start=None, page_end=None,
            section_path=" > ".join(piece.section) if piece.section else None,
        )
