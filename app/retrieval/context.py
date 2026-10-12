"""Evidence packaging under a hard token budget."""
from dataclasses import dataclass, field

from app.retrieval.schema import Candidate


@dataclass
class Evidence:
    label: str  # S1, S2 ... used as citation marker
    candidate: Candidate
    filename: str | None
    text_block: str
    tokens: int


@dataclass
class BuiltContext:
    items: list[Evidence] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)  # chunk_ids over budget
    tokens: int = 0


class TokenCounter:
    """Uses a HF tokenizer when available; otherwise a conservative char estimate."""

    def __init__(self, tokenizer_id: str | None):
        self._id, self._tok = tokenizer_id, None

    def count(self, text: str) -> int:
        if self._tok is None and self._id:
            try:
                from transformers import AutoTokenizer
                self._tok = AutoTokenizer.from_pretrained(self._id)
            except Exception:
                self._id = None
        if self._tok is not None:
            return len(self._tok.encode(text, add_special_tokens=False))
        return len(text) // 2 + 1


def pages(c: Candidate) -> str:
    if c.page_start is None:
        return "unknown"
    return str(c.page_start) if c.page_end in (None, c.page_start) else f"{c.page_start}-{c.page_end}"


def format_block(label, c: Candidate, filename) -> str:
    return (f"[{label}] source: {filename or 'unknown'} | document_id: {c.document_id} | "
            f"pages: {pages(c)} | section: {c.section_path or 'n/a'} | "
            f"relevance: {c.selection}\n{c.text}\n")


def build_context(selected: list[Candidate], filenames: dict, budget: int,
                  counter: TokenCounter, reserved_tokens: int = 0) -> BuiltContext:
    """Add whole chunks in order; a chunk that does not fit is skipped, never cut.
    `reserved_tokens` covers instructions, query and generation overhead. The hard
    budget wins over the minimum-evidence target."""
    ctx, room = BuiltContext(), budget - reserved_tokens
    for c in selected:
        label = f"S{len(ctx.items) + 1}"
        block = format_block(label, c, filenames.get(c.document_id))
        n = counter.count(block)
        if n > room:
            ctx.dropped.append(c.chunk_id)
            continue
        room -= n
        ctx.tokens += n
        ctx.items.append(Evidence(label, c, filenames.get(c.document_id), block, n))
    return ctx
