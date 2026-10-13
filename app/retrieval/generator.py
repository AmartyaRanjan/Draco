"""Prompt helpers and citation extraction for retrieval-only /query (answer generation: app.inference)."""
import re
from dataclasses import dataclass, field

from app.retrieval.context import BuiltContext, pages

SYSTEM = (
    "You answer strictly from the numbered evidence blocks provided. "
    "Cite every claim with the block label in square brackets, e.g. [S1]. "
    "Only use labels that exist. If the evidence does not contain the answer, say "
    "that the provided documents are insufficient instead of guessing. "
    "Never invent sources, page numbers or facts. Answer in the user's language."
)
WEAK_NOTE = ("Note: no evidence block passed the relevance threshold; the blocks are "
             "low-confidence fallbacks. Say so if they do not answer the question.")


@dataclass
class GenerationResult:
    answer: str
    citations: list[dict]
    invalid_markers: list[str] = field(default_factory=list)
    timings: dict = field(default_factory=dict)
    usage: dict = field(default_factory=dict)
    model: str | None = None


def build_messages(query: str, ctx: BuiltContext, weak: bool) -> list[dict]:
    evidence = "\n".join(e.text_block for e in ctx.items)
    user = f"Evidence:\n{evidence}\n{WEAK_NOTE if weak else ''}\nQuestion: {query}"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def extract_citations(answer: str, ctx: BuiltContext):
    by_label = {e.label: e for e in ctx.items}
    seen, cites, invalid = set(), [], []
    for label in re.findall(r"\[(S\d+)\]", answer):
        if label in seen:
            continue
        seen.add(label)
        e = by_label.get(label)
        if e is None:
            invalid.append(label)
            continue
        c = e.candidate
        cites.append({"label": label, "chunk_id": c.chunk_id, "document_id": c.document_id,
                      "filename": e.filename, "page_start": c.page_start,
                      "page_end": c.page_end, "pages": pages(c), "section_path": c.section_path})
    return cites, invalid
