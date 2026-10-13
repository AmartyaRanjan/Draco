"""Per-request evidence registry: the only source of valid citations."""
from dataclasses import dataclass, field

from app.inference.schemas import Citation
from app.retrieval.context import TokenCounter, format_block
from app.retrieval.schema import Candidate


@dataclass
class EvidenceRegistry:
    counter: TokenCounter
    items: dict[str, tuple[Candidate, str | None]] = field(default_factory=dict)  # label -> (cand, filename)
    _ids: dict[str, str] = field(default_factory=dict)  # chunk_id -> label

    def add_many(self, candidates: list[Candidate], filenames: dict, budget: int) -> tuple[str, list[str]]:
        """Add whole chunks in order within `budget` tokens (never truncating). Returns rendered text
        and the labels of the chunks shown. Re-found chunks keep their original label."""
        blocks, labels = [], []
        for c in candidates:
            label = self._ids.get(c.chunk_id)
            new = label is None
            if new:
                label = f"S{len(self.items) + 1}"
            block = format_block(label, c, filenames.get(c.document_id))
            n = self.counter.count(block)
            if n > budget:
                continue
            budget -= n
            if new:
                self.items[label] = (c, filenames.get(c.document_id))
                self._ids[c.chunk_id] = label
            blocks.append(block)
            labels.append(label)
        return "\n".join(blocks), labels

    def citation(self, label: str) -> Citation | None:
        hit = self.items.get(label)
        if hit is None:
            return None
        c, fn = hit
        return Citation(label=label, document_id=c.document_id, chunk_id=c.chunk_id, filename=fn,
                        page_start=c.page_start, page_end=c.page_end, section_path=c.section_path)
