"""Run-scoped evidence store. The only source of citations; private chunks and web passages
have distinct identities and a web passage is never presented as a database chunk."""
from typing import Literal

from pydantic import BaseModel, model_validator


class EvidenceItem(BaseModel):
    ref: str  # citation marker: S# (private chunk) or W# (web passage)
    kind: Literal["private", "web"]
    text: str
    # private identity (must be real index identifiers)
    document_id: str | None = None
    chunk_id: str | None = None
    filename: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    section_path: str | None = None
    # web identity
    url: str | None = None
    title: str | None = None
    passage_id: str | None = None  # stable id within the fetched page, e.g. "p3"
    fetched_at: float | None = None
    # ranking: raw retrieval scores, never percentages
    colbert_score: float | None = None
    rrf_score: float | None = None
    selection: str | None = None
    status: Literal["raw", "summarized", "omitted"] = "raw"  # omitted: dropped from the prompt for budget; still in the store

    @model_validator(mode="after")
    def _identity(self):
        if self.kind == "private" and not (self.document_id and self.chunk_id):
            raise ValueError("private evidence needs document_id and chunk_id")
        if self.kind == "web" and not (self.url and self.passage_id):
            raise ValueError("web evidence needs url and passage_id")
        if self.kind == "web" and (self.document_id or self.chunk_id):
            raise ValueError("web evidence must not carry document/chunk ids")
        return self

    def source_key(self) -> str:
        return f"{self.document_id}:{self.chunk_id}" if self.kind == "private" else f"{self.url}#{self.passage_id}"

    def locator(self) -> dict:
        """Provenance fields exactly as stored (what a citation resolves to)."""
        if self.kind == "private":
            return {"ref": self.ref, "kind": "private", "document_id": self.document_id,
                    "chunk_id": self.chunk_id, "filename": self.filename,
                    "page_start": self.page_start, "page_end": self.page_end}
        return {"ref": self.ref, "kind": "web", "url": self.url, "passage_id": self.passage_id,
                "title": self.title}

    def render(self, compact: bool = False) -> str:
        if compact:  # step 1 of compaction: drop metadata formatting
            return f"[{self.ref}] {self.text}"
        if self.kind == "private":
            pg = "" if self.page_start is None else f" | pages: {self.page_start}" + (
                f"-{self.page_end}" if self.page_end not in (None, self.page_start) else "")
            return (f"[{self.ref}] source: {self.filename or 'unknown'} | document_id: {self.document_id}"
                    f"{pg} | section: {self.section_path or 'n/a'}\n{self.text}\n")
        return f"[{self.ref}] WEB source: {self.url} | title: {self.title or 'n/a'} (untrusted)\n{self.text}\n"


class EvidenceStore:
    """Holds every evidence item ever gathered in a run; summaries never delete originals."""

    def __init__(self):
        self.items: dict[str, EvidenceItem] = {}
        self._by_key: dict[str, str] = {}
        self._n = {"S": 0, "W": 0}

    def add(self, item_kwargs: dict) -> EvidenceItem:
        """Add (or return the existing item for the same source key). `ref` is assigned here."""
        kind = item_kwargs["kind"]
        probe = EvidenceItem(ref="tmp", **item_kwargs)
        existing = self._by_key.get(probe.source_key())
        if existing:
            return self.items[existing]
        prefix = "S" if kind == "private" else "W"
        self._n[prefix] += 1
        probe.ref = f"{prefix}{self._n[prefix]}"
        self.items[probe.ref] = probe
        self._by_key[probe.source_key()] = probe.ref
        return probe

    def get(self, ref: str) -> EvidenceItem | None:
        return self.items.get(ref)

    def raw(self) -> list[EvidenceItem]:
        return [i for i in self.items.values() if i.status == "raw"]

    def mark_summarized(self, refs):
        for r in refs:
            if r in self.items:
                self.items[r].status = "summarized"

    def mark_omitted(self, refs):
        for r in refs:
            if r in self.items and self.items[r].status == "raw":
                self.items[r].status = "omitted"

    def refs(self) -> list[str]:
        return list(self.items)
