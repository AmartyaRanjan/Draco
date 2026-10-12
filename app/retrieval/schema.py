from dataclasses import dataclass, field, asdict


@dataclass(slots=True)
class Candidate:
    chunk_id: str
    document_id: str
    text: str
    document_version: str | None = None
    chunk_index: int | None = None
    page_start: int | None = None
    page_end: int | None = None
    section_path: str | None = None
    dense_rank: int | None = None
    dense_score: float | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None
    rrf_score: float = 0.0
    rrf_rank: int | None = None
    colbert_score: float | None = None  # raw MaxSim score, not a probability
    calibrated_probability: float | None = None
    selection: str | None = None  # passed_threshold | fallback_below_threshold
    conflict: bool = False

    def diag(self) -> dict:
        d = asdict(self)
        d.pop("text")
        return d


class RetrievalError(Exception):
    """Controlled failure; `stage` names where it happened."""

    def __init__(self, message, stage, status_code=503):
        super().__init__(message)
        self.stage = stage
        self.status_code = status_code
