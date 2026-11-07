from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class ChunkRecord:
    chunk_id: str
    document_id: str
    document_version: str
    chunk_index: int
    text: str  # the displayed / cited text; never includes the context header
    page_start: int | None = None
    page_end: int | None = None
    section_path: str | None = None
    # Document/section context prepended for embedding and BM25 only (see app.ingestion.headers).
    context_header: str | None = None

    def index_text(self) -> str:
        """Text used for the dense embedding and the BM25 `search_text` field."""
        if not self.context_header:
            return self.text
        return f"{self.context_header}\n\n{self.text}"

    def to_payload(self) -> dict[str, Any]:
        """Serialize the chunk for Qdrant and OpenSearch."""
        return asdict(self)
