
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class ChunkRecord:
    chunk_id: str
    document_id: str
    document_version: str
    chunk_index: int
    text: str
    page_start: int | None = None
    page_end: int | None = None
    section_path: str | None = None

    def to_payload(self) -> dict[str, Any]:
        """Serialize the chunk for Qdrant and OpenSearch."""
        return asdict(self)
