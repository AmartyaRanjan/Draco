"""Expose the starter app's ingestion worker through the generic `ingest_workload`."""
from __future__ import annotations

from typing import Any

from draco.adapters.ingestion import ingest_workload
from draco.core.protocols import Workload


def ingestion_workload(process_document: Any = None) -> Workload:
    """`process_document` defaults to `app.ingestion.worker.process_document` (imported on first use, since
    it pulls in the ML stack). Prefer `draco.adapters.ingestion.ingest_workload` with your own callable."""
    def ingest(document_id: str, file_path: str) -> Any:
        fn = process_document
        if fn is None:
            from app.ingestion.worker import process_document as fn  # heavy import, only when used
        return fn(document_id, file_path)
    return ingest_workload(ingest)
