
import logging
from pathlib import Path

from app.db.registry import (
    get_document,
    save_pages,
    update_status,
)

logger = logging.getLogger(__name__)


def process_document(document_id: str, file_path: str):
    update_status(document_id, "processing", error=None)

    try:
        path = Path(file_path)

        if not path.is_file():
            raise FileNotFoundError(
                f"Uploaded file does not exist: {file_path}"
            )

        if path.suffix.lower() != ".pdf":
            raise NotImplementedError(
                f"Unsupported file format: {path.suffix}. "
                "Only PDF ingestion is implemented currently."
            )

        print(f"Processing document: {document_id}")
        print(f"File location: {file_path}")

        # Keep heavy conversion imports out of the API process.
        from app.ingestion.convert import convert_pdf

        conversion = convert_pdf(file_path)

        # Preserve the existing page-level audit records.
        page_records = conversion.page_records()
        save_pages(document_id, page_records)

        low_yield = [p["page_no"] for p in page_records if p.get("low_yield")]
        if low_yield:
            logger.warning(
                "Document %s: %d page(s) yielded little text (likely images "
                "or handwriting): %s",
                document_id,
                len(low_yield),
                low_yield,
            )

        # Reuse the registered content hash as a version identifier
        # when available, making retries deterministic.
        document_record = get_document(document_id) or {}
        document_version = (
            document_record.get("content_hash") or "1"
        )

        from app.ingestion.pipeline import (
            ingest_converted_document,
        )

        result = ingest_converted_document(
            document_id=document_id,
            document=conversion.document,
            document_version=document_version,
        )

        update_status(
            document_id,
            "completed",
            error=None,
        )

        return {
            "document_id": document_id,
            "status": "completed",
            "chunk_count": result["chunk_count"],
        }

    except Exception as exc:
        update_status(
            document_id,
            "failed",
            error=str(exc),
        )
        raise
