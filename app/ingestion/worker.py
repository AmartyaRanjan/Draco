import logging
from datetime import UTC, datetime
from pathlib import Path

from app.db.registry import (
    get_document,
    save_pages,
    update_status,
)
from app.ingestion.formats import (
    check_docx,
    check_html,
    detect_format,
    read_text_file,
)
from app.ingestion.headers import title_from_filename

logger = logging.getLogger(__name__)


def _convert_and_index(document_id: str, path: Path, fmt: str, document_version: str,
                       title: str | None) -> dict:
    """Format-specific conversion, then chunk + index. Heavy imports stay lazy so the API process
    (and the md/txt path) never loads Docling."""
    if fmt == "pdf":
        from app.ingestion.convert import convert_pdf

        # convert_pdf runs the structural PDF preflight (bytes/pages/images) first.
        conversion = convert_pdf(str(path))

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

        from app.ingestion.pipeline import ingest_converted_document

        return ingest_converted_document(
            document_id=document_id,
            document=conversion.document,
            document_version=document_version,
            title=title,
        )

    if fmt in ("docx", "html"):
        # Limits first: a rejected file is never handed to the parser.
        (check_docx if fmt == "docx" else check_html)(path)
        from app.ingestion.convert import convert_office

        document = convert_office(str(path))
        from app.ingestion.pipeline import ingest_converted_document

        return ingest_converted_document(
            document_id=document_id,
            document=document,
            document_version=document_version,
            title=title,
        )

    # markdown / text: lightweight path, no Docling.
    text = read_text_file(path)
    from app.ingestion.pipeline import ingest_text_document

    return ingest_text_document(
        document_id=document_id,
        text=text,
        fmt=fmt,
        document_version=document_version,
        title=title,
    )


def _title(document_record: dict, path: Path) -> str | None:
    return title_from_filename(document_record.get("filename") or path.name)


def process_document(document_id: str, file_path: str):
    update_status(document_id, "processing", error=None)

    try:
        path = Path(file_path)

        if not path.is_file():
            raise FileNotFoundError(
                f"Uploaded file does not exist: {file_path}"
            )

        # Raises NotImplementedError for an unsupported extension.
        fmt = detect_format(path)

        logger.info("Processing document %s (%s)", document_id, fmt)

        # Reuse the registered content hash as a version identifier
        # when available, making retries deterministic.
        document_record = get_document(document_id) or {}
        document_version = (
            document_record.get("content_hash") or "1"
        )

        result = _convert_and_index(
            document_id, path, fmt, document_version, _title(document_record, path)
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


def reindex_document(document_id: str) -> dict:
    """Re-ingest one completed document in place (e.g. to add contextual chunk headers).

    The document stays `completed` and searchable throughout: chunks are written under a fresh
    version and the previous version's chunks are deleted only after every batch succeeded
    (the pipeline's stale-version cleanup). A failure leaves the old chunks and the status as they
    were and is raised to the caller. The registry row is not modified."""
    document_record = get_document(document_id)
    if document_record is None:
        raise KeyError(f"document {document_id!r} not found")
    if document_record.get("status") != "completed":
        raise ValueError(f"document {document_id!r} is {document_record.get('status')!r}, not completed")
    path = Path(document_record["file_path"])
    if not path.is_file():
        raise FileNotFoundError(f"Uploaded file does not exist: {path}")
    fmt = detect_format(path)
    base = document_record.get("content_hash") or "1"
    version = f"{base}:reindex-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%f')}"
    result = _convert_and_index(document_id, path, fmt, version, _title(document_record, path))
    return {"document_id": document_id, "status": "completed", "format": fmt,
            "chunk_count": result["chunk_count"], "document_version": version}
