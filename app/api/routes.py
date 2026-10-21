import hashlib
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.api.chat import read_user, upload_user
from app.core.queue import ingestion_queue
from app.db.registry import (
    get_document_for_owner,
    get_pages_for_owner,
    register_upload,
)
from app.ingestion.dispatch import submit_document

router = APIRouter()

UPLOAD_DIR = (
    Path(__file__).resolve().parents[1]
    / "storage"
    / "uploads"
)

MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB


@router.post("/documents", status_code=202)
async def upload_document(file: UploadFile = File(...), user: str = Depends(upload_user)):
    if not file.filename:
        raise HTTPException(400, "Filename is required")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    document_id = str(uuid4())

    # Generate the storage filename ourselves.
    # Do not trust the uploaded filename as a filesystem path.
    safe_filename = Path(file.filename).name
    if not safe_filename or not safe_filename.strip() or safe_filename in (".", ".."):
        raise HTTPException(400, "Filename is required")
    if len(safe_filename) > 100:
        suffix = Path(safe_filename).suffix[:16]
        safe_filename = safe_filename[: 100 - len(suffix)] + suffix
    if Path(safe_filename).suffix.lower() != ".pdf":
        # The worker only converts PDFs; failing here beats queueing a job that is certain to fail.
        raise HTTPException(415, "Only PDF files are supported")
    file_path = UPLOAD_DIR / f"{document_id}_{safe_filename}"

    size = 0
    digest = hashlib.sha256()
    registered = False

    try:
        with file_path.open("wb") as destination:
            while True:
                chunk = await file.read(1024 * 1024)

                if not chunk:
                    break

                if size == 0 and not chunk.lstrip(b"\x00 \t\r\n")[:5] == b"%PDF-":
                    raise HTTPException(415, "File content is not a PDF")
                size += len(chunk)
                digest.update(chunk)

                if size > MAX_FILE_SIZE:
                    raise HTTPException(
                        413,
                        "File exceeds the 100 MB limit",
                    )

                destination.write(chunk)

        existing = register_upload(
            document_id=document_id,
            filename=safe_filename,
            file_path=str(file_path.resolve()),
            content_hash=digest.hexdigest(),
            owner_id=user,
        )

        if existing is not None:
            # Repeated requests also recover a persisted upload whose publication
            # was interrupted before it reached the queue.
            file_path.unlink(missing_ok=True)
            job_id = submit_document(existing["document_id"], ingestion_queue)
            if not job_id:
                raise HTTPException(503, "Ingestion is being recovered. Please retry.")
            registry_document = get_document_for_owner(existing["document_id"], user)
            if registry_document is None:
                raise HTTPException(404, "Document not found")
            return JSONResponse(
                status_code=200,
                content={
                    "document_id": registry_document["document_id"],
                    "job_id": job_id,
                    "filename": registry_document["filename"],
                    "status": registry_document["status"],
                    "duplicate": True,
                    "message": "Identical file already registered; existing ingestion reused",
                },
            )

        registered = True

        job_id = submit_document(document_id, ingestion_queue)
        if not job_id:
            raise RuntimeError("Ingestion publication is already in progress")

        return {
            "document_id": document_id,
            "job_id": job_id,
            "filename": safe_filename,
            "status": "queued",
            "duplicate": False,
            "message": "File accepted for ingestion",
        }

    except HTTPException:
        if not registered:
            file_path.unlink(missing_ok=True)
        raise

    except Exception as exc:
        if not registered:
            file_path.unlink(missing_ok=True)
        raise HTTPException(
            503,
            "Could not queue the file. The persisted upload will be retried.",
        ) from exc

    finally:
        await file.close()

@router.get("/documents/{document_id}")
def document_status(document_id: str, user: str = Depends(read_user)):
    document = get_document_for_owner(document_id, user)

    if document is None:
        raise HTTPException(404, "Document not found")

    document.pop("owner_id", None)
    return document


@router.get("/documents/{document_id}/pages")
def document_pages(document_id: str, user: str = Depends(read_user)):
    if get_document_for_owner(document_id, user) is None:
        raise HTTPException(404, "Document not found")

    return get_pages_for_owner(document_id, user)
