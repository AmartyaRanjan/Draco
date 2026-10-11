from pathlib import Path

from app.db.registry import save_pages, update_status

def process_document(document_id: str, file_path: str):
    update_status(document_id, "processing")

    try:
        print(f"Processing document: {document_id}")
        print(f"File location: {file_path}")

        if Path(file_path).suffix.lower() == ".pdf":
            # Imported here so the API process, which imports this module to
            # enqueue jobs, does not need to load Docling.
            from app.ingestion.convert import convert_pdf

            conversion = convert_pdf(file_path)
            save_pages(document_id, conversion.page_records())
            # conversion.document is the Docling document for the next stage.

        # TODO: Replace this with our real pipeline:
        # 1. Docling conversion (done above for PDFs; other formats pending)
        # 2. Chunking
        # 3. Embedding generation
        # 4. Qdrant storage

        update_status(document_id, "completed")

        return {
            "document_id": document_id,
            "status": "completed",
        }

    except Exception as exc:
        update_status(
            document_id,
            "failed",
            error=str(exc),
        )
        raise