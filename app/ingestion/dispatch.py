"""Durable publication of persisted documents to the RQ ingestion queue."""
import logging

from rq.exceptions import DuplicateJobError

from app.db import registry
from app.ingestion.worker import process_document

logger = logging.getLogger("rag.ingestion")
_ACTIVE_QUEUE_STATES = {"queued", "started", "deferred", "scheduled"}


def _queue_status(job):
    if job is None:
        return None
    status = job.get_status(refresh=True)
    return str(getattr(status, "value", status)).lower()


def _existing_job(queue, job_id):
    if not job_id:
        return None, None
    job = queue.fetch_job(job_id)
    return job, _queue_status(job)


def submit_document(document_id, queue):
    """Publish a registered document at most once per durable outbox attempt."""
    registry.ensure_ingestion_outbox(document_id)
    document = registry.get_document(document_id)
    if document is None:
        raise KeyError(f"document {document_id!r} not found")
    outbox = registry.get_ingestion_outbox(document_id)
    if outbox["state"] in ("completed", "failed"):
        return outbox["queue_job_id"] or document["job_id"]

    if outbox["state"] in ("queued", "running"):
        job, status = _existing_job(queue, outbox["queue_job_id"])
        if status in _ACTIVE_QUEUE_STATES:
            return outbox["queue_job_id"]
        if status == "finished":
            document = registry.get_document(document_id)
        if status == "finished" and document["status"] == "completed":
            return outbox["queue_job_id"]
        registry.requeue_missing_ingestion(document_id)

    claim = registry.claim_ingestion_outbox(document_id)
    if claim is None:
        outbox = registry.get_ingestion_outbox(document_id)
        if outbox and outbox["state"] == "publishing":
            job, status = _existing_job(queue, outbox["queue_job_id"])
            if status in _ACTIVE_QUEUE_STATES:
                registry.mark_ingestion_enqueued(document_id, outbox["queue_job_id"])
                return outbox["queue_job_id"]
        return None

    try:
        job = queue.enqueue(
            process_document,
            document_id,
            document["file_path"],
            job_timeout=3600,
            result_ttl=86400,
            job_id=claim["queue_job_id"],
        )
    except DuplicateJobError:
        try:
            existing, status = _existing_job(queue, claim["queue_job_id"])
            latest = registry.get_document(document_id)
            if status in _ACTIVE_QUEUE_STATES:
                job = existing
            elif status == "finished" and latest and latest["status"] == "completed":
                registry.update_status(document_id, "completed")
                return claim["queue_job_id"]
            elif existing is not None:
                existing.delete()
                registry.requeue_missing_ingestion(document_id)
                return submit_document(document_id, queue)
            else:
                raise RuntimeError("Duplicate queue job could not be fetched")
        except Exception as exc:
            registry.release_ingestion_outbox(document_id, type(exc).__name__)
            raise
    except Exception as exc:
        registry.release_ingestion_outbox(document_id, type(exc).__name__)
        raise

    if job is None:
        registry.release_ingestion_outbox(document_id, "Queue returned no job")
        raise RuntimeError("Queue returned no job for persisted ingestion request")
    registry.mark_ingestion_enqueued(document_id, job.id)
    return job.id


def reconcile_ingestion_outbox(queue):
    """Recover pending publications and queued jobs lost from the external queue."""
    recovered = 0
    for outbox in registry.list_ingestion_outbox():
        try:
            job_id = submit_document(outbox["document_id"], queue)
            if job_id:
                recovered += 1
        except Exception as exc:
            logger.exception(
                "ingestion_outbox_reconciliation_failed",
                extra={"document_id": outbox["document_id"], "error": type(exc).__name__},
            )
    return recovered
