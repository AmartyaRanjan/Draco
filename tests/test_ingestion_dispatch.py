import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from threading import Event, Thread
from unittest.mock import patch

from app.db import registry
from app.ingestion.dispatch import reconcile_ingestion_outbox, submit_document
from rq.exceptions import DuplicateJobError


class FakeJob:
    def __init__(self, job_id):
        self.id = job_id
        self.status = "queued"

    def get_status(self, refresh=True):
        return self.status

    def delete(self):
        raise AssertionError("active jobs must not be deleted")


class FakeQueue:
    def __init__(self):
        self.jobs = {}
        self.enqueue_calls = 0
        self.accept_then_raise = False
        self.block = False
        self.entered = Event()
        self.release = Event()

    def enqueue(self, *args, job_id, **kwargs):
        self.enqueue_calls += 1
        if job_id in self.jobs:
            raise DuplicateJobError(job_id)
        job = FakeJob(job_id)
        self.jobs[job_id] = job
        self.entered.set()
        if self.accept_then_raise:
            self.accept_then_raise = False
            raise RuntimeError("queue response lost")
        if self.block:
            self.release.wait(timeout=2)
        return job

    def fetch_job(self, job_id):
        return self.jobs.get(job_id)


class DurableDispatchTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        patcher = patch.object(registry, "DB_PATH", Path(temp.name) / "documents.db")
        patcher.start()
        self.addCleanup(patcher.stop)
        registry.init_db()
        self.queue = FakeQueue()
        self.file_path = Path(temp.name) / "upload.pdf"
        self.file_path.write_bytes(b"%PDF-1.7")

    def register(self, document_id="doc-1"):
        self.assertIsNone(registry.register_upload(
            document_id, "upload.pdf", str(self.file_path), f"hash-{document_id}", "alice"
        ))
        return document_id

    def test_persisted_request_is_recovered_after_pre_enqueue_crash(self):
        document_id = self.register()

        self.assertEqual(registry.get_ingestion_outbox(document_id)["state"], "pending")
        self.assertEqual(reconcile_ingestion_outbox(self.queue), 1)
        self.assertEqual(registry.get_ingestion_outbox(document_id)["state"], "queued")
        self.assertEqual(registry.get_document(document_id)["status"], "queued")
        self.assertEqual(self.queue.enqueue_calls, 1)

    def test_queue_acceptance_with_lost_confirmation_is_idempotently_reconciled(self):
        document_id = self.register()
        self.queue.accept_then_raise = True
        with self.assertRaisesRegex(RuntimeError, "response lost"):
            submit_document(document_id, self.queue)

        with closing(registry.get_connection()) as conn, conn:
            conn.execute("UPDATE ingestion_outbox SET next_attempt_at=0")

        self.assertEqual(reconcile_ingestion_outbox(self.queue), 1)
        self.assertEqual(self.queue.enqueue_calls, 2)
        self.assertEqual(len(self.queue.jobs), 1)
        self.assertEqual(registry.get_ingestion_outbox(document_id)["state"], "queued")

    def test_repeated_reconciliation_does_not_duplicate_an_active_job(self):
        document_id = self.register()
        job_id = submit_document(document_id, self.queue)
        self.queue.jobs[job_id].status = "started"

        reconcile_ingestion_outbox(self.queue)
        reconcile_ingestion_outbox(self.queue)

        self.assertEqual(self.queue.enqueue_calls, 1)
        self.assertEqual(registry.get_ingestion_outbox(document_id)["queue_job_id"], job_id)

    def test_only_one_concurrent_publisher_claims_pending_work(self):
        document_id = self.register()
        self.queue.block = True
        results = []

        first = Thread(target=lambda: results.append(submit_document(document_id, self.queue)))
        first.start()
        self.assertTrue(self.queue.entered.wait(timeout=2))
        second = Thread(target=lambda: results.append(submit_document(document_id, self.queue)))
        second.start()
        self.queue.release.set()
        first.join(timeout=2)
        second.join(timeout=2)

        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertEqual(self.queue.enqueue_calls, 1)
        self.assertEqual(len(self.queue.jobs), 1)

    def test_worker_terminal_status_is_preserved(self):
        document_id = self.register()
        job_id = submit_document(document_id, self.queue)
        registry.update_status(document_id, "completed")

        self.assertEqual(submit_document(document_id, self.queue), job_id)
        self.assertEqual(registry.get_ingestion_outbox(document_id)["state"], "completed")
        self.assertEqual(registry.get_document(document_id)["status"], "completed")
        self.assertEqual(self.queue.enqueue_calls, 1)

    def test_lost_job_within_attempt_budget_is_requeued(self):
        document_id = self.register()
        job_id = submit_document(document_id, self.queue)
        with closing(registry.get_connection()) as conn, conn:
            conn.execute(
                "UPDATE ingestion_outbox SET attempts = ? WHERE document_id = ?",
                (registry.MAX_INGESTION_ATTEMPTS - 1, document_id),
            )
        self.queue.jobs[job_id].status = "failed"  # worker died on the job

        self.assertIsNotNone(submit_document(document_id, self.queue))
        self.assertEqual(self.queue.enqueue_calls, 2)
        self.assertEqual(registry.get_ingestion_outbox(document_id)["state"], "queued")
        self.assertEqual(registry.get_document(document_id)["status"], "queued")

    def test_lost_job_beyond_attempt_budget_fails_terminally(self):
        # A job that always kills the worker (OOM on a hostile PDF) must not be
        # requeued forever: after the budget the document fails terminally.
        document_id = self.register()
        job_id = submit_document(document_id, self.queue)
        with closing(registry.get_connection()) as conn, conn:
            conn.execute(
                "UPDATE ingestion_outbox SET attempts = ? WHERE document_id = ?",
                (registry.MAX_INGESTION_ATTEMPTS, document_id),
            )
        self.queue.jobs[job_id].status = "failed"

        self.assertIsNone(submit_document(document_id, self.queue))
        self.assertEqual(self.queue.enqueue_calls, 1)
        self.assertEqual(registry.get_ingestion_outbox(document_id)["state"], "failed")
        self.assertEqual(registry.get_document(document_id)["status"], "failed")
        self.assertIn("attempts", registry.get_document(document_id)["error"])
        # Terminal: reconciliation leaves it alone instead of looping.
        self.assertEqual(reconcile_ingestion_outbox(self.queue), 0)
        self.assertEqual(self.queue.enqueue_calls, 1)


if __name__ == "__main__":
    unittest.main()
