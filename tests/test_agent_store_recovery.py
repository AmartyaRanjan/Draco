import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from app.agent.store import AgentStore


class IngestionClaimRecoveryTests(unittest.TestCase):
    def test_execution_resume_claim_is_atomic_and_owner_scoped(self):
        with tempfile.TemporaryDirectory() as temp:
            store = AgentStore(Path(temp) / "agent.db")
            store.create_execution("exec-1", "alice", "tenant-a", {"message": "test"})

            self.assertIsNone(store.get_execution("exec-1", "bob", "tenant-a"))
            self.assertIsNone(store.get_execution("exec-1", "alice", "tenant-b"))
            self.assertTrue(store.update_execution("exec-1", "alice", "tenant-a", status="waiting"))

            record, claimed = store.claim_execution_resume("exec-1", "alice", "tenant-a")
            self.assertTrue(claimed)
            self.assertEqual(record["status"], "resuming")
            duplicate, claimed = store.claim_execution_resume("exec-1", "alice", "tenant-a")
            self.assertFalse(claimed)
            self.assertEqual(duplicate["status"], "resuming")

    def test_stale_resume_claim_can_be_reclaimed_without_stale_worker_overwriting_result(self):
        with tempfile.TemporaryDirectory() as temp:
            store = AgentStore(Path(temp) / "agent.db")
            store.create_execution("exec-stale", "alice", "tenant-a", {"message": "test"})
            store.update_execution("exec-stale", "alice", "tenant-a", status="waiting")
            first, claimed = store.claim_execution_resume("exec-stale", "alice", "tenant-a")
            self.assertTrue(claimed)
            store._write("UPDATE agent_executions SET updated_at=0 WHERE execution_id=?", ("exec-stale",))

            recovered, claimed = store.claim_execution_resume(
                "exec-stale", "alice", "tenant-a", recover_interrupted=True
            )
            self.assertTrue(claimed)
            self.assertEqual(recovered["status"], "resuming")
            self.assertNotEqual(first["resume_token"], recovered["resume_token"])
            self.assertFalse(
                store.update_execution(
                    "exec-stale",
                    "alice",
                    "tenant-a",
                    status="completed",
                    expected_status="resuming",
                    resume_token=first["resume_token"],
                )
            )
            self.assertTrue(
                store.update_execution(
                    "exec-stale",
                    "alice",
                    "tenant-a",
                    status="completed",
                    expected_status="resuming",
                    resume_token=recovered["resume_token"],
                )
            )

    def test_concurrent_resume_claims_have_exactly_one_winner(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "agent.db"
            store = AgentStore(path)
            store.create_execution("exec-2", "alice", "tenant-a", {"message": "test"})
            store.update_execution("exec-2", "alice", "tenant-a", status="waiting")
            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(
                    pool.map(
                        lambda _: AgentStore(path).claim_execution_resume("exec-2", "alice", "tenant-a")[1],
                        range(2),
                    )
                )
            self.assertEqual(results.count(True), 1)

    def test_expired_execution_cannot_be_resumed(self):
        with tempfile.TemporaryDirectory() as temp:
            store = AgentStore(Path(temp) / "agent.db")
            store.create_execution("exec-expired", "alice", "tenant-a", {"message": "test"})
            store.update_execution("exec-expired", "alice", "tenant-a", status="waiting")
            store._write("UPDATE agent_executions SET expires_at=0 WHERE execution_id=?", ("exec-expired",))
            record, claimed = store.claim_execution_resume("exec-expired", "alice", "tenant-a")
            self.assertFalse(claimed)
            self.assertEqual(record["status"], "expired")

    def test_live_claim_is_exclusive_and_expired_claim_is_reused(self):
        with tempfile.TemporaryDirectory() as temp:
            store = AgentStore(Path(temp) / "agent.db")
            source = store.create_source(
                user_id="alice",
                url="https://example.org/report",
                normalized_url="https://example.org/report",
                url_hash="url-hash",
                title="report",
                content_hash="content-hash",
                content_text="approved content",
                reason="test",
                preview={},
                trust={},
                ttl_hours=1,
            )

            job, created = store.claim_job(source["source_id"], "idempotency-key")
            active_job, reclaimed = store.claim_job(source["source_id"], "idempotency-key")
            self.assertTrue(created)
            self.assertFalse(reclaimed)
            self.assertEqual(active_job["job_id"], job["job_id"])

            store._write("UPDATE ingestion_jobs SET lease_until=0 WHERE job_id=?", (job["job_id"],))
            recovered_job, recovered = store.claim_job(source["source_id"], "idempotency-key")

            self.assertTrue(recovered)
            self.assertEqual(recovered_job["job_id"], job["job_id"])
            self.assertFalse(store.claim_job(source["source_id"], "idempotency-key")[1])

    def test_approval_expires_with_its_source(self):
        with tempfile.TemporaryDirectory() as temp:
            store = AgentStore(Path(temp) / "agent.db")
            source = store.create_source(
                user_id="alice",
                url="https://example.org/report",
                normalized_url="https://example.org/report",
                url_hash="url-hash",
                title="report",
                content_hash="content-hash",
                content_text="content",
                reason="test",
                preview={},
                trust={},
                ttl_hours=1,
            )
            store.approve_source(source["source_id"], "alice", ttl_hours=1)
            store._write(
                "UPDATE sources SET expires_at=?, status='expired' WHERE source_id=?",
                (time.time() - 1, source["source_id"]),
            )
            self.assertIsNone(store.valid_approval(source["source_id"], "alice"))


if __name__ == "__main__":
    unittest.main()
