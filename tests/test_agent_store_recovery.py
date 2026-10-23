import tempfile
import time
import unittest
from pathlib import Path

from app.agent.store import AgentStore


class IngestionClaimRecoveryTests(unittest.TestCase):
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
                user_id="alice", url="https://example.org/report",
                normalized_url="https://example.org/report", url_hash="url-hash",
                title="report", content_hash="content-hash", content_text="content",
                reason="test", preview={}, trust={}, ttl_hours=1,
            )
            store.approve_source(source["source_id"], "alice", ttl_hours=1)
            store._write("UPDATE sources SET expires_at=?, status='expired' WHERE source_id=?",
                         (time.time() - 1, source["source_id"]))
            self.assertIsNone(store.valid_approval(source["source_id"], "alice"))


if __name__ == "__main__":
    unittest.main()
