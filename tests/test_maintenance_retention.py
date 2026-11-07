"""Retention in scripts/agent_maintenance.py: conversation history and agent state, with dry-run."""
import importlib.util
import json
import sqlite3
import tempfile
import time
import unittest
from contextlib import closing, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from app.agent.store import AgentStore
from app.inference.memory import SqliteConversationStore

_spec = importlib.util.spec_from_file_location(
    "agent_maintenance", Path(__file__).resolve().parents[1] / "scripts" / "agent_maintenance.py")
M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M)

DAY = 86400


def count(path, table, where="1=1", args=()):
    with closing(sqlite3.connect(path)) as c:
        return c.execute(f"SELECT count(*) FROM {table} WHERE {where}", args).fetchone()[0]


class ConversationRetentionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "conversations.db"
        store = SqliteConversationStore(self.path)
        for i, conv in enumerate(("old", "old", "active", "new")):
            resp = json.dumps({"metadata": {"request_id": f"r{i}"}})
            store.save_turn("alice", conv, "q", resp, "a", [])
        with closing(sqlite3.connect(self.path)) as c, c:
            for t in ("conversations", "messages", "inference_runs"):
                c.execute(f"UPDATE {t} SET created_at=datetime('now','-200 days') WHERE conversation_id IN ('old','active')")
            # 'active' started long ago but had a turn yesterday: it must survive whole.
            c.execute("UPDATE messages SET created_at=datetime('now','-1 days') WHERE id=(SELECT max(id) FROM messages "
                      "WHERE conversation_id='active')")

    def test_dry_run_counts_and_changes_nothing(self):
        r = M.purge_conversations(self.path, 90, dry_run=True)
        self.assertEqual((r["conversations"], r["messages"], r["inference_runs"]), (1, 4, 2))
        self.assertEqual(count(self.path, "messages"), 8)

    def test_deletes_whole_stale_conversations_only(self):
        r = M.purge_conversations(self.path, 90)
        self.assertEqual((r["conversations"], r["messages"], r["inference_runs"]), (1, 4, 2))
        for t in ("conversations", "messages", "inference_runs"):
            self.assertEqual(count(self.path, t, "conversation_id='old'"), 0)
        self.assertEqual(count(self.path, "messages", "conversation_id='active'"), 2)
        self.assertEqual(count(self.path, "messages", "conversation_id='new'"), 2)

    def test_zero_days_disables_and_missing_db_is_not_created(self):
        self.assertEqual(M.purge_conversations(self.path, 0)["skipped"], "retention disabled")
        missing = Path(self.tmp.name) / "nope.db"
        self.assertEqual(M.purge_conversations(missing, 30)["skipped"], "database not found")
        self.assertFalse(missing.exists())


class AgentStateRetentionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store_path = Path(self.tmp.name) / "agent.db"
        self.ckpt = Path(self.tmp.name) / "ckpt.db"
        store = AgentStore(self.store_path)
        old = time.time() - 60 * DAY
        for eid, status, ts in (("done-old", "completed", old), ("fail-old", "failed", old),
                                ("run-old", "running", old), ("done-new", "completed", time.time())):
            store.create_execution(eid, "alice", "t", {})
            with closing(sqlite3.connect(self.store_path)) as c, c:
                c.execute("UPDATE agent_executions SET status=?, updated_at=? WHERE execution_id=?", (status, ts, eid))
        with closing(sqlite3.connect(self.store_path)) as c, c:
            for sid, status, exp, job in (("s-rej", "rejected", old, False), ("s-pend-new", "pending_approval", time.time(), False),
                                          ("s-ingested", "ingested", old, True)):
                c.execute("INSERT INTO sources (source_id,user_id,url,normalized_url,url_hash,content_hash,content_text,"
                          "preview_json,trust_json,status,created_at,expires_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                          (sid, "alice", "u", "u", "h" + sid, "c", "text", "{}", "{}", status, exp, exp))
                c.execute("INSERT INTO approvals (approval_id,source_id,user_id,url_hash,content_hash,decision,decided_at,"
                          "expires_at) VALUES (?,?,?,?,?,?,?,?)", ("a" + sid, sid, "alice", "h", "c", "x", exp, exp))
                if job:
                    c.execute("INSERT INTO ingestion_jobs (job_id,source_id,idempotency_key,status,created_at,updated_at) "
                              "VALUES (?,?,?,?,?,?)", ("j" + sid, sid, "k" + sid, "done", exp, exp))
        from app.agent.checkpoint import SqliteCheckpointSaver
        SqliteCheckpointSaver(self.ckpt)  # creates the real schema
        with closing(sqlite3.connect(self.ckpt)) as c, c:
            for thread, ts in (("done-old", old), ("run-old", old), ("orphan-old", old), ("done-new", time.time())):
                c.execute("INSERT INTO graph_checkpoints (thread_id,checkpoint_ns,checkpoint_id,created_at,checkpoint_type,"
                          "checkpoint_bytes,checkpoint_crc,metadata_type,metadata_bytes,metadata_crc) "
                          "VALUES (?,?,?,?,?,?,?,?,?,?)", (thread, "", "1", ts, "json", b"{}", 0, "json", b"{}", 0))
                c.execute("INSERT INTO graph_channel_blobs VALUES (?,?,?,?,?,?,?)", (thread, "", "ch", "1", "json", b"{}", 0))

    def test_dry_run(self):
        r = M.purge_agent_state(self.store_path, self.ckpt, 30, dry_run=True)
        self.assertEqual((r["executions"], r["checkpoint_threads"], r["sources"], r["approvals"]), (2, 2, 1, 1))
        self.assertEqual(count(self.store_path, "agent_executions"), 4)
        self.assertEqual(count(self.ckpt, "graph_checkpoints"), 4)

    def test_purges_terminal_state_and_keeps_live_and_recent(self):
        r = M.purge_agent_state(self.store_path, self.ckpt, 30)
        self.assertEqual((r["executions"], r["checkpoint_threads"], r["sources"], r["approvals"]), (2, 2, 1, 1))
        with closing(sqlite3.connect(self.store_path)) as c:
            self.assertEqual({x[0] for x in c.execute("SELECT execution_id FROM agent_executions")}, {"run-old", "done-new"})
            self.assertEqual({x[0] for x in c.execute("SELECT source_id FROM sources")}, {"s-pend-new", "s-ingested"})
        with closing(sqlite3.connect(self.ckpt)) as c:
            self.assertEqual({x[0] for x in c.execute("SELECT thread_id FROM graph_checkpoints")}, {"run-old", "done-new"})
            self.assertEqual({x[0] for x in c.execute("SELECT thread_id FROM graph_channel_blobs")}, {"run-old", "done-new"})


class MainTests(unittest.TestCase):
    def test_cli_dry_run_reports_json_and_flags_override_env(self):
        with tempfile.TemporaryDirectory() as d:
            paths = (str(Path(d) / "conv.db"), str(Path(d) / "agent.db"), str(Path(d) / "ckpt.db"))
            out = StringIO()
            with patch.object(M, "_default_paths", lambda: paths), \
                 patch.object(M, "_artifacts_and_feedback", lambda dry: {"dry": dry}), \
                 patch.dict("os.environ", {"CONVERSATION_RETENTION_DAYS": "7"}), redirect_stdout(out):
                code = M.main(["--dry-run", "--agent-state-days", "0"])
            report = json.loads(out.getvalue())
            self.assertEqual(code, 0)
            self.assertEqual((report["dry_run"], report["conversation_days"], report["agent_state_days"]), (True, 7, 0))
            self.assertEqual(report["artifacts_feedback"], {"dry": True})
            self.assertEqual(report["agent_state"]["skipped"], "retention disabled")
            self.assertFalse(Path(paths[0]).exists())

    def test_failed_step_sets_exit_code_but_others_run(self):
        out = StringIO()
        with patch.object(M, "_default_paths", lambda: ("", "", "")), \
             patch.object(M, "_artifacts_and_feedback", lambda dry: 1 / 0), redirect_stdout(out):
            code = M.main(["--dry-run"])
        report = json.loads(out.getvalue())
        self.assertEqual(code, 1)
        self.assertIn("ZeroDivisionError", report["artifacts_feedback"]["error"])
        self.assertIn("conversations", report)


if __name__ == "__main__":
    unittest.main()
