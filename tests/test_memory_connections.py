"""Regression tests: SqliteConversationStore must close SQLite handles.

sqlite3.Connection as a context manager only commits/rolls back and never
closes. Without an explicit close the file handle survives until cyclic GC,
holding a lock on Windows and leaking descriptors in long-lived API processes
(same root cause as PR #3, which covered the ingestion registry only).
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.inference.memory import SqliteConversationStore


def _connection(fetchone=None, fetchall=None):
    conn = MagicMock(name="sqlite-connection")
    conn.__enter__.return_value = conn
    conn.__exit__.return_value = False
    conn.execute.return_value.fetchone.return_value = fetchone
    conn.execute.return_value.fetchall.return_value = fetchall if fetchall is not None else []
    return conn


class ConversationStoreCloseTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store = SqliteConversationStore(Path(tmp.name) / "c.db")

    def test_owner_closes(self):
        conn = _connection(fetchone=None)
        with patch.object(self.store, "_conn", return_value=conn):
            self.assertIsNone(self.store.owner("missing"))
        conn.close.assert_called_once()

    def test_load_missing_conversation_closes(self):
        conn = _connection(fetchone=None)
        with patch.object(self.store, "_conn", return_value=conn):
            ctx = self.store.load("alice", "missing", 20)
        conn.close.assert_called_once()
        self.assertEqual((ctx.summary, ctx.recent), (None, []))

    def test_load_closes(self):
        conv = {"user_id": "alice", "summary": "s"}
        rows = [{"role": "user", "content": "hi"}]
        conn = _connection(fetchone=conv, fetchall=rows)
        with patch.object(self.store, "_conn", return_value=conn):
            ctx = self.store.load("alice", "c1", 20)
        conn.close.assert_called_once()
        self.assertEqual(ctx.recent, rows)

    def test_save_turn_closes(self):
        conn = _connection()
        response_json = json.dumps({"metadata": {"request_id": "r1"}})
        with patch.object(self.store, "_conn", return_value=conn):
            self.store.save_turn("alice", "c1", "hi", response_json, "hello", {})
        conn.close.assert_called_once()

    def test_real_store_round_trip(self):
        # End-to-end on a real file: writes must commit and reads must see them.
        response_json = json.dumps({"metadata": {"request_id": "r1"}})
        self.store.save_turn("alice", "c1", "hi", response_json, "hello", {"k": "v"})
        self.assertEqual(self.store.owner("c1"), "alice")
        ctx = self.store.load("alice", "c1", 20)
        self.assertEqual([m["content"] for m in ctx.recent], ["hi", "hello"])
        with self.assertRaises(PermissionError):
            self.store.load("bob", "c1", 20)


if __name__ == "__main__":
    unittest.main()
