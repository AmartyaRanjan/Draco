"""Regression tests: agent list_documents tool must close SQLite handles.

registry.get_connection() returns a sqlite3.Connection whose context manager
only commits/rolls back and never closes. Without an explicit close the file
handle survives until cyclic GC, holding a lock on Windows and leaking
descriptors in long-lived worker processes (same root cause as PR #3).
"""
import asyncio
import unittest
from unittest.mock import MagicMock, patch

from app.inference import tools as tools_mod


def _connection():
    conn = MagicMock(name="sqlite-connection")
    conn.__enter__.return_value = conn
    conn.__exit__.return_value = False
    cursor = MagicMock(name="cursor")
    cursor.__iter__.return_value = iter([])
    conn.execute.return_value = cursor
    return conn


class ListToolConnectionTests(unittest.TestCase):
    def test_list_documents_closes_connection(self):
        conn = _connection()
        with patch.object(tools_mod.registry, "get_connection", return_value=conn):
            text, summary = asyncio.run(
                tools_mod._list(
                    tools_mod.ListArgs(),
                    tools_mod.ToolContext(user_id="u", search=None),
                )
            )
        conn.close.assert_called_once()
        self.assertEqual((text, summary), ("No documents.", "Listed 0 documents"))

    def test_list_documents_returns_rows_before_close(self):
        conn = _connection()
        rows = [
            {"document_id": "d1", "filename": "a.pdf", "status": "completed"},
            {"document_id": "d2", "filename": "b.pdf", "status": "queued"},
        ]

        # Use real dicts via cursor iteration of Row-like dicts.
        def fake_execute(*a, **k):
            cur = MagicMock()
            cur.__iter__.return_value = iter(rows)
            return cur

        conn.execute.side_effect = fake_execute
        with patch.object(tools_mod.registry, "get_connection", return_value=conn):
            text, summary = asyncio.run(
                tools_mod._list(
                    tools_mod.ListArgs(),
                    tools_mod.ToolContext(user_id="u", search=None),
                )
            )
        conn.close.assert_called_once()
        self.assertIn("d1 | a.pdf | completed", text)
        self.assertIn("d2 | b.pdf | queued", text)
        self.assertEqual(summary, "Listed 2 documents")


if __name__ == "__main__":
    unittest.main()
