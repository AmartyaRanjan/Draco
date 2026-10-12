"""Regression tests: registry helpers must deterministically close SQLite handles.

sqlite3.Connection as a context manager only commits/rolls back; it never
closes. Relying on cyclic GC to release the file handle holds an exclusive
lock on Windows (PermissionError on temp-dir cleanup) and leaks descriptors
in long-lived API/worker processes.
"""
import unittest
from unittest.mock import MagicMock, patch

from app.db import registry


def _connection(fetchone=None, fetchall=None):
    conn = MagicMock(name="sqlite-connection")
    # `with conn:` protocol used for commit/rollback.
    conn.__enter__.return_value = conn
    conn.__exit__.return_value = False
    # PRAGMA queries iterate rows like {"name": ...}; empty is fine.
    conn.execute.return_value.__iter__.return_value = iter([])
    if fetchone is not None or True:
        conn.execute.return_value.fetchone.return_value = fetchone
    if fetchall is not None or True:
        conn.execute.return_value.fetchall.return_value = fetchall if fetchall is not None else []
    return conn


class ConnectionCloseTests(unittest.TestCase):
    def assertCloses(self, func, *args, **kwargs):
        conn = _connection()
        with patch.object(registry, "get_connection", return_value=conn):
            func(*args, **kwargs)
        conn.close.assert_called_once()

    def test_init_db_closes(self):
        self.assertCloses(registry.init_db)

    def test_create_document_closes(self):
        self.assertCloses(registry.create_document, "d", "f.pdf", "/tmp/f.pdf")

    def test_register_upload_closes(self):
        conn = _connection(fetchone=None)
        with patch.object(registry, "get_connection", return_value=conn):
            registry.register_upload("d", "f.pdf", "/tmp/f.pdf", "h")
        conn.close.assert_called_once()

    def test_save_pages_closes(self):
        self.assertCloses(registry.save_pages, "d", [])

    def test_get_pages_closes(self):
        self.assertCloses(registry.get_pages, "d")

    def test_update_status_closes(self):
        self.assertCloses(registry.update_status, "d", "queued")

    def test_get_document_closes(self):
        conn = _connection(fetchone=None)
        with patch.object(registry, "get_connection", return_value=conn):
            self.assertIsNone(registry.get_document("missing"))
        conn.close.assert_called_once()

    def test_set_job_id_closes(self):
        self.assertCloses(registry.set_job_id, "d", "job-1")


if __name__ == "__main__":
    unittest.main()
