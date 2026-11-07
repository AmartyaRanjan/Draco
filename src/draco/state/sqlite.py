from __future__ import annotations

import asyncio
import sqlite3
import threading
from collections.abc import Callable, Sequence
from typing import Any, TypeVar

from draco.core.errors import PersistenceError
from draco.state._sql import SqlStore, Tx
from draco.state.stores import PayloadCipher

T = TypeVar("T")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS executions (
  execution_id TEXT PRIMARY KEY, parent_execution_id TEXT, session_id TEXT, agent_id TEXT,
  correlation_id TEXT, target TEXT NOT NULL, status TEXT NOT NULL, started_at REAL NOT NULL,
  finished_at REAL, idempotency_key TEXT, output_json TEXT, error_json TEXT, usage_json TEXT,
  attempts INTEGER NOT NULL DEFAULT 1, input_json TEXT, tenant_id TEXT, user_id TEXT, roles_json TEXT, policy_json TEXT);
CREATE UNIQUE INDEX IF NOT EXISTS ux_exec_key ON executions(idempotency_key) WHERE idempotency_key IS NOT NULL;
CREATE INDEX IF NOT EXISTS ix_exec_parent ON executions(parent_execution_id);
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT, execution_id TEXT NOT NULL, name TEXT NOT NULL,
  ts REAL NOT NULL, data_json TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS ix_events_exec ON events(execution_id, id);
CREATE TABLE IF NOT EXISTS steps (
  execution_id TEXT NOT NULL, step TEXT NOT NULL, status TEXT NOT NULL, data_json TEXT,
  created_at REAL NOT NULL, PRIMARY KEY (execution_id, step));
"""
# added after the first release; created on open for older databases
_LATE_INDEXES = """
CREATE INDEX IF NOT EXISTS ix_exec_status_time ON executions(status, finished_at);
CREATE INDEX IF NOT EXISTS ix_events_ts ON events(ts);
"""


class _SqliteTx:
    def __init__(self, db: sqlite3.Connection):
        self.db = db

    def q(self, sql: str, params: Sequence[Any] = ()) -> list[Any]:
        return self.db.execute(sql, tuple(params)).fetchall()

    def x(self, sql: str, params: Sequence[Any] = ()) -> int:
        return self.db.execute(sql, tuple(params)).rowcount


class SqliteStore(SqlStore):
    """Durable local backend (execution history, events, step checkpoints). One connection guarded
    by a lock, work done off the event loop. Pass ':memory:' for an ephemeral database.
    `cipher` (e.g. `draco.security.crypto.FernetCipher`) encrypts payload columns at rest."""

    def __init__(self, path: str, *, cipher: PayloadCipher | None = None):
        super().__init__(cipher=cipher)
        self.path = path
        self._lock = threading.Lock()
        try:
            self._db = sqlite3.connect(path, check_same_thread=False)
            self._db.row_factory = sqlite3.Row
            if path != ":memory:":
                self._db.execute("PRAGMA journal_mode=WAL")
            if cipher is not None:  # overwrite deleted/replaced content instead of leaving it in free pages
                self._db.execute("PRAGMA secure_delete=ON")
            self._db.executescript(_SCHEMA)
            have = {r["name"] for r in self._db.execute("PRAGMA table_info(executions)")}
            for col in ("tenant_id TEXT", "user_id TEXT", "roles_json TEXT", "policy_json TEXT"):  # databases created before identity existed
                if col.split()[0] not in have:
                    self._db.execute(f"ALTER TABLE executions ADD COLUMN {col}")
            self._db.executescript(_LATE_INDEXES)
            self._db.commit()
        except sqlite3.Error as exc:
            raise PersistenceError(f"cannot open state database: {exc}", retryable=False) from exc

    def close(self) -> None:
        with self._lock:
            self._db.close()

    async def _run(self, fn: Callable[[Tx], T]) -> T:
        def work() -> T:
            with self._lock:
                try:
                    with self._db:  # commit or rollback
                        return fn(_SqliteTx(self._db))
                except sqlite3.Error as exc:
                    raise PersistenceError(f"state store error: {type(exc).__name__}") from exc
        return await asyncio.to_thread(work)

    async def vacuum(self) -> None:
        """Return space freed by `purge()` to the filesystem (rewrites the file; run off-peak)."""
        def work() -> None:
            with self._lock:
                try:
                    self._db.execute("VACUUM")
                    if self.path != ":memory:":
                        self._db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                except sqlite3.Error as exc:
                    raise PersistenceError(f"state store error: {type(exc).__name__}") from exc
        await asyncio.to_thread(work)
