from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import time
from collections.abc import Callable
from typing import Any, TypeVar

from draco.core.errors import PersistenceError
from draco.state.stores import ExecutionRecord, StepRecord, dumps

T = TypeVar("T")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS executions (
  execution_id TEXT PRIMARY KEY, parent_execution_id TEXT, session_id TEXT, agent_id TEXT,
  correlation_id TEXT, target TEXT NOT NULL, status TEXT NOT NULL, started_at REAL NOT NULL,
  finished_at REAL, idempotency_key TEXT, output_json TEXT, error_json TEXT, usage_json TEXT,
  attempts INTEGER NOT NULL DEFAULT 1, input_json TEXT);
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


class SqliteStore:
    """Durable local backend (execution history, events, step checkpoints). One connection guarded
    by a lock, work done off the event loop. Pass ':memory:' for an ephemeral database."""

    def __init__(self, path: str):
        self._lock = threading.Lock()
        try:
            self._db = sqlite3.connect(path, check_same_thread=False)
            self._db.row_factory = sqlite3.Row
            if path != ":memory:":
                self._db.execute("PRAGMA journal_mode=WAL")
            self._db.executescript(_SCHEMA)
        except sqlite3.Error as exc:
            raise PersistenceError(f"cannot open state database: {exc}", retryable=False) from exc

    def close(self) -> None:
        with self._lock:
            self._db.close()

    async def _run(self, fn: Callable[[sqlite3.Connection], T]) -> T:
        def work() -> T:
            with self._lock:
                try:
                    with self._db:  # commit or rollback
                        return fn(self._db)
                except sqlite3.Error as exc:
                    raise PersistenceError(f"state store error: {type(exc).__name__}") from exc
        return await asyncio.to_thread(work)

    @staticmethod
    def _rec(r: sqlite3.Row) -> ExecutionRecord:
        return ExecutionRecord(
            execution_id=r["execution_id"], target=r["target"], status=r["status"],
            started_at=r["started_at"], parent_execution_id=r["parent_execution_id"],
            session_id=r["session_id"], agent_id=r["agent_id"], correlation_id=r["correlation_id"],
            finished_at=r["finished_at"], idempotency_key=r["idempotency_key"],
            output=json.loads(r["output_json"]) if r["output_json"] else None,
            error=json.loads(r["error_json"]) if r["error_json"] else None,
            usage=json.loads(r["usage_json"]) if r["usage_json"] else {}, attempts=r["attempts"],
            input=json.loads(r["input_json"]) if r["input_json"] else None)

    async def save_execution(self, rec: ExecutionRecord) -> None:
        def f(db: sqlite3.Connection) -> None:
            db.execute(
                "INSERT INTO executions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(execution_id) DO UPDATE SET status=excluded.status, finished_at=excluded.finished_at,"
                " idempotency_key=excluded.idempotency_key, output_json=excluded.output_json, error_json=excluded.error_json, usage_json=excluded.usage_json,"
                " attempts=excluded.attempts",
                (rec.execution_id, rec.parent_execution_id, rec.session_id, rec.agent_id, rec.correlation_id,
                 rec.target, rec.status, rec.started_at, rec.finished_at, rec.idempotency_key,
                 None if rec.output is None else dumps(rec.output),
                 None if rec.error is None else dumps(rec.error), dumps(rec.usage), rec.attempts,
                 None if rec.input is None else dumps(rec.input)))
        await self._run(f)

    async def get_execution(self, execution_id: str) -> ExecutionRecord | None:
        def f(db: sqlite3.Connection) -> ExecutionRecord | None:
            r = db.execute("SELECT * FROM executions WHERE execution_id=?", (execution_id,)).fetchone()
            return self._rec(r) if r else None
        return await self._run(f)

    async def find_by_key(self, key: str) -> ExecutionRecord | None:
        def f(db: sqlite3.Connection) -> ExecutionRecord | None:
            r = db.execute("SELECT * FROM executions WHERE idempotency_key=?", (key,)).fetchone()
            return self._rec(r) if r else None
        return await self._run(f)

    async def list_children(self, execution_id: str) -> list[ExecutionRecord]:
        def f(db: sqlite3.Connection) -> list[ExecutionRecord]:
            return [self._rec(r) for r in db.execute(
                "SELECT * FROM executions WHERE parent_execution_id=? ORDER BY started_at", (execution_id,))]
        return await self._run(f)

    async def append_event(self, execution_id: str, name: str, ts: float, data: dict[str, Any]) -> None:
        await self._run(lambda db: db.execute(
            "INSERT INTO events(execution_id,name,ts,data_json) VALUES (?,?,?,?)",
            (execution_id, name, ts, dumps(data))))

    async def list_events(self, execution_id: str) -> list[dict[str, Any]]:
        def f(db: sqlite3.Connection) -> list[dict[str, Any]]:
            return [{"name": r["name"], "ts": r["ts"], "data": json.loads(r["data_json"])} for r in db.execute(
                "SELECT * FROM events WHERE execution_id=? ORDER BY id", (execution_id,))]
        return await self._run(f)

    async def begin_step(self, execution_id: str, step: str) -> None:
        await self._run(lambda db: db.execute(
            "INSERT INTO steps VALUES (?,?,?,?,?) ON CONFLICT(execution_id,step) DO UPDATE SET status='started',"
            " data_json=NULL, created_at=excluded.created_at", (execution_id, step, "started", None, time.time())))

    async def complete_step(self, execution_id: str, step: str, data: Any) -> None:
        await self._run(lambda db: db.execute(
            "INSERT INTO steps VALUES (?,?,?,?,?) ON CONFLICT(execution_id,step) DO UPDATE SET status='done',"
            " data_json=excluded.data_json, created_at=excluded.created_at",
            (execution_id, step, "done", dumps(data), time.time())))

    @staticmethod
    def _step(r: sqlite3.Row) -> StepRecord:
        return StepRecord(r["execution_id"], r["step"], r["status"],
                          json.loads(r["data_json"]) if r["data_json"] else None, r["created_at"])

    async def get_step(self, execution_id: str, step: str) -> StepRecord | None:
        def f(db: sqlite3.Connection) -> StepRecord | None:
            r = db.execute("SELECT * FROM steps WHERE execution_id=? AND step=?", (execution_id, step)).fetchone()
            return self._step(r) if r else None
        return await self._run(f)

    async def list_steps(self, execution_id: str) -> list[StepRecord]:
        def f(db: sqlite3.Connection) -> list[StepRecord]:
            return [self._step(r) for r in db.execute(
                "SELECT * FROM steps WHERE execution_id=? ORDER BY created_at", (execution_id,))]
        return await self._run(f)
