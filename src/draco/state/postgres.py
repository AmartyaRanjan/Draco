"""Postgres backend for execution history, events and step checkpoints (`pip install draco[postgres]`).

Same protocols, schema shape, retention and encryption behaviour as `SqliteStore`; suitable for several
harness processes sharing one database. psycopg 3 is imported only when a store is constructed."""
from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable, Sequence
from typing import Any, TypeVar

from draco.core.errors import ConfigurationError, PersistenceError
from draco.state._sql import SqlStore, Tx
from draco.state.stores import PayloadCipher

T = TypeVar("T")

_SCHEMA = (
    "CREATE TABLE IF NOT EXISTS {executions} ("
    " execution_id TEXT PRIMARY KEY, parent_execution_id TEXT, session_id TEXT, agent_id TEXT,"
    " correlation_id TEXT, target TEXT NOT NULL, status TEXT NOT NULL, started_at DOUBLE PRECISION NOT NULL,"
    " finished_at DOUBLE PRECISION, idempotency_key TEXT, output_json TEXT, error_json TEXT, usage_json TEXT,"
    " attempts INTEGER NOT NULL DEFAULT 1, input_json TEXT, tenant_id TEXT, user_id TEXT, roles_json TEXT,"
    " policy_json TEXT)",
    # databases created before identity existed (mirrors the SQLite in-place migration)
    "ALTER TABLE {executions} ADD COLUMN IF NOT EXISTS tenant_id TEXT",
    "ALTER TABLE {executions} ADD COLUMN IF NOT EXISTS user_id TEXT",
    "ALTER TABLE {executions} ADD COLUMN IF NOT EXISTS roles_json TEXT",
    "ALTER TABLE {executions} ADD COLUMN IF NOT EXISTS policy_json TEXT",
    "CREATE UNIQUE INDEX IF NOT EXISTS {executions}_ux_key ON {executions}(idempotency_key)"
    " WHERE idempotency_key IS NOT NULL",
    "CREATE INDEX IF NOT EXISTS {executions}_ix_parent ON {executions}(parent_execution_id)",
    "CREATE INDEX IF NOT EXISTS {executions}_ix_status_time ON {executions}(status, finished_at)",
    "CREATE TABLE IF NOT EXISTS {events} ("
    " id BIGSERIAL PRIMARY KEY, execution_id TEXT NOT NULL, name TEXT NOT NULL,"
    " ts DOUBLE PRECISION NOT NULL, data_json TEXT NOT NULL)",
    "CREATE INDEX IF NOT EXISTS {events}_ix_exec ON {events}(execution_id, id)",
    "CREATE INDEX IF NOT EXISTS {events}_ix_ts ON {events}(ts)",
    "CREATE TABLE IF NOT EXISTS {steps} ("
    " execution_id TEXT NOT NULL, step TEXT NOT NULL, status TEXT NOT NULL, data_json TEXT,"
    " created_at DOUBLE PRECISION NOT NULL, PRIMARY KEY (execution_id, step))",
)


class _PgTx:
    def __init__(self, conn: Any):
        self.conn = conn

    def q(self, sql: str, params: Sequence[Any] = ()) -> list[Any]:
        with self.conn.cursor() as cur:
            cur.execute(sql, tuple(params))
            return list(cur.fetchall())

    def x(self, sql: str, params: Sequence[Any] = ()) -> int:
        with self.conn.cursor() as cur:
            cur.execute(sql, tuple(params))
            return int(cur.rowcount)


class PostgresStore(SqlStore):
    """`dsn` is a libpq connection string or URL (`postgresql://user@host/db`). Keep the password out of it
    and in `PGPASSWORD`/`~/.pgpass`, or pass a `draco.security.Secret`. Tables are named
    `<table_prefix>executions|events|steps` (default prefix `draco_`) so they can share a database.

    One connection guarded by a lock, used off the event loop; a broken connection is re-opened on the
    next call. Each operation is one transaction."""

    placeholder = "%s"
    lock_victims = " FOR UPDATE"  # a concurrent writer waits for the purge to commit

    def __init__(self, dsn: Any, *, table_prefix: str = "draco_", cipher: PayloadCipher | None = None,
                 connect_timeout_s: float = 10.0):
        super().__init__(table_prefix=table_prefix, cipher=cipher)
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ConfigurationError("PostgresStore needs psycopg 3: pip install 'draco[postgres]'") from exc
        self._psycopg, self._dict_row = psycopg, dict_row
        self._dsn = dsn
        self._timeout = connect_timeout_s
        self._lock = threading.Lock()
        self._conn: Any = None
        with self._lock:
            conn = self._connect()
            try:
                with conn.transaction():
                    with conn.cursor() as cur:
                        cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (self.tables["executions"],))
                        for stmt in _SCHEMA:
                            cur.execute(stmt.format(**self.tables))
            except psycopg.Error as exc:
                raise PersistenceError(f"cannot open state database: {type(exc).__name__}", retryable=False) from exc

    def _connect(self) -> Any:
        if self._conn is not None and not self._conn.closed and not self._conn.broken:
            return self._conn
        dsn = self._dsn.reveal() if hasattr(self._dsn, "reveal") else str(self._dsn)
        try:
            self._conn = self._psycopg.connect(dsn, autocommit=True, row_factory=self._dict_row,
                                               connect_timeout=max(1, int(self._timeout)))
        except self._psycopg.Error as exc:  # the DSN may contain credentials: report only the type
            raise PersistenceError(f"cannot connect to state database: {type(exc).__name__}") from exc
        return self._conn

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    async def _run(self, fn: Callable[[Tx], T]) -> T:
        def work() -> T:
            with self._lock:
                conn = self._connect()
                try:
                    with conn.transaction():  # commit or rollback
                        return fn(_PgTx(conn))
                except self._psycopg.Error as exc:
                    raise PersistenceError(f"state store error: {type(exc).__name__}") from exc
        return await asyncio.to_thread(work)

    async def drop_tables(self) -> None:
        """Delete the store's tables and all their data (tests and teardown only)."""
        await self._run(lambda tx: tx.x("DROP TABLE IF EXISTS {} , {}, {}".format(*self.tables.values())))
