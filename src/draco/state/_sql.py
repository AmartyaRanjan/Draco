"""SQL logic shared by the SQLite and Postgres stores: one schema shape, one set of statements, optional
payload encryption and retention. Backends supply a transaction runner and their placeholder style."""
from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator, Sequence
from typing import Any, Protocol, TypeVar

from draco.core.errors import PersistenceError
from draco.state.stores import (
    ACTIVE_STATUSES,
    ENCRYPTED_PREFIX,
    ExecutionRecord,
    PayloadCipher,
    PurgeResult,
    StepRecord,
    dumps,
)

T = TypeVar("T")
_CHUNK = 500
_ACTIVE = ",".join(f"'{s}'" for s in sorted(ACTIVE_STATUSES))
# Executions that must survive any purge: waiting/running ones and all of their descendants.
_PROTECTED = (
    "WITH RECURSIVE protected(id) AS ("
    f" SELECT execution_id FROM {{executions}} WHERE status IN ({_ACTIVE})"
    " UNION SELECT c.execution_id FROM {executions} c JOIN protected p ON c.parent_execution_id = p.id)"
    " SELECT id FROM protected")
# (table, key columns, encrypted payload column)
_ENCRYPTED_COLUMNS = (("executions", ("execution_id",), "input_json"), ("executions", ("execution_id",), "output_json"),
                      ("executions", ("execution_id",), "error_json"), ("events", ("id",), "data_json"),
                      ("steps", ("execution_id", "step"), "data_json"))


class Tx(Protocol):
    def q(self, sql: str, params: Sequence[Any] = ()) -> list[Any]: ...  # rows support row["column"]
    def x(self, sql: str, params: Sequence[Any] = ()) -> int: ...  # returns the affected row count


class SqlStore:
    """ExecutionStore + EventStore + CheckpointStore + RetentionStore over a SQL backend.

    With a `cipher`, the payload columns (execution input/output/error, event data, step data) are
    encrypted before they reach the database; identifiers, status, timestamps, tenant/user/roles, usage
    and policy stay in clear text because lookups, ownership checks and retention query them."""

    placeholder = "?"
    lock_victims = ""  # e.g. " FOR UPDATE" where the backend supports row locks

    def __init__(self, *, table_prefix: str = "", cipher: PayloadCipher | None = None):
        if table_prefix and not (table_prefix.replace("_", "").isalnum() and table_prefix[0].isalpha()):
            raise ValueError("table_prefix must be alphanumeric/underscore and start with a letter")
        self.tables = {t: f"{table_prefix}{t}" for t in ("executions", "events", "steps")}
        self.cipher = cipher
        self._cache: dict[str, str] = {}

    async def _run(self, fn: Callable[[Tx], T]) -> T:
        raise NotImplementedError

    def sql(self, s: str) -> str:
        out = self._cache.get(s)
        if out is None:
            out = s.format(**self.tables)
            if self.placeholder != "?":
                out = out.replace("?", self.placeholder)
            self._cache[s] = out
        return out

    # -- payload encoding -------------------------------------------------------------------
    def _enc(self, v: Any) -> str:
        text = dumps(v)
        return ENCRYPTED_PREFIX + self.cipher.encrypt(text) if self.cipher is not None else text

    def _enc_opt(self, v: Any) -> str | None:
        return None if v is None else self._enc(v)

    def _dec(self, s: str | None, default: Any = None) -> Any:
        if not s:
            return default
        if s.startswith(ENCRYPTED_PREFIX):
            if self.cipher is None:
                raise PersistenceError("state is encrypted but no encryption key is configured", retryable=False)
            try:
                s = self.cipher.decrypt(s[len(ENCRYPTED_PREFIX):])
            except Exception as exc:  # never include the token or key material in the error
                raise PersistenceError("cannot decrypt persisted state (wrong or retired key)", retryable=False) from exc
        return json.loads(s)

    def _rec(self, r: Any) -> ExecutionRecord:
        return ExecutionRecord(
            execution_id=r["execution_id"], target=r["target"], status=r["status"],
            started_at=r["started_at"], parent_execution_id=r["parent_execution_id"],
            session_id=r["session_id"], agent_id=r["agent_id"], correlation_id=r["correlation_id"],
            finished_at=r["finished_at"], idempotency_key=r["idempotency_key"],
            output=self._dec(r["output_json"]), error=self._dec(r["error_json"]),
            usage=json.loads(r["usage_json"]) if r["usage_json"] else {}, attempts=r["attempts"],
            input=self._dec(r["input_json"]), tenant_id=r["tenant_id"], user_id=r["user_id"],
            roles=json.loads(r["roles_json"]) if r["roles_json"] else [],
            policy=json.loads(r["policy_json"]) if r["policy_json"] else None)

    def _step(self, r: Any) -> StepRecord:
        return StepRecord(r["execution_id"], r["step"], r["status"], self._dec(r["data_json"]), r["created_at"])

    # -- ExecutionStore ------------------------------------------------------------------------
    async def save_execution(self, rec: ExecutionRecord) -> None:
        sql = self.sql(
            "INSERT INTO {executions}(execution_id,parent_execution_id,session_id,agent_id,correlation_id,target,status,"
            "started_at,finished_at,idempotency_key,output_json,error_json,usage_json,attempts,input_json,tenant_id,user_id,"
            "roles_json,policy_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(execution_id) DO UPDATE SET status=excluded.status, finished_at=excluded.finished_at,"
            " idempotency_key=excluded.idempotency_key, output_json=excluded.output_json, error_json=excluded.error_json,"
            " usage_json=excluded.usage_json, attempts=excluded.attempts")
        params = (rec.execution_id, rec.parent_execution_id, rec.session_id, rec.agent_id, rec.correlation_id,
                  rec.target, rec.status, rec.started_at, rec.finished_at, rec.idempotency_key,
                  self._enc_opt(rec.output), self._enc_opt(rec.error), dumps(rec.usage), rec.attempts,
                  self._enc_opt(rec.input), rec.tenant_id, rec.user_id, dumps(rec.roles),
                  None if rec.policy is None else dumps(rec.policy))
        await self._run(lambda tx: tx.x(sql, params))

    async def get_execution(self, execution_id: str) -> ExecutionRecord | None:
        rows = await self._run(lambda tx: tx.q(self.sql("SELECT * FROM {executions} WHERE execution_id=?"), (execution_id,)))
        return self._rec(rows[0]) if rows else None

    async def find_by_key(self, key: str) -> ExecutionRecord | None:
        rows = await self._run(lambda tx: tx.q(self.sql("SELECT * FROM {executions} WHERE idempotency_key=?"), (key,)))
        return self._rec(rows[0]) if rows else None

    async def list_children(self, execution_id: str) -> list[ExecutionRecord]:
        rows = await self._run(lambda tx: tx.q(self.sql(
            "SELECT * FROM {executions} WHERE parent_execution_id=? ORDER BY started_at"), (execution_id,)))
        return [self._rec(r) for r in rows]

    # -- EventStore ------------------------------------------------------------------------------
    async def append_event(self, execution_id: str, name: str, ts: float, data: dict[str, Any]) -> None:
        payload = self._enc(data)
        await self._run(lambda tx: tx.x(self.sql(
            "INSERT INTO {events}(execution_id,name,ts,data_json) VALUES (?,?,?,?)"), (execution_id, name, ts, payload)))

    async def list_events(self, execution_id: str) -> list[dict[str, Any]]:
        rows = await self._run(lambda tx: tx.q(self.sql(
            "SELECT name, ts, data_json FROM {events} WHERE execution_id=? ORDER BY id"), (execution_id,)))
        return [{"name": r["name"], "ts": r["ts"], "data": self._dec(r["data_json"], {})} for r in rows]

    # -- CheckpointStore ---------------------------------------------------------------------------
    async def begin_step(self, execution_id: str, step: str) -> None:
        params = (execution_id, step, "started", None, time.time())
        await self._run(lambda tx: tx.x(self.sql(
            "INSERT INTO {steps}(execution_id,step,status,data_json,created_at) VALUES (?,?,?,?,?)"
            " ON CONFLICT(execution_id,step) DO UPDATE SET status='started', data_json=NULL,"
            " created_at=excluded.created_at"), params))

    async def complete_step(self, execution_id: str, step: str, data: Any) -> None:
        params = (execution_id, step, "done", self._enc(data), time.time())
        await self._run(lambda tx: tx.x(self.sql(
            "INSERT INTO {steps}(execution_id,step,status,data_json,created_at) VALUES (?,?,?,?,?)"
            " ON CONFLICT(execution_id,step) DO UPDATE SET status='done', data_json=excluded.data_json,"
            " created_at=excluded.created_at"), params))

    async def get_step(self, execution_id: str, step: str) -> StepRecord | None:
        rows = await self._run(lambda tx: tx.q(self.sql(
            "SELECT * FROM {steps} WHERE execution_id=? AND step=?"), (execution_id, step)))
        return self._step(rows[0]) if rows else None

    async def list_steps(self, execution_id: str) -> list[StepRecord]:
        rows = await self._run(lambda tx: tx.q(self.sql(
            "SELECT * FROM {steps} WHERE execution_id=? ORDER BY created_at"), (execution_id,)))
        return [self._step(r) for r in rows]

    # -- RetentionStore ----------------------------------------------------------------------------
    async def purge(self, *, older_than: float | None = None, events_older_than: float | None = None) -> PurgeResult:
        def work(tx: Tx) -> PurgeResult:
            n_exec = n_events = n_steps = 0
            if older_than is not None:
                victims = [r["execution_id"] for r in tx.q(self.sql(
                    f"SELECT execution_id FROM {{executions}} WHERE status NOT IN ({_ACTIVE})"
                    " AND COALESCE(finished_at, started_at) < ? AND execution_id NOT IN (" + _PROTECTED + ")"
                    + self.lock_victims), (older_than,))]
                for chunk in _chunks(victims):
                    marks = ",".join("?" * len(chunk))
                    n_events += tx.x(self.sql(f"DELETE FROM {{events}} WHERE execution_id IN ({marks})"), chunk)
                    n_steps += tx.x(self.sql(f"DELETE FROM {{steps}} WHERE execution_id IN ({marks})"), chunk)
                    n_exec += tx.x(self.sql(f"DELETE FROM {{executions}} WHERE execution_id IN ({marks})"
                                            f" AND status NOT IN ({_ACTIVE})"), chunk)
                # events/steps whose execution record never existed or is already gone
                n_events += tx.x(self.sql("DELETE FROM {events} WHERE ts < ? AND execution_id NOT IN"
                                          " (SELECT execution_id FROM {executions})"), (older_than,))
                n_steps += tx.x(self.sql("DELETE FROM {steps} WHERE created_at < ? AND execution_id NOT IN"
                                         " (SELECT execution_id FROM {executions})"), (older_than,))
            if events_older_than is not None:
                n_events += tx.x(self.sql("DELETE FROM {events} WHERE ts < ? AND execution_id NOT IN ("
                                          + _PROTECTED + ")"), (events_older_than,))
            return PurgeResult(n_exec, n_events, n_steps)
        return await self._run(work)

    # -- encryption maintenance ----------------------------------------------------------------------
    async def rotate_encryption(self) -> int:
        """Re-encrypt every payload under the cipher's current primary key (and encrypt any plaintext left
        from before encryption was enabled). Run after adding a new key, before retiring the old one."""
        cipher = self.cipher
        if cipher is None:
            raise PersistenceError("rotate_encryption needs a cipher", retryable=False)

        def work(tx: Tx) -> int:
            n = 0
            for table, keys, col in _ENCRYPTED_COLUMNS:
                rows = tx.q(self.sql(f"SELECT {', '.join(keys)}, {col} FROM {{{table}}} WHERE {col} IS NOT NULL"))
                where = " AND ".join(f"{k}=?" for k in keys)
                for r in rows:
                    v = r[col]
                    if v.startswith(ENCRYPTED_PREFIX):
                        try:
                            new = ENCRYPTED_PREFIX + cipher.rotate(v[len(ENCRYPTED_PREFIX):])
                        except Exception as exc:
                            raise PersistenceError("cannot decrypt persisted state (wrong or retired key)",
                                                   retryable=False) from exc
                    else:
                        new = ENCRYPTED_PREFIX + cipher.encrypt(v)
                    n += tx.x(self.sql(f"UPDATE {{{table}}} SET {col}=? WHERE {where}"), (new, *(r[k] for k in keys)))
            return n
        return await self._run(work)


def _chunks(ids: list[Any]) -> Iterator[list[Any]]:
    for i in range(0, len(ids), _CHUNK):
        yield ids[i:i + _CHUNK]
