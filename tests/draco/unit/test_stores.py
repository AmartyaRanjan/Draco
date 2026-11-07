"""One contract suite for every state backend: in-memory, SQLite, and Postgres when DRACO_TEST_POSTGRES_URL
is set (e.g. postgresql://postgres:test@127.0.0.1:5432/postgres). Covers the protocols, retention and
(for the SQL stores) encryption at rest."""
import asyncio
import importlib.util
import os
import time
import uuid

import pytest

from draco.core import DracoSettings, PersistenceError, Status, Waiting
from draco.harness import Harness, step
from draco.security import FernetCipher, Secret
from draco.state import (
    CheckpointStore,
    EventStore,
    ExecutionRecord,
    ExecutionStore,
    InMemoryStore,
    PurgeResult,
    RetentionStore,
    SqliteStore,
    apply_retention,
)

PG_URL = os.environ.get("DRACO_TEST_POSTGRES_URL")
needs_pg = pytest.mark.skipif(not PG_URL, reason="set DRACO_TEST_POSTGRES_URL to run against Postgres")
DAY = 86400.0
NOW = 1_800_000_000.0
OLD = NOW - 40 * DAY


def _pg_store(prefix=None, cipher=None):
    from draco.state.postgres import PostgresStore
    return PostgresStore(PG_URL, table_prefix=prefix or f"t{uuid.uuid4().hex[:10]}_", cipher=cipher)


class Backends:
    """Creates stores of one kind; tracks them for cleanup. `same()` reopens the same database."""

    def __init__(self, kind, tmp_path):
        self.kind, self.tmp, self.made = kind, tmp_path, []
        self.prefix = f"t{uuid.uuid4().hex[:10]}_"

    def new(self, cipher=None):
        if self.kind == "memory":
            s = InMemoryStore()
        elif self.kind == "sqlite":
            s = SqliteStore(str(self.tmp / "state.db"), cipher=cipher)
        else:
            s = _pg_store(self.prefix, cipher)
        self.made.append(s)
        return s

    def close(self):
        if self.kind == "postgres" and self.made:
            asyncio.run(self.made[-1].drop_tables())
        for s in self.made:
            if hasattr(s, "close"):
                s.close()


ALL = ["memory", "sqlite", pytest.param("postgres", marks=needs_pg)]
SQL = ["sqlite", pytest.param("postgres", marks=needs_pg)]


@pytest.fixture(params=ALL)
def backends(request, tmp_path):
    b = Backends(request.param, tmp_path)
    yield b
    b.close()


@pytest.fixture(params=SQL)
def sql_backends(request, tmp_path):
    b = Backends(request.param, tmp_path)
    yield b
    b.close()


def rec(eid, status="completed", *, at=NOW, parent=None, **kw):
    return ExecutionRecord(eid, "w", status, at, parent_execution_id=parent,
                           finished_at=None if status in ("running", "waiting") else at + 1, **kw)


# -- protocol contract ---------------------------------------------------------------------------------
async def test_store_implements_every_protocol_and_round_trips(backends):
    s = backends.new()
    for proto in (ExecutionStore, EventStore, CheckpointStore, RetentionStore):
        assert isinstance(s, proto)
    r = rec("e1", "failed", idempotency_key="acme\x1fk1", tenant_id="acme", user_id="alice", roles=["analyst"],
            input={"q": "hello", "n": [1, 2]}, output=None, error={"code": "x", "message": "boom"},
            usage={"tokens": 3}, policy={"permissions": ["read"]}, session_id="s1", agent_id="a", correlation_id="c")
    await s.save_execution(r)
    got = await s.get_execution("e1")
    assert got == r
    assert (await s.find_by_key("acme\x1fk1")).execution_id == "e1"
    assert await s.find_by_key("nope") is None and await s.get_execution("nope") is None
    r.status, r.output, r.attempts = "completed", {"answer": 42}, 2
    await s.save_execution(r)  # upsert updates status/output/attempts
    got = await s.get_execution("e1")
    assert (got.status, got.output, got.attempts, got.tenant_id) == ("completed", {"answer": 42}, 2, "acme")
    await s.save_execution(rec("c2", at=NOW + 2, parent="e1"))
    await s.save_execution(rec("c1", at=NOW + 1, parent="e1"))
    assert [c.execution_id for c in await s.list_children("e1")] == ["c1", "c2"]
    await s.append_event("e1", "a", 1.0, {"x": 1})
    await s.append_event("e1", "b", 2.0, {"y": [2]})
    assert await s.list_events("e1") == [{"name": "a", "ts": 1.0, "data": {"x": 1}}, {"name": "b", "ts": 2.0, "data": {"y": [2]}}]
    await s.begin_step("e1", "fetch")
    assert (await s.get_step("e1", "fetch")).status == "started"
    await s.complete_step("e1", "fetch", {"rows": 3})
    st = await s.get_step("e1", "fetch")
    assert (st.status, st.data) == ("done", {"rows": 3})
    await s.begin_step("e1", "fetch")  # re-begin clears the result
    assert (await s.get_step("e1", "fetch")).data is None
    await s.complete_step("e1", "second", None)
    assert [x.step for x in await s.list_steps("e1")] == ["fetch", "second"]


async def test_harness_run_and_resume_through_the_store(backends):
    h = Harness(DracoSettings(), store=backends.new())
    calls = {"n": 0}

    async def w(ctx, x):
        async def once():
            calls["n"] += 1
            return x * 2
        v = await step(ctx, "double", once)
        if ctx.metadata.get("resumed"):
            return v
        return Waiting({"need": "approval"})
    h.register("w", w)
    r = await h.run("w", 21, tenant_id="acme")
    assert r.status == Status.waiting
    assert (await h.store.get_execution(r.execution_id)).status == "waiting"
    assert any(e["name"] == "execution.started" for e in await h.store.list_events(r.execution_id))
    from draco.core import Identity
    r2 = await h.resume(r.execution_id, identity=Identity("u", tenant_id="acme"))
    assert (r2.status, r2.output, calls["n"]) == (Status.completed, 42, 1)  # step replayed, not re-run


# -- retention ---------------------------------------------------------------------------------------------
async def _seed(s):
    await s.save_execution(rec("old_done", at=OLD))
    await s.save_execution(rec("old_failed", "failed", at=OLD))
    await s.save_execution(rec("old_waiting", "waiting", at=OLD))
    await s.save_execution(rec("old_running", "running", at=OLD))
    await s.save_execution(rec("old_child_of_waiting", "completed", at=OLD, parent="old_waiting"))
    await s.save_execution(rec("old_grandchild", "failed", at=OLD, parent="old_child_of_waiting"))
    await s.save_execution(rec("old_child_of_done", at=OLD, parent="old_done"))
    await s.save_execution(rec("new_done", at=NOW - DAY))
    for eid in ("old_done", "old_failed", "old_waiting", "old_running", "old_child_of_waiting", "old_grandchild",
                "old_child_of_done", "new_done"):
        await s.append_event(eid, "old", OLD, {"k": eid})
        await s.append_event(eid, "new", NOW - 1, {"k": eid})
        await s.complete_step(eid, "s", {"v": 1})
    await s.append_event("ghost", "orphan", OLD, {})  # no execution record
    await s.append_event("ghost2", "orphan", NOW - 1, {})


async def test_purge_removes_old_terminal_executions_and_never_active_ones(backends):
    s = backends.new()
    await _seed(s)
    res = await s.purge(older_than=NOW - 30 * DAY)
    assert res == PurgeResult(executions=3, events=7, steps=3)  # 3 records x 2 events + 1 orphan; 3 x 1 step
    for gone in ("old_done", "old_failed", "old_child_of_done"):
        assert await s.get_execution(gone) is None
        assert await s.list_events(gone) == [] and await s.list_steps(gone) == []
    for kept in ("old_waiting", "old_running", "old_child_of_waiting", "old_grandchild", "new_done"):
        assert await s.get_execution(kept) is not None, kept
        assert len(await s.list_events(kept)) == 2 and len(await s.list_steps(kept)) == 1
    assert await s.list_events("ghost") == [] and len(await s.list_events("ghost2")) == 1
    assert await s.purge(older_than=NOW - 30 * DAY) == PurgeResult()  # idempotent


async def test_event_retention_keeps_records_and_active_events(backends):
    s = backends.new()
    await _seed(s)
    res = await s.purge(events_older_than=NOW - 7 * DAY)
    # old events of: old_done, old_failed, old_child_of_done, new_done, ghost; protected: waiting/running + descendants
    assert res == PurgeResult(executions=0, events=5, steps=0)
    assert [e["name"] for e in await s.list_events("old_done")] == ["new"]
    assert await s.get_execution("old_done") is not None and len(await s.list_steps("old_done")) == 1
    for kept in ("old_waiting", "old_running", "old_child_of_waiting", "old_grandchild"):
        assert len(await s.list_events(kept)) == 2


async def test_apply_retention_from_settings(backends):
    s = backends.new()
    await _seed(s)
    assert await apply_retention(s, DracoSettings(), now=NOW) == PurgeResult()  # nothing configured: no-op
    st = DracoSettings.from_env({"DRACO_EXECUTION_RETENTION_DAYS": "30", "DRACO_EVENT_RETENTION_DAYS": "0.5"})
    res = await apply_retention(s, st, now=NOW)
    assert res.executions == 3
    assert [e["name"] for e in await s.list_events("new_done")] == ["new"]  # its 40-day-old event is gone
    assert len(await s.list_events("old_waiting")) == 2


async def test_apply_retention_rejects_a_store_without_purge():
    from draco.core import ConfigurationError

    class Plain:
        pass
    with pytest.raises(ConfigurationError):
        await apply_retention(Plain(), DracoSettings(event_retention_days=1))


# -- encryption at rest ----------------------------------------------------------------------------------
needs_crypto = pytest.mark.skipif(importlib.util.find_spec("cryptography") is None,
                                  reason="encryption needs the 'crypto' extra (cryptography)")
MARK = "TOP-SECRET-PAYLOAD-7f3a"


def _key():
    return FernetCipher.generate_key()


def _cipher(*keys):
    return FernetCipher([Secret(k) for k in keys])


async def _raw_text(b, s) -> str:
    """Everything the database holds, as text, read around the store."""
    if b.kind == "sqlite":
        s.close()
        data = b"".join(p.read_bytes() for p in b.tmp.iterdir() if p.name.startswith("state.db"))
        return data.decode("latin-1")
    import psycopg
    out = []
    with psycopg.connect(PG_URL) as conn:
        for t in s.tables.values():
            out += [repr(r) for r in conn.execute(f"SELECT * FROM {t}").fetchall()]
    return "\n".join(out)


@needs_crypto
async def test_encrypted_store_is_transparent_and_raw_db_has_no_plaintext(sql_backends):
    b = sql_backends
    s = b.new(cipher=_cipher(FernetCipher.generate_key()))
    h = Harness(DracoSettings(), store=s)

    async def w(ctx, x):
        await step(ctx, "remember", lambda: _echo(x))
        return {"echo": x}
    h.register("w", w)
    r = await h.run("w", {"note": MARK})
    assert r.status == Status.completed
    got = await s.get_execution(r.execution_id)
    assert got.input == {"note": MARK} and got.output == {"echo": {"note": MARK}}
    assert (await s.get_step(r.execution_id, "remember")).data == {"note": MARK}
    await s.append_event(r.execution_id, "custom", time.time(), {"payload": MARK})
    assert (await s.list_events(r.execution_id))[-1]["data"] == {"payload": MARK}
    raw = await _raw_text(b, s)
    assert r.execution_id in raw  # sanity: we are looking at the right database
    assert MARK not in raw and "enc:v1:" in raw


async def _echo(x):
    return x


@needs_crypto
async def test_key_rotation_and_missing_or_wrong_key(sql_backends):
    b = sql_backends
    k_old, k_new = FernetCipher.generate_key(), FernetCipher.generate_key()
    plain = b.new()  # rows written before encryption was enabled stay readable
    await plain.save_execution(rec("legacy", input={"m": MARK}))
    s1 = b.new(cipher=_cipher(k_old))
    await s1.save_execution(rec("e", input={"m": MARK}))
    await s1.append_event("e", "x", 1.0, {"m": MARK})
    assert (await s1.get_execution("legacy")).input == {"m": MARK}
    with pytest.raises(PersistenceError):  # encrypted rows, no key
        await b.new().get_execution("e")
    with pytest.raises(PersistenceError):  # encrypted rows, wrong key
        await b.new(cipher=_cipher(k_new)).get_execution("e")
    s2 = b.new(cipher=_cipher(k_new, k_old))  # new primary, old still accepted
    assert (await s2.get_execution("e")).input == {"m": MARK}
    assert await s2.rotate_encryption() >= 3  # e.input, legacy.input, the event, the run's json columns
    s3 = b.new(cipher=_cipher(k_new))  # old key retired
    assert (await s3.get_execution("e")).input == {"m": MARK}
    assert (await s3.get_execution("legacy")).input == {"m": MARK}
    assert (await s3.list_events("e"))[0]["data"] == {"m": MARK}
    if b.kind == "sqlite":  # the plaintext written before encryption lingers in the WAL until checkpointed
        for other in b.made[:-1]:
            other.close()
        await s3.vacuum()
    assert MARK not in await _raw_text(b, s3)


@needs_crypto
def test_cipher_never_reveals_keys():
    from draco.core import ConfigurationError
    k = FernetCipher.generate_key()
    c = FernetCipher.from_secret(Secret(f"{k}, {FernetCipher.generate_key()}"))
    assert c.key_count == 2 and k not in repr(c)
    assert c.decrypt(c.encrypt("hi")) == "hi"
    with pytest.raises(ConfigurationError) as e:
        FernetCipher([Secret("not-a-key-but-sensitive")])
    assert "sensitive" not in str(e.value)


@needs_crypto
def test_open_store_from_settings(tmp_path, monkeypatch):
    from draco.core import ConfigurationError
    from draco.harness import open_store
    assert isinstance(open_store(DracoSettings()), InMemoryStore)
    monkeypatch.setenv("DRACO_STATE_KEY", FernetCipher.generate_key())
    st = DracoSettings(state_path=str(tmp_path / "x.db"), state_encryption_key="DRACO_STATE_KEY")
    s = open_store(st)
    assert isinstance(s, SqliteStore) and s.cipher is not None
    s.close()
    h = Harness(st)  # the harness builds the configured store itself
    assert isinstance(h.store, SqliteStore) and h.store.cipher is not None
    h.store.close()
    with pytest.raises(ConfigurationError):
        open_store(DracoSettings(state_encryption_key="DRACO_STATE_KEY"))  # nowhere to encrypt
    with pytest.raises(ConfigurationError):
        open_store(DracoSettings(state_url="mysql://x"))
    with pytest.raises(ConfigurationError) as e:  # a password in the URL would end up in logs and reprs
        DracoSettings.from_env({"DRACO_STATE_URL": "postgresql://u:hunter2@db/x"})
    assert "hunter2" not in str(e.value)
    with pytest.raises(ConfigurationError):
        open_store(DracoSettings(state_path=str(tmp_path / "y.db"), state_encryption_key="DRACO_MISSING_KEY"))


# -- Postgres specifics -------------------------------------------------------------------------------------
@needs_pg
async def test_postgres_migrates_a_pre_identity_schema_in_place():
    import psycopg
    prefix = f"t{uuid.uuid4().hex[:10]}_"
    with psycopg.connect(PG_URL, autocommit=True) as conn:
        conn.execute(f"CREATE TABLE {prefix}executions (execution_id TEXT PRIMARY KEY, parent_execution_id TEXT,"
                     " session_id TEXT, agent_id TEXT, correlation_id TEXT, target TEXT NOT NULL, status TEXT NOT NULL,"
                     " started_at DOUBLE PRECISION NOT NULL, finished_at DOUBLE PRECISION, idempotency_key TEXT,"
                     " output_json TEXT, error_json TEXT, usage_json TEXT, attempts INTEGER NOT NULL DEFAULT 1, input_json TEXT)")
        conn.execute(f"INSERT INTO {prefix}executions(execution_id,target,status,started_at) VALUES ('old','w','failed',1.0)")
    s = _pg_store(prefix)
    try:
        assert (await s.get_execution("old")).tenant_id is None
        await s.save_execution(ExecutionRecord("n", "w", "running", 2.0, tenant_id="acme", user_id="alice", roles=["r"]))
        got = await s.get_execution("n")
        assert (got.tenant_id, got.user_id, got.roles) == ("acme", "alice", ["r"])
        _pg_store(prefix).close()  # reopening an up-to-date schema is a no-op
    finally:
        await s.drop_tables()
        s.close()


@needs_pg
async def test_postgres_store_reconnects_after_the_connection_drops():
    s = _pg_store()
    try:
        await s.save_execution(rec("a"))
        s._conn.close()  # simulate a dropped connection
        assert (await s.get_execution("a")).execution_id == "a"
    finally:
        await s.drop_tables()
        s.close()


@needs_pg
def test_postgres_connection_errors_do_not_leak_the_dsn():
    from draco.state.postgres import PostgresStore
    with pytest.raises(PersistenceError) as e:
        PostgresStore("postgresql://user:hunter2-secret@127.0.0.1:1/db", connect_timeout_s=1)
    assert "hunter2" not in str(e.value) and "hunter2" not in repr(e.value.details)


def test_invalid_table_prefix_is_rejected():
    from draco.state._sql import SqlStore
    with pytest.raises(ValueError):
        SqlStore(table_prefix="x; DROP TABLE y")
