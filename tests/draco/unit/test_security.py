"""Identity, tenant isolation, authorization and audit: enforced before protected side effects, failing closed."""
import pytest

from draco.core import DracoSettings, ExecutionContext, Policy, PolicyViolation, Status
from draco.core.identity import Identity
from draco.harness import Harness
from draco.security import Resource, RoleAuthorizer, Secret
from draco.state.stores import ExecutionRecord

ALICE = Identity("alice", "acme", frozenset({"analyst"}))
BOB = Identity("bob", "globex", frozenset({"analyst"}))


def make(**kw):
    h = Harness(DracoSettings(), **kw)
    seen = []

    async def w(ctx, inp):
        seen.append((ctx.user_id, ctx.tenant_id, ctx.identity))
        return inp
    h.register("w", w)
    return h, seen


@pytest.mark.parametrize("bad", ["", "  ", "anonymous", "Guest", "*"])
def test_placeholder_identity_values_are_rejected(bad):
    with pytest.raises(PolicyViolation):
        Identity(bad, "acme")
    with pytest.raises(PolicyViolation):
        Identity("alice", bad)


async def test_identity_is_the_source_of_user_and_tenant_and_cannot_be_contradicted():
    h, seen = make()
    r = await h.run("w", 1, identity=ALICE)
    assert r.status == Status.completed and seen == [("alice", "acme", ALICE)]
    r = await h.run("w", 1, identity=ALICE, tenant_id="globex")  # caller-supplied tenant contradicts identity
    assert r.status == Status.failed and r.error.code == "policy" and len(seen) == 1
    r = await h.run("w", 1, identity=ALICE, user_id="mallory")
    assert r.error.code == "policy"


async def test_children_inherit_identity_and_cannot_change_it_or_widen_policy():
    h, _ = make()
    out = {}

    async def parent(ctx, inp):
        same = await h.run("w", 1, parent=ctx)
        other = await h.run("w", 1, parent=ctx, identity=BOB)
        tenant = await h.run("w", 1, parent=ctx, tenant_id="globex")
        wide = await h.run("w", 1, parent=ctx, policy=Policy(permissions=frozenset({"read", "write"})))
        narrow = await h.run("w", 1, parent=ctx, policy=Policy(permissions=frozenset()))
        meta = await h.run("w", 1, parent=ctx, metadata={"tenant_id": "globex"})
        out.update(same=same, other=other, tenant=tenant, wide=wide, narrow=narrow, meta=meta)
        return "ok"
    h.register("parent", parent)
    assert (await h.run("parent", identity=ALICE)).status == Status.completed
    assert out["same"].status == Status.completed and out["narrow"].status == Status.completed
    for k in ("other", "tenant", "wide", "meta"):
        assert out[k].status == Status.failed and out[k].error.code == "policy", k


async def test_untrusted_metadata_cannot_carry_trusted_fields():
    h, _ = make()
    r = await h.run("w", 1, identity=ALICE, metadata={"identity": "x"})
    assert r.error.code == "policy"


async def test_require_identity_fails_closed_and_audits():
    h, seen = make(authorizer=RoleAuthorizer(require_identity=True))
    denied = []
    h.on("security.denied", lambda e: denied.append(e.data))
    r = await h.run("w", 1)  # no identity
    assert r.status == Status.failed and r.error.code == "policy" and seen == []
    assert denied and denied[0]["action"] == "workload.run" and denied[0]["subject"] is None
    assert (await h.run("w", 1, identity=ALICE)).status == Status.completed
    # a bare tenant_id claim is not an identity
    assert (await h.run("w", 1, tenant_id="acme")).status == Status.failed


async def test_role_grants_decide_workload_access_and_broken_authorizer_denies():
    h, seen = make(authorizer=RoleAuthorizer({"analyst": {"workload.*"}}))
    assert (await h.run("w", 1, identity=ALICE)).ok
    viewer = Identity("v", "acme", frozenset({"viewer"}))
    assert (await h.run("w", 1, identity=viewer)).status == Status.failed

    class Boom:
        def authorize(self, *a):
            raise RuntimeError("db down")
    h2, seen2 = make(authorizer=Boom())
    assert (await h2.run("w", 1, identity=ALICE)).status == Status.failed and seen2 == []


async def test_idempotency_keys_are_tenant_scoped():
    h, seen = make()
    await h.run("w", "A-data", identity=ALICE, idempotency_key="k")
    b = await h.run("w", "B-data", identity=BOB, idempotency_key="k")
    assert b.output == "B-data" and not b.replayed and len(seen) == 2  # not replayed from acme
    again = await h.run("w", "x", identity=ALICE, idempotency_key="k")
    assert again.replayed and again.output == "A-data"


async def test_resume_and_cancel_are_tenant_isolated():
    h = Harness(DracoSettings())
    n = {"c": 0}

    async def flaky(ctx, inp):
        n["c"] += 1
        if n["c"] == 1:
            raise RuntimeError("boom")
        return "done"
    h.register("flaky", flaky)
    r = await h.run("flaky", identity=ALICE)
    assert r.status == Status.failed
    from draco.core import ContractError
    with pytest.raises(ContractError):  # bob cannot even learn that it exists
        await h.resume(r.execution_id, identity=BOB)
    with pytest.raises(ContractError):
        await h.resume(r.execution_id)
    ok = await h.resume(r.execution_id, identity=ALICE)
    assert ok.status == Status.completed
    rec = await h.store.get_execution(r.execution_id)
    assert (rec.tenant_id, rec.user_id) == ("acme", "alice")

    import asyncio
    started = asyncio.Event()

    async def slow(ctx, inp):
        started.set()
        await asyncio.sleep(5)
    h.register("slow", slow)
    task = asyncio.create_task(h.run("slow", identity=ALICE))
    await started.wait()
    eid = next(iter(h._inflight))
    assert h.cancel(eid, identity=BOB) is False and h.cancel(eid) is False
    assert h.cancel(eid, identity=ALICE) is True
    assert (await task).status == Status.cancelled


async def test_sqlite_store_persists_identity_and_migrates_old_schema(tmp_path):
    import sqlite3

    from draco.state.sqlite import SqliteStore
    path = str(tmp_path / "old.db")
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE executions (execution_id TEXT PRIMARY KEY, parent_execution_id TEXT, session_id TEXT,"
               " agent_id TEXT, correlation_id TEXT, target TEXT NOT NULL, status TEXT NOT NULL, started_at REAL NOT NULL,"
               " finished_at REAL, idempotency_key TEXT, output_json TEXT, error_json TEXT, usage_json TEXT,"
               " attempts INTEGER NOT NULL DEFAULT 1, input_json TEXT)")
    db.execute("INSERT INTO executions(execution_id,target,status,started_at) VALUES ('old','w','failed',1.0)")
    db.commit(), db.close()
    store = SqliteStore(path)
    assert (await store.get_execution("old")).tenant_id is None
    await store.save_execution(ExecutionRecord("n", "w", "running", 2.0, tenant_id="acme", user_id="alice", roles=["r"]))
    rec = await store.get_execution("n")
    assert (rec.tenant_id, rec.user_id, rec.roles) == ("acme", "alice", ["r"])


def test_secret_never_renders_its_value():
    import copy
    import pickle
    s = Secret("sk-live-123456789")
    assert "123456789" not in repr(s) and "123456789" not in f"{s}" and s.reveal() == "sk-live-123456789"
    with pytest.raises(TypeError):
        pickle.dumps(s)
    with pytest.raises(TypeError):
        copy.deepcopy(s)


async def test_policy_narrowing_rules():
    root = Policy(permissions=frozenset({"read", "write"}), allowed_tools=frozenset({"a", "b"}), denied_tools=frozenset({"x"}))
    assert Policy(permissions=frozenset({"read"}), allowed_tools=frozenset({"a"}), denied_tools=frozenset({"x"})).narrows(root)
    assert Policy(permissions=frozenset({"read"}), allowed_tools=None, denied_tools=frozenset({"x"})).narrows(root)  # None = no tools
    assert not Policy(allowed_tools=frozenset({"a"})).narrows(Policy())  # parent has no tools: cannot grant one
    assert not Policy(permissions=frozenset({"read"}), allowed_tools=frozenset({"a"})).narrows(root)  # drops a denial
    assert not Policy(allow_approval_required=True).narrows(Policy())
    ctx = ExecutionContext(policy=root)
    with pytest.raises(PolicyViolation):
        ctx.child(policy=Policy(permissions=frozenset({"admin"})))


def test_resource_defaults():
    assert Resource("tool", "t").tenant_id is None


async def test_audit_events_pseudonymise_subject_and_tenant_in_traces_and_persisted_events():
    h, _ = make(authorizer=RoleAuthorizer({"analyst": {"workload.*"}}))
    r = await h.run("w", 1, identity=ALICE)
    trace = str(h.tracer.trace(r.execution_id))
    persisted = str(await h.store.list_events(r.execution_id))
    for dump in (trace, persisted):
        assert "security.allowed" in dump and "alice" not in dump and "acme" not in dump
