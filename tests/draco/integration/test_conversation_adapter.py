"""draco.adapters.conversation: the starter app's SqliteConversationStore used from Draco workloads.
Skipped when the `app` package (and its dependencies) is not importable, e.g. in a wheel-only install."""
import json

import pytest

pytest.importorskip("app.inference.memory", reason="the starter `app` package is not importable")

from draco.adapters.conversation import sqlite_conversation_store  # noqa: E402
from draco.core import DracoSettings, Identity, Status  # noqa: E402
from draco.harness import Harness  # noqa: E402


async def test_app_conversation_store_used_by_a_workload_keeps_sessions_separate(tmp_path):
    store = sqlite_conversation_store(str(tmp_path / "conv.db"))
    h = Harness(DracoSettings())

    async def chat(ctx, msg):
        history = store.load(ctx.user_id, ctx.session_id, 10).recent
        answer = f"{len(history)} earlier messages"
        store.save_turn(ctx.user_id, ctx.session_id, msg, json.dumps({"answer": answer, "metadata": {"request_id": ctx.execution_id}}), answer, {})  # run id = execution id
        return answer
    h.register("chat", chat)
    a1 = await h.run("chat", "hi", user_id="alice", session_id="c1")
    a2 = await h.run("chat", "again", user_id="alice", session_id="c1")
    b1 = await h.run("chat", "hi", user_id="bob", session_id="c2")
    assert [r.status for r in (a1, a2, b1)] == [Status.completed] * 3
    assert (a1.output, a2.output, b1.output) == ("0 earlier messages", "2 earlier messages", "0 earlier messages")
    runs = store._conn().execute("SELECT run_id FROM inference_runs").fetchall()
    assert {r["run_id"] for r in runs} == {a1.execution_id, a2.execution_id, b1.execution_id}  # turns tied to executions
    # the app store enforces ownership: bob cannot read alice's conversation
    r = await h.run("chat", "peek", user_id="bob", session_id="c1")
    assert r.status == Status.failed


async def test_conversation_survives_reopen_and_idempotent_replay_does_not_duplicate_turns(tmp_path):
    path = str(tmp_path / "conv.db")
    h = Harness(DracoSettings())
    alice = Identity("alice", "acme")

    def make_chat(store):
        async def chat(ctx, msg):
            history = store.load(ctx.user_id, ctx.session_id, 10).recent
            answer = f"turn {len(history) // 2 + 1}"
            store.save_turn(ctx.user_id, ctx.session_id, msg, json.dumps({"answer": answer, "metadata": {"request_id": ctx.execution_id}}), answer, {})
            return answer
        return chat
    h.register("chat", make_chat(sqlite_conversation_store(path)))
    first = await h.run("chat", "hello", identity=alice, session_id="c1", idempotency_key="m1")
    replay = await h.run("chat", "hello", identity=alice, session_id="c1", idempotency_key="m1")
    assert first.output == "turn 1" and replay.replayed and replay.output == "turn 1"
    reopened = sqlite_conversation_store(path)  # a fresh store over the same file (e.g. after a restart)
    assert reopened.owner("c1") == "alice"
    assert [m["content"] for m in reopened.load("alice", "c1", 10).recent] == ["hello", "turn 1"]  # replay wrote nothing
    h2 = Harness(DracoSettings())
    h2.register("chat", make_chat(reopened))
    second = await h2.run("chat", "again", identity=alice, session_id="c1")
    assert second.output == "turn 2"
    with pytest.raises(PermissionError):
        reopened.load("mallory", "c1", 10)
