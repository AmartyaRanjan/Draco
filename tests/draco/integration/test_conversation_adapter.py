import json

from draco.adapters.conversation import sqlite_conversation_store
from draco.core import DracoSettings, Status
from draco.harness import Harness


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
