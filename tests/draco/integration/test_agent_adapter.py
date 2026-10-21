"""The existing LangGraph AgentRunner and arbitrary compiled graphs, run through the harness."""
import pytest
from app.inference.schemas import InferenceError

from draco.adapters import agent_runner_workload
from draco.adapters.langgraph import graph_workload
from draco.core import DracoSettings, Status
from draco.harness import Harness
from tests.agent_helpers import FakeRetrieval, ai, cand, make_runner


async def test_agent_runner_through_harness_returns_normalized_result():
    runner, _ = make_runner([ai("Revenue was 5 million dollars [S1].")],
                            retrieval=FakeRetrieval([cand(1, "Revenue was 5 million dollars in 2023.")]))
    h = Harness(DracoSettings(default_timeout_s=10))
    h.register("assistant", agent_runner_workload(runner))
    r = await h.run("assistant", "What was the revenue?", user_id="alice", session_id="S")
    assert r.status == Status.completed
    assert r.output["status"] == "ok" and r.output["citations"][0]["ref"] == "S1"
    assert r.agent_id == "assistant" and r.session_id == "S"
    rec = await h.store.get_execution(r.execution_id)
    assert rec.status == "completed" and rec.output["answer"].startswith("Revenue")


async def test_agent_runner_errors_are_typed_and_invalid_input_is_a_contract_error():
    runner, _ = make_runner([InferenceError("Model server down", "upstream_unavailable", 503)],
                            retrieval=FakeRetrieval([cand(1, "x")]))
    h = Harness(DracoSettings(default_timeout_s=10))
    h.register("assistant", agent_runner_workload(runner))
    r = await h.run("assistant", "What is x?", user_id="alice")
    assert r.status == Status.failed and r.error.code == "provider" and r.error.retryable
    r = await h.run("assistant", {"message": ""})
    assert r.error.code == "contract"


async def test_agent_runs_are_isolated_across_concurrent_sessions():
    import asyncio
    runner, _ = make_runner([ai("A [S1]."), ai("B [S1].")], retrieval=FakeRetrieval([cand(1, "alpha beta")]))
    h = Harness(DracoSettings(default_timeout_s=10))
    h.register("assistant", agent_runner_workload(runner))
    rs = await asyncio.gather(h.run("assistant", "q1", user_id="u1", session_id="s1"),
                              h.run("assistant", "q2", user_id="u2", session_id="s2"))
    assert all(r.status == Status.completed for r in rs)
    assert rs[0].execution_id != rs[1].execution_id


async def test_compiled_langgraph_runs_with_execution_id_as_thread():
    langgraph = pytest.importorskip("langgraph.graph")
    from typing import TypedDict

    class S(TypedDict):
        n: int


    def inc(state):
        return {"n": state["n"] + 1}
    g = langgraph.StateGraph(S)
    g.add_node("inc", inc)
    g.set_entry_point("inc")
    g.set_finish_point("inc")
    from langgraph.checkpoint.memory import InMemorySaver
    saver = InMemorySaver()
    compiled = g.compile(checkpointer=saver)
    h = Harness(DracoSettings())
    h.register("graph", graph_workload(compiled, to_state=lambda n: {"n": n}))
    r = await h.run("graph", 41)
    assert r.status == Status.completed and r.output == {"n": 42}
    assert saver.get_tuple({"configurable": {"thread_id": r.execution_id}}) is not None  # keyed to the execution
