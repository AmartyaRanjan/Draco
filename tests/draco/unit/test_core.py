import time

import pytest

from draco.core import (
    BudgetExhausted,
    BudgetLimits,
    BudgetTracker,
    ConfigurationError,
    DracoSettings,
    ErrorInfo,
    Event,
    EventBus,
    ExecutionCancelled,
    ExecutionContext,
    ExecutionTimeout,
    Policy,
    PolicyViolation,
    Status,
    status_for,
)


def test_child_context_propagates_identity_and_limits():
    root = ExecutionContext(session_id="s", agent_id="a", budget=BudgetTracker(BudgetLimits(max_depth=2)),
                            deadline=time.monotonic() + 10)
    c = root.child(agent_id="b")
    assert c.parent_execution_id == root.execution_id and c.correlation_id == root.correlation_id
    assert c.session_id == "s" and c.depth == 1 and c.budget is root.budget
    c2 = c.child()
    with pytest.raises(BudgetExhausted):
        c2.child()
    root.cancel.cancel("stop")
    assert c2.cancel.cancelled  # cancellation flows downward
    with pytest.raises(ExecutionCancelled):
        c2.check()


def test_child_does_not_cancel_parent_and_metadata_is_isolated():
    root = ExecutionContext(metadata={"k": 1})
    c = root.child(metadata={"z": 2})
    c.cancel.cancel()
    assert not root.cancel.cancelled
    assert "z" not in root.metadata and c.metadata == {"k": 1, "z": 2}


def test_deadline_check():
    ctx = ExecutionContext(deadline=time.monotonic() - 1)
    with pytest.raises(ExecutionTimeout):
        ctx.check()


def test_budget_counters():
    b = BudgetTracker(BudgetLimits(max_model_calls=2, max_tool_calls=1, max_tokens=10, max_retries=1))
    b.charge_model_call(4)
    b.charge_model_call(4)
    with pytest.raises(BudgetExhausted) as e:
        b.charge_model_call()
    assert e.value.resource == "model_calls"
    b.charge_tool_call()
    with pytest.raises(BudgetExhausted):
        b.charge_tool_call()
    with pytest.raises(BudgetExhausted):
        b.add_tokens(5)
    with pytest.raises(ValueError):
        BudgetLimits(max_tool_calls=0)


def test_policy():
    with pytest.raises(PolicyViolation):
        Policy().authorize_tool("anything")
    p = Policy(permissions=frozenset({"read"}), denied_tools=frozenset({"bad"}),
               allowed_tools=frozenset({"ok", "write", "bad", "appr"}))
    p.authorize_tool("ok")
    with pytest.raises(PolicyViolation):
        p.authorize_tool("bad")
    with pytest.raises(PolicyViolation):
        p.authorize_tool("write", frozenset({"write"}))
    with pytest.raises(PolicyViolation):
        p.authorize_tool("appr", requires_approval=True)
    with pytest.raises(PolicyViolation):
        p.authorize_tool("unlisted")


def test_error_info_hides_unknown_exception_text():
    assert "secret" not in ErrorInfo.from_exception(RuntimeError("secret path /x")).message
    assert status_for(BudgetExhausted("x", 1)) == Status.budget_exhausted
    assert ErrorInfo.from_exception(ExecutionTimeout("t")).retryable


def test_settings_from_env_validation():
    s = DracoSettings.from_env({"DRACO_MAX_TOOL_CALLS": "5", "DRACO_REDACT_KEYS": "a, b"})
    assert s.max_tool_calls == 5 and s.redact_keys == ("a", "b")
    with pytest.raises(ConfigurationError) as e:
        DracoSettings.from_env({"DRACO_MAX_TOOL_CALLS": "-1"})
    assert "DRACO_MAX_TOOL_CALLS" in str(e.value)


async def test_event_bus_failure_semantics():
    bus, order = EventBus(), []
    bus.subscribe("a.*", lambda e: (_ for _ in ()).throw(RuntimeError("telemetry")))
    bus.subscribe("a.b", lambda e: order.append("normal"))
    await bus.emit(Event("a.b"))  # telemetry failure is isolated
    assert order == ["normal"] and len(bus.handler_errors) == 1

    def deny(e):
        raise PolicyViolation("no")
    bus.subscribe("a.b", deny, critical=True)
    with pytest.raises(PolicyViolation):
        await bus.emit(Event("a.b"))
    assert order == ["normal"]  # critical handler ran first and blocked the rest
