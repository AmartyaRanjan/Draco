"""`ExecutionResult.raise_for_status` must rebuild the original typed error, not a bare DracoError."""
import pytest

from draco import Harness, Policy
from draco.agents import AgentRegistry, Router, RouteRequest
from draco.core.errors import (
    BudgetExhausted,
    ContractError,
    DracoError,
    ExecutionTimeout,
    PolicyViolation,
    ProviderError,
)
from draco.core.results import ErrorInfo, ExecutionResult, Status


def failed(exc: BaseException) -> ExecutionResult:
    return ExecutionResult(Status.failed, "e1", error=ErrorInfo.from_exception(exc))


@pytest.mark.parametrize("exc", [
    PolicyViolation("no access", details={"action": "x"}),
    ContractError("bad input", details={"reason": "not_found"}),
    ProviderError("upstream 503", retryable=True),
    ProviderError("upstream 400", retryable=False),
])
def test_failure_keeps_class_code_retryable_and_details(exc):
    with pytest.raises(type(exc)) as caught:
        failed(exc).raise_for_status()
    got = caught.value
    assert (got.code, got.retryable, got.details, got.message) == (exc.code, exc.retryable, exc.details, exc.message)


def test_unknown_category_keeps_its_code():
    result = ExecutionResult(Status.failed, "e", error=ErrorInfo("Whatever", "internal", "boom"))
    with pytest.raises(DracoError) as caught:
        result.raise_for_status()
    assert caught.value.code == "internal" and type(caught.value) is DracoError


def test_budget_and_timeout_keep_their_details():
    budget = ExecutionResult(Status.budget_exhausted, "e", error=ErrorInfo.from_exception(BudgetExhausted("user_rate", 3)))
    with pytest.raises(BudgetExhausted) as b:
        budget.raise_for_status()
    assert b.value.details == {"resource": "user_rate", "limit": 3}
    timeout = ExecutionResult(Status.timeout, "e", error=ErrorInfo.from_exception(ExecutionTimeout("slow")))
    with pytest.raises(ExecutionTimeout):
        timeout.raise_for_status()


async def test_router_propagates_a_child_agents_error_type():
    h = Harness()
    agents = AgentRegistry(h)

    async def denied(ctx, _):
        raise PolicyViolation("document access denied")

    async def flaky(ctx, _):
        raise ProviderError("model down", retryable=True)

    agents.register("denied", denied)
    agents.register("flaky", flaky)
    router = Router(h, agents)
    policy = Policy()
    first = await router.run(RouteRequest("x", agent="denied"), policy=policy)
    assert first.status == Status.failed and first.error.code == "policy"
    assert first.error.message == "document access denied"
    second = await router.run(RouteRequest("x", agent="flaky"))
    assert second.error.code == "provider" and second.error.retryable is True
