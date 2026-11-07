"""Time-window rate limits: token bucket semantics, harness admission per tenant and per user."""
import pytest

from draco.core import ConfigurationError, DracoSettings, Identity, Status
from draco.harness import Harness
from draco.policies import InMemoryRateLimiter, RateLimit, RateLimiter, limits_from_settings


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def test_token_bucket_burst_refill_and_retry_after():
    clock = Clock()
    rl = InMemoryRateLimiter(clock=clock)
    lim = RateLimit(per_minute=60, burst=3)  # one per second, three at once
    assert [rl.try_acquire("k", lim) for _ in range(3)] == [0.0, 0.0, 0.0]
    wait = rl.try_acquire("k", lim)
    assert wait == pytest.approx(1.0)
    assert rl.try_acquire("other", lim) == 0.0  # keys are independent
    clock.t += 0.5
    assert rl.try_acquire("k", lim) == pytest.approx(0.5)  # a refusal consumes nothing
    clock.t += 0.5
    assert rl.try_acquire("k", lim) == 0.0
    clock.t += 3600
    assert [rl.try_acquire("k", lim) for _ in range(4)][-1] > 0  # refill is capped at the burst


def test_idle_buckets_are_evicted():
    clock = Clock()
    rl = InMemoryRateLimiter(clock=clock, max_keys=10)
    lim = RateLimit(per_minute=600, burst=1)
    for i in range(50):
        rl.try_acquire(f"k{i}", lim)
        clock.t += 1  # every earlier bucket refilled
    assert len(rl._buckets) <= 11


def test_settings_from_env_and_defaults():
    st = DracoSettings.from_env({"DRACO_TENANT_RATE_PER_MINUTE": "30", "DRACO_USER_RATE_PER_MINUTE": "0.5",
                                 "DRACO_USER_RATE_BURST": "2"})
    t, u = limits_from_settings(st)
    assert t == RateLimit(30.0, 30) and u == RateLimit(0.5, 2)
    assert limits_from_settings(DracoSettings()) == (None, None)
    with pytest.raises(ConfigurationError):
        DracoSettings.from_env({"DRACO_TENANT_RATE_PER_MINUTE": "-1"})


def _harness(clock, **settings):
    h = Harness(DracoSettings(**settings), rate_limiter=InMemoryRateLimiter(clock=clock))

    async def w(ctx, x):
        if x == "child":
            return "c"
        return (await h.run("w", "child", parent=ctx)).output if x == "spawn" else "ok"
    h.register("w", w)
    return h


async def test_tenant_rate_limit_rejects_excess_root_runs_with_a_distinct_reason():
    clock = Clock()
    h = _harness(clock, tenant_rate_per_minute=60, tenant_rate_burst=2)
    acme, beta = Identity("alice", "acme"), Identity("bob", "beta")
    assert (await h.run("w", "spawn", identity=acme)).output == "c"  # children are not rate limited
    assert (await h.run("w", 1, identity=Identity("carol", "acme"))).ok
    r = await h.run("w", 1, identity=acme)
    assert r.status == Status.budget_exhausted
    assert r.error.details["resource"] == "tenant_rate" and r.error.retryable
    assert 0 < r.error.details["retry_after_s"] <= 1.0
    rec = await h.store.get_execution(r.execution_id)
    assert rec.status == "budget_exhausted"  # recorded like any other outcome
    assert (await h.run("w", 1, identity=beta)).ok  # another tenant has its own bucket
    assert (await h.run("w", 1)).ok  # no tenant: tenant limit does not apply
    clock.t += 1.0
    assert (await h.run("w", 1, identity=acme)).ok


async def test_user_rate_limit_is_per_tenant_and_user():
    clock = Clock()
    h = _harness(clock, user_rate_per_minute=6, user_rate_burst=1)
    a1, a2 = Identity("alice", "acme"), Identity("alice", "beta")
    assert (await h.run("w", 1, identity=a1)).ok
    r = await h.run("w", 1, identity=a1)
    assert r.status == Status.budget_exhausted and r.error.details["resource"] == "user_rate"
    assert r.error.details["retry_after_s"] == pytest.approx(10.0)
    assert (await h.run("w", 1, identity=a2)).ok  # same subject in another tenant: separate bucket
    assert (await h.run("w", 1, identity=Identity("bob", "acme"))).ok
    assert (await h.run("w", 1, user_id="dave")).ok and not (await h.run("w", 1, user_id="dave")).ok  # untrusted ids too


async def test_idempotent_replays_do_not_consume_rate():
    clock = Clock()
    h = _harness(clock, tenant_rate_per_minute=1, tenant_rate_burst=1)
    acme = Identity("alice", "acme")
    first = await h.run("w", 1, identity=acme, idempotency_key="k")
    again = await h.run("w", 1, identity=acme, idempotency_key="k")
    assert first.ok and again.replayed and again.ok
    assert (await h.run("w", 1, identity=acme)).status == Status.budget_exhausted


async def test_rate_limiter_is_pluggable():
    seen = []

    class Deny:
        async def acquire(self, key, limit):
            seen.append((key, limit))
            return 5.0
    assert isinstance(Deny(), RateLimiter)
    h = Harness(DracoSettings(tenant_rate_per_minute=100), rate_limiter=Deny())
    h.register("w", lambda ctx, x: _ok())
    r = await h.run("w", identity=Identity("alice", "acme"))
    assert r.status == Status.budget_exhausted and r.error.details["retry_after_s"] == 5.0
    assert seen == [("tenant\x1facme", RateLimit(100.0, 100))]


async def test_no_limiter_without_configuration():
    assert Harness(DracoSettings()).rate_limiter is None


async def _ok():
    return "ok"
