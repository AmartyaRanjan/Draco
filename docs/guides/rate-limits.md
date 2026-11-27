# Rate-limit tenants and users

[Budgets](../concepts/budgets.md) bound what **one run** may use. Rate limits and concurrency caps bound **how many runs start**, so one noisy tenant cannot starve the others. This guide covers the three controls and how to share them across processes.

All of them apply to **top-level runs** only (not nested child runs), and all of them **reject** a run that is over the limit rather than queuing it. A rejected run ends as `budget_exhausted` with a `retryable` error, and nothing of your workload has run.

## The three controls

| Control | Setting | Env variable | Bounds |
|---|---|---|---|
| Tenant rate | `tenant_rate_per_minute`, `tenant_rate_burst` | `DRACO_TENANT_RATE_PER_MINUTE`, `DRACO_TENANT_RATE_BURST` | Runs started per minute by one tenant |
| User rate | `user_rate_per_minute`, `user_rate_burst` | `DRACO_USER_RATE_PER_MINUTE`, `DRACO_USER_RATE_BURST` | Runs started per minute by one user within a tenant |
| Tenant concurrency | `max_concurrency_per_tenant` | `DRACO_MAX_CONCURRENCY_PER_TENANT` | Runs of one tenant in flight at the same time |

The rates are **token buckets**: the bucket holds up to `burst` runs, and refills at `per_minute / 60` per second. A burst lets a tenant fire a few runs at once after being idle; the rate caps the sustained pace. If you give no burst, it defaults to the per-minute rate (at least 1).

Limits are keyed by the run's **identity**. A run with no tenant skips the tenant limits, and a run with no user skips the user limit, so combine rate limits with `require_identity` ([security model](../concepts/security.md)).

## Example

<!-- doctest -->
```python
import asyncio
from draco import DracoSettings, Harness, Identity

async def work(ctx, seconds):
    await asyncio.sleep(seconds)
    return "ok"

def describe(r):
    if r.ok:
        return "completed"
    return f"{r.status.value} ({r.error.details['resource']}, retryable={r.error.retryable})"

async def main():
    acme, globex = Identity("alice", "acme"), Identity("gus", "globex")

    # 1. Tenant rate: a burst of 2, then refused. Another tenant has its own bucket.
    h = Harness(DracoSettings(tenant_rate_per_minute=60, tenant_rate_burst=2))
    h.register("work", work)
    for who, identity in (("acme", acme), ("acme", acme), ("acme", acme), ("globex", globex)):
        print("tenant rate:", who, describe(await h.run("work", 0, identity=identity)))

    # 2. User rate: each user has a bucket of their own.
    h = Harness(DracoSettings(user_rate_per_minute=60, user_rate_burst=1))
    h.register("work", work)
    for user in ("alice", "alice", "bob"):
        print("user rate:", user, describe(await h.run("work", 0, identity=Identity(user, "acme"))))

    # 3. Concurrency: one run per tenant at a time. The second is rejected, not queued.
    h = Harness(DracoSettings(max_concurrency_per_tenant=1))
    h.register("work", work)
    first = asyncio.create_task(h.run("work", 0.2, identity=acme))
    await asyncio.sleep(0.05)
    print("concurrency: second acme run", describe(await h.run("work", 0, identity=acme)))
    print("concurrency: globex run", describe(await h.run("work", 0, identity=globex)))
    print("concurrency: first acme run", describe(await first))

asyncio.run(main())
```

<!-- output -->
```text
tenant rate: acme completed
tenant rate: acme completed
tenant rate: acme budget_exhausted (tenant_rate, retryable=True)
tenant rate: globex completed
user rate: alice completed
user rate: alice budget_exhausted (user_rate, retryable=True)
user rate: bob completed
concurrency: second acme run budget_exhausted (tenant_concurrency, retryable=False)
concurrency: globex run completed
concurrency: first acme run completed
```

A rate-limit refusal carries `retry_after_s` in `error.details`, the seconds until a slot frees up, so an HTTP service can answer `429` with a `Retry-After` header. A concurrency refusal is not marked retryable, because there is no useful wait time to report.

## Share the limit across processes

The built-in `InMemoryRateLimiter` keeps its buckets **in the memory of one process**. If you run four copies of your service, a tenant effectively gets four times the limit. Concurrency caps are per process for the same reason.

To enforce a limit across processes, implement the `RateLimiter` protocol on top of a shared store such as Redis and pass it to the harness. The protocol is one method:

```python
from draco.policies.ratelimit import RateLimit

class SharedLimiter:
    async def acquire(self, key: str, limit: RateLimit) -> float:
        """Take one admission for `key`. Return 0.0 if admitted, otherwise the seconds until one
        would be available. Consume nothing on refusal."""
        ...

h = Harness(settings, rate_limiter=SharedLimiter())
```

`RateLimit` carries `per_minute` and `burst`. Keys look like `tenant\x1f<tenant_id>` and `user\x1f<tenant_id>\x1f<user_id>`, so one limiter serves both kinds of bucket. The refill and the take must be **one atomic step** in your store, or two processes can both admit the last slot; in Redis that means a Lua script or an equivalent atomic operation.

## See also

- [Budgets and limits](../concepts/budgets.md)
- [Configuration reference](../reference/configuration.md)
