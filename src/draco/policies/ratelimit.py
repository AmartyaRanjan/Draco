"""Time-window rate limits for root executions.

`RateLimiter` is the pluggable contract (implement it over Redis or another shared store for several
processes); `InMemoryRateLimiter` is a per-process token bucket. The harness asks it before admitting
a root execution, per tenant and optionally per (tenant, user); a refusal ends the execution
`budget_exhausted` with resource `tenant_rate` or `user_rate` and a `retry_after_s` detail."""
from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from draco.core.config import DracoSettings


@dataclass(frozen=True)
class RateLimit:
    per_minute: float  # sustained admission rate
    burst: int  # bucket capacity: admissions allowed at once after an idle period

    @property
    def per_second(self) -> float:
        return self.per_minute / 60.0


@runtime_checkable
class RateLimiter(Protocol):
    async def acquire(self, key: str, limit: RateLimit) -> float:
        """Take one admission for `key`. Returns 0.0 when admitted, otherwise the seconds until one
        would be available (nothing is consumed on refusal)."""
        ...


class InMemoryRateLimiter:
    """Token bucket per key, in process memory. Idle full buckets are evicted, so memory is bounded
    by the number of keys active within roughly one refill period (capped at `max_keys`)."""

    def __init__(self, *, clock: Callable[[], float] = time.monotonic, max_keys: int = 100_000):
        self._clock, self._max = clock, max_keys
        self._buckets: dict[str, tuple[float, float]] = {}  # key -> (tokens, updated_at)
        self._lock = threading.Lock()

    async def acquire(self, key: str, limit: RateLimit) -> float:
        return self.try_acquire(key, limit)

    def try_acquire(self, key: str, limit: RateLimit) -> float:
        now = self._clock()
        with self._lock:
            tokens, at = self._buckets.get(key, (float(limit.burst), now))
            tokens = min(float(limit.burst), tokens + (now - at) * limit.per_second)
            if tokens >= 1.0:
                self._buckets[key] = (tokens - 1.0, now)
                if len(self._buckets) > self._max:
                    self._evict(now, limit)
                return 0.0
            self._buckets[key] = (tokens, now)
            return (1.0 - tokens) / limit.per_second

    def _evict(self, now: float, limit: RateLimit) -> None:
        full = [k for k, (t, at) in self._buckets.items() if t + (now - at) * limit.per_second >= limit.burst]
        for k in full:
            del self._buckets[k]
        while len(self._buckets) > self._max:  # still too many: drop the least recently updated
            del self._buckets[min(self._buckets, key=lambda k: self._buckets[k][1])]


def limits_from_settings(settings: DracoSettings) -> tuple[RateLimit | None, RateLimit | None]:
    """(tenant limit, user limit) from `tenant_rate_per_minute`/`user_rate_per_minute` and their bursts."""
    def make(rate: float | None, burst: int | None) -> RateLimit | None:
        if rate is None:
            return None
        return RateLimit(rate, burst if burst is not None else max(1, int(rate)))
    return (make(settings.tenant_rate_per_minute, settings.tenant_rate_burst),
            make(settings.user_rate_per_minute, settings.user_rate_burst))
