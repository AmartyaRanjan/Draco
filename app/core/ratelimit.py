"""Token-bucket rate limiting keyed by authenticated user (or client address).

Redis holds the buckets so every worker process and replica shares one budget. If Redis is down the limiter
fails over to a per-process in-memory bucket instead of failing open completely or taking the API down; a short
circuit breaker stops every request paying a connect timeout while Redis is away.
"""
import logging
import ipaddress
import threading
import time
from dataclasses import dataclass

from fastapi import Depends, HTTPException, Request

from app.core import observability as obs
from app.core.http_settings import get_platform_settings

logger = logging.getLogger("rag.ratelimit")

# KEYS[1] bucket; ARGV: rate tokens/s, burst, cost. Time comes from Redis so replicas with skewed clocks agree.
# cost 0 is a peek: allowed only while at least one whole token is available.
_LUA = """
local t = redis.call('TIME')
local now = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)
local rate, burst, cost = tonumber(ARGV[1]), tonumber(ARGV[2]), tonumber(ARGV[3])
local d = redis.call('HMGET', KEYS[1], 'tokens', 'ts')
local tokens = tonumber(d[1]) or burst
local ts = tonumber(d[2]) or now
tokens = math.min(burst, tokens + math.max(0, now - ts) / 1000 * rate)
local need = math.max(cost, 1)
local ok = 0
if tokens >= need then tokens = tokens - cost; ok = 1 end
redis.call('HSET', KEYS[1], 'tokens', tostring(tokens), 'ts', now)
redis.call('PEXPIRE', KEYS[1], math.ceil(burst / rate * 1000) + 1000)
local retry = 0
if ok == 0 then retry = (need - tokens) / rate end
return {ok, tostring(tokens), tostring(retry)}
"""


@dataclass(frozen=True, slots=True)
class Decision:
    allowed: bool
    limit: int
    remaining: int
    retry_after_s: float


class MemoryLimiter:
    """Per-process buckets. Used when Redis is unavailable and in tests; not shared across workers."""

    def __init__(self, clock=time.monotonic):
        self._b: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()
        self._clock = clock

    async def hit(self, policy: str, key: str, per_minute: float, burst: float, cost: float = 1) -> Decision:
        rate, now, k = per_minute / 60.0, self._clock(), f"{policy}:{key}"
        with self._lock:
            tokens, ts = self._b.get(k, (burst, now))
            tokens = min(burst, tokens + max(0.0, now - ts) * rate)
            need = max(cost, 1)
            ok = tokens >= need
            if ok:
                tokens -= cost
            self._b[k] = (tokens, now)
            if len(self._b) > 50_000:  # bound memory: drop buckets that have fully refilled
                full = [x for x, (t, s) in self._b.items() if t + (now - s) * rate >= burst]
                for x in full:
                    del self._b[x]
        return Decision(ok, int(burst), int(tokens), 0.0 if ok else (need - tokens) / rate)


class RedisLimiter:
    BREAKER_S = 5.0
    # These policies protect expensive or security-sensitive operations. A
    # local fallback would multiply their budget across workers during a Redis
    # outage, so reject them until the shared limiter is healthy again.
    FAIL_CLOSED_POLICIES = frozenset({"authfail", "upload", "heavy"})

    def __init__(self, url: str, fallback: MemoryLimiter | None = None):
        import redis.asyncio as aredis
        self._r = aredis.Redis.from_url(url, socket_timeout=0.25, socket_connect_timeout=0.25)
        self._script = self._r.register_script(_LUA)
        self._fallback = fallback or MemoryLimiter()
        self._down_until = 0.0

    async def hit(self, policy: str, key: str, per_minute: float, burst: float, cost: float = 1) -> Decision:
        if time.monotonic() < self._down_until:
            if policy in self.FAIL_CLOSED_POLICIES:
                return Decision(False, int(burst), 0, self.BREAKER_S)
            return await self._fallback.hit(policy, key, per_minute, burst, cost)
        try:
            ok, tokens, retry = await self._script(keys=[f"rl:{policy}:{key}"], args=[per_minute / 60.0, burst, cost])
            return Decision(bool(ok), int(burst), int(float(tokens)), float(retry))
        except Exception as exc:  # redis.RedisError, timeouts, connection refused
            self._down_until = time.monotonic() + self.BREAKER_S
            obs.RATE_LIMIT_BACKEND_ERRORS.inc()
            logger.warning("ratelimit_backend_unavailable", extra={"error": type(exc).__name__})
            if policy in self.FAIL_CLOSED_POLICIES:
                return Decision(False, int(burst), 0, self.BREAKER_S)
            return await self._fallback.hit(policy, key, per_minute, burst, cost)


_limiter = None


def get_limiter():
    global _limiter
    if _limiter is None:
        _limiter = RedisLimiter(get_platform_settings().redis_url)
    return _limiter


def set_limiter(limiter) -> None:
    """Swap the backend (tests, or a deployment that wants memory-only)."""
    global _limiter
    _limiter = limiter


def client_ip_from(headers, peer: str | None) -> str:
    """Peer address, or the address our own proxies saw. X-Forwarded-For is attacker-controlled, so it is only
    read when TRUSTED_PROXY_HOPS says how many entries from the right were appended by our infrastructure."""
    settings = get_platform_settings()
    hops = settings.trusted_proxy_hops
    xff = headers.get("x-forwarded-for")
    peer_is_trusted = False
    if peer and settings.trusted_proxy_cidrs:
        try:
            peer_address = ipaddress.ip_address(peer)
            peer_is_trusted = any(
                peer_address in (
                    network if hasattr(network, "subnet_of")
                    else ipaddress.ip_network(network, strict=False)
                )
                for network in settings.trusted_proxy_cidrs
            )
        except (TypeError, ValueError):
            peer_is_trusted = False
    if hops > 0 and peer_is_trusted and xff:
        parts = [p.strip() for p in xff.split(",") if p.strip()]
        if len(parts) >= hops:
            return parts[-hops]
    return peer or "unknown"


def client_ip(request: Request) -> str:
    return client_ip_from(request.headers, request.client.host if request.client else None)


async def check(policy: str, key: str, cost: float = 1) -> Decision:
    s = get_platform_settings()
    if not s.rate_limit_enabled:
        return Decision(True, 0, 0, 0.0)
    per_min, burst = s.policies[policy]
    return await get_limiter().hit(policy, key, per_min, burst, cost)


def reject(policy: str, d: Decision, what: str = "Rate limit exceeded") -> HTTPException:
    obs.RATE_LIMIT_REJECTIONS.labels(policy).inc()
    wait = max(1, int(d.retry_after_s + 0.999))
    return HTTPException(429, f"{what}. Retry in {wait}s.",
                         headers={"Retry-After": str(wait), "X-RateLimit-Limit": str(d.limit), "X-RateLimit-Remaining": "0"})


def limited(policy: str, user_dep):
    """Dependency factory: authenticate with `user_dep`, then spend one token from `policy` for that user."""
    async def dep(request: Request, user: str = Depends(user_dep)) -> str:
        d = await check(policy, f"user:{user}")
        request.state.ratelimit = d
        if not d.allowed:
            raise reject(policy, d)
        return user
    return dep
