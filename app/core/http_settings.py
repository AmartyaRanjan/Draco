"""Process-level settings for the HTTP layer: environment, limits, CORS, proxies. Read once, validated at startup."""
import os
import ipaddress
from dataclasses import dataclass
from functools import lru_cache


def _b(name, default="false"):
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes")


def _policy(name, rate, burst):
    """'RATE_LIMIT_<NAME>=<per-minute>,<burst>' -> (per_minute, burst)."""
    raw = os.getenv(f"RATE_LIMIT_{name}", f"{rate},{burst}")
    try:
        per_min, b = (float(x) for x in raw.split(","))
    except ValueError as exc:
        raise ValueError(f"RATE_LIMIT_{name} must look like '30,10' (per-minute rate, burst); got {raw!r}") from exc
    if per_min <= 0 or b < 1:
        raise ValueError(f"RATE_LIMIT_{name}: rate must be > 0 and burst >= 1.")
    return per_min, b


@dataclass(frozen=True, slots=True)
class PlatformSettings:
    env: str                       # development | production
    log_level: str
    log_json: bool
    cors_origins: tuple[str, ...]
    max_body_bytes: int            # every route except uploads
    max_upload_bytes: int
    trusted_proxy_hops: int        # how many X-Forwarded-For entries (from the right) are our own proxies
    trusted_proxy_cidrs: tuple     # network ranges allowed to send forwarded headers
    rate_limit_enabled: bool
    redis_url: str
    policies: dict                 # name -> (per_minute, burst)
    llm_max_concurrent: int
    llm_queue_wait_s: float
    metrics_token: str | None
    ready_cache_s: float

    @property
    def production(self) -> bool:
        return self.env == "production"

    def validate(self, auth_disabled: bool = False) -> "PlatformSettings":
        if self.env not in ("development", "production"):
            raise ValueError("APP_ENV must be 'development' or 'production'.")
        if self.max_body_bytes < 1024 or self.max_upload_bytes < self.max_body_bytes:
            raise ValueError("Body limits: MAX_BODY_BYTES >= 1024 and MAX_UPLOAD_BYTES >= MAX_BODY_BYTES.")
        if self.llm_max_concurrent < 1 or self.llm_queue_wait_s < 0:
            raise ValueError("LLM_MAX_CONCURRENT must be >= 1 and LLM_QUEUE_WAIT_S >= 0.")
        if self.trusted_proxy_hops < 0:
            raise ValueError("TRUSTED_PROXY_HOPS must be >= 0.")
        if self.trusted_proxy_hops and not self.trusted_proxy_cidrs:
            raise ValueError("TRUSTED_PROXY_CIDRS is required when TRUSTED_PROXY_HOPS > 0.")
        if self.production:
            if auth_disabled:
                raise ValueError("APP_AUTH_DISABLED=true is not allowed when APP_ENV=production.")
            if "*" in self.cors_origins:
                raise ValueError("CORS_ORIGINS must list explicit origins when APP_ENV=production.")
            if not self.rate_limit_enabled:
                raise ValueError("RATE_LIMIT_ENABLED=false is not allowed when APP_ENV=production.")
        return self


@lru_cache
def get_platform_settings() -> PlatformSettings:
    trusted_proxy_cidrs = tuple(
        ipaddress.ip_network(raw.strip(), strict=False)
        for raw in os.getenv("TRUSTED_PROXY_CIDRS", "").split(",")
        if raw.strip()
    )
    return PlatformSettings(
        env=os.getenv("APP_ENV", "development").strip().lower(),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        log_json=_b("LOG_JSON", "true"),
        cors_origins=tuple(o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()),
        max_body_bytes=int(os.getenv("MAX_BODY_BYTES", str(1024 * 1024))),
        max_upload_bytes=int(os.getenv("MAX_UPLOAD_BYTES", str(101 * 1024 * 1024))),
        trusted_proxy_hops=int(os.getenv("TRUSTED_PROXY_HOPS", "0")),
        trusted_proxy_cidrs=trusted_proxy_cidrs,
        rate_limit_enabled=_b("RATE_LIMIT_ENABLED", "true"),
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        policies={
            # LLM-backed routes: the GPU serves ~2 sequences at ~20 tok/s, so these are deliberately low.
            "heavy": _policy("HEAVY", 20, 6),
            "upload": _policy("UPLOAD", 10, 5),
            "read": _policy("READ", 120, 30),
            # Per client address, before auth; only a backstop against floods.
            "ip": _policy("IP", 300, 60),
            # Failed API-key attempts per address: brute-force guard.
            "authfail": _policy("AUTHFAIL", 10, 5),
        },
        llm_max_concurrent=int(os.getenv("LLM_MAX_CONCURRENT", "4")),
        llm_queue_wait_s=float(os.getenv("LLM_QUEUE_WAIT_S", "10")),
        metrics_token=os.getenv("METRICS_TOKEN") or None,
        ready_cache_s=float(os.getenv("READY_CACHE_S", "5")),
    )
