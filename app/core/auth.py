"""API-key grants (user, scopes, optional expiry), route scope enforcement and the auth audit log.

`APP_API_KEYS` is a comma-separated list of entries:

    key:user                                   all non-admin scopes, never expires (the original format)
    key:user:read+chat                         only these scopes
    key:user:agent:expires=2027-01-31          scopes + expiry
    key:user:expires=2027-01-31                all non-admin scopes, with expiry

Scopes are joined with `+` (`,` separates entries). Known scopes: read, chat, agent, upload, admin; `admin`
implies every scope. Expiry is a UTC date: the key stops working at 00:00 UTC on that date.

Enforcement is a router-level dependency (`require_scope`) applied in `app/main.py`. It runs before the route's own
authentication dependency and handles only keys that are present in `APP_API_KEYS`: expired keys get 401, keys
without the route's scope get 403. Unknown or missing keys are left to the route's authentication dependency,
which owns the 401 response and the brute-force lockout, so the lockout is charged exactly once.

Audit events go to the `rag.audit` logger with `event` in {auth_success, auth_failure, auth_expired,
auth_scope_denied}. They never contain a key; `key_id` is a short SHA-256 fingerprint of a *configured* key, so
operators can tell which key was used without the log being a credential store.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import logging
import os
from dataclasses import dataclass
from functools import lru_cache

from fastapi import HTTPException, Request

logger = logging.getLogger("rag.audit")

SCOPES = frozenset({"read", "chat", "agent", "upload", "admin"})
DEFAULT_SCOPES = frozenset(SCOPES - {"admin"})  # what a legacy `key:user` entry may do (unchanged behaviour)


@dataclass(frozen=True, slots=True)
class ApiKey:
    user: str
    scopes: frozenset[str] = DEFAULT_SCOPES
    expires: dt.date | None = None  # first UTC date on which the key is no longer valid

    def expired(self, now: dt.datetime | None = None) -> bool:
        if self.expires is None:
            return False
        now = now or dt.datetime.now(dt.UTC)
        return now.date() >= self.expires

    def allows(self, scope: str) -> bool:
        return "admin" in self.scopes or scope in self.scopes


def key_id(key: str) -> str:
    """Non-reversible short fingerprint used in logs instead of the key."""
    return hashlib.sha256(key.encode()).hexdigest()[:12]


def parse_api_keys(raw: str) -> dict[str, ApiKey]:
    """Parse APP_API_KEYS. Raises ValueError on malformed entries; messages never include the key itself."""
    out: dict[str, ApiKey] = {}
    for n, entry in enumerate(filter(None, (p.strip() for p in raw.split(","))), 1):
        parts = entry.split(":")
        if len(parts) < 2 or not parts[0] or not parts[1]:
            raise ValueError(f"APP_API_KEYS entry #{n}: expected 'key:user[:scopes][:expires=YYYY-MM-DD]'.")
        key, user, rest = parts[0], parts[1], parts[2:]
        scopes, expires = None, None
        for opt in rest:
            if opt.startswith("expires="):
                if expires is not None:
                    raise ValueError(f"APP_API_KEYS entry #{n} (user {user}): expires given twice.")
                try:
                    expires = dt.date.fromisoformat(opt.removeprefix("expires="))
                except ValueError:
                    raise ValueError(f"APP_API_KEYS entry #{n} (user {user}): expires must be YYYY-MM-DD.") from None
            else:
                if scopes is not None:
                    raise ValueError(f"APP_API_KEYS entry #{n} (user {user}): scopes given twice.")
                scopes = frozenset(s.strip() for s in opt.split("+") if s.strip())
                unknown = scopes - SCOPES
                if not scopes or unknown:
                    raise ValueError(f"APP_API_KEYS entry #{n} (user {user}): unknown scope(s) {sorted(unknown)}; "
                                     f"allowed: {sorted(SCOPES)}.")
        if key in out:
            raise ValueError(f"APP_API_KEYS entry #{n} (user {user}): duplicate key.")
        out[key] = ApiKey(user, scopes if scopes is not None else DEFAULT_SCOPES, expires)
    return out


@lru_cache
def get_api_keys() -> dict[str, ApiKey]:
    return parse_api_keys(os.getenv("APP_API_KEYS", ""))


def _auth_disabled() -> bool:
    return os.getenv("APP_AUTH_DISABLED", "false").lower() == "true"


def lookup(supplied: str, keys: dict[str, ApiKey]) -> tuple[str, ApiKey] | None:
    """Constant-time comparison against every configured key (no early exit on a match)."""
    found = None
    for key, grant in keys.items():
        if hmac.compare_digest(key.encode(), supplied.encode()):
            found = (key, grant)
    return found


def audit(event: str, request: Request, **fields) -> None:
    level = logging.INFO if event == "auth_success" else logging.WARNING
    client = request.client.host if request.client else None
    logger.log(level, event, extra={"event": event, "path": request.url.path, "method": request.method,
                                    "client_ip": getattr(request.state, "client_ip", None) or client, **fields})


def require_scope(default: str, by_method: dict[str, str] | None = None):
    """Router dependency: the request's API key must hold `by_method[METHOD]` (or `default`)."""
    for s in (default, *(by_method or {}).values()):
        if s not in SCOPES:
            raise ValueError(f"unknown scope {s!r}")

    async def guard(request: Request) -> None:
        if _auth_disabled():
            return
        scope = (by_method or {}).get(request.method, default)
        supplied = request.headers.get("x-api-key")
        if not supplied:
            audit("auth_failure", request, reason="missing_key", scope=scope)
            return  # the route's auth dependency answers 401 and charges the lockout
        # Do not reveal "expired" / "forbidden" to a throttled address (lockout, or limiter fail-closed): route auth answers 429.
        from app.core import ratelimit
        blocked = await ratelimit.check("authfail", ratelimit.client_ip(request), cost=0)
        if not blocked.allowed:
            audit("auth_failure", request, reason="auth_throttled", scope=scope)
            return
        hit = lookup(supplied, get_api_keys())
        if hit is None:
            audit("auth_failure", request, reason="unknown_key", scope=scope)
            return
        key, grant = hit
        if grant.expired():
            audit("auth_expired", request, user=grant.user, key_id=key_id(key), scope=scope,
                  expired_on=grant.expires.isoformat() if grant.expires else None)
            await ratelimit.check("authfail", ratelimit.client_ip(request))  # an expired key is a failed attempt
            raise HTTPException(401, "API key expired")
        if not grant.allows(scope):
            audit("auth_scope_denied", request, user=grant.user, key_id=key_id(key), scope=scope,
                  granted=sorted(grant.scopes))
            raise HTTPException(403, f"API key lacks the '{scope}' scope")
        request.state.auth_scopes = grant.scopes
        audit("auth_success", request, user=grant.user, key_id=key_id(key), scope=scope)

    return guard
