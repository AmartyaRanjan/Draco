"""Trusted caller identity. Built by application code from a verified credential, never from
model output or request payload fields."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from draco.core.errors import PolicyViolation


@dataclass(frozen=True)
class Identity:
    """Who is acting. `tenant_id` is mandatory so that tenant isolation cannot be skipped by omission."""
    subject: str
    tenant_id: str
    roles: frozenset[str] = frozenset()
    claims: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        for name in ("subject", "tenant_id"):
            v = getattr(self, name)
            if not isinstance(v, str) or not v.strip() or v.strip().lower() in _PLACEHOLDERS:
                raise PolicyViolation(f"identity {name} must be a real, non-placeholder value")
        object.__setattr__(self, "roles", frozenset(self.roles))


_PLACEHOLDERS = {"anonymous", "none", "null", "unknown", "guest", "*"}


@runtime_checkable
class Authenticator(Protocol):
    """Application integration point: turn a credential (token, session, mTLS subject ...) into an
    Identity or raise PolicyViolation. Draco ships no authentication provider."""

    async def authenticate(self, credential: Any) -> Identity: ...
