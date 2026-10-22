"""Keep credentials and bulky payloads out of traces and logs."""
from __future__ import annotations

import hashlib
import re
from typing import Any

_TOKEN_RE = re.compile(r"(sk-[A-Za-z0-9_-]{8,}|Bearer\s+[A-Za-z0-9._~+/=-]{8,})")


class Redactor:
    def __init__(self, keys: tuple[str, ...] = ("key", "token", "password", "secret", "authorization"),
                 max_chars: int = 2000, max_items: int = 50, pii_keys: tuple[str, ...] = ("user_id", "tenant_id", "subject", "resource_tenant")):
        self.keys, self.max_chars, self.max_items = tuple(k.lower() for k in keys), max_chars, max_items
        self.pii_keys = tuple(k.lower() for k in pii_keys)

    def redact(self, value: Any, _key: str = "", _depth: int = 0) -> Any:
        if _key and any(k in _key.lower() for k in self.keys):
            return "<redacted>"
        if _key and _key.lower() in self.pii_keys and value is not None:
            # stable pseudonym: traces can be correlated per user without storing the raw identifier
            return "id:" + hashlib.sha256(str(value).encode()).hexdigest()[:12]
        if _depth > 6:
            return "<truncated>"
        if isinstance(value, str):
            v = _TOKEN_RE.sub("<redacted>", value)
            return v if len(v) <= self.max_chars else v[: self.max_chars] + f"...<{len(v) - self.max_chars} more chars>"
        if isinstance(value, dict):
            return {str(k): self.redact(v, str(k), _depth + 1) for k, v in list(value.items())[: self.max_items]}
        if isinstance(value, (list, tuple, set, frozenset)):
            items = [self.redact(v, "", _depth + 1) for v in list(value)[: self.max_items]]
            return items
        if isinstance(value, (int, float, bool)) or value is None:
            return value
        return f"<{type(value).__name__}>"
