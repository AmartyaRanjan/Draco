"""Secret values that do not leak through repr, str, logging or JSON encoding of events."""
from __future__ import annotations

import os
from typing import Protocol, runtime_checkable

from draco.core.errors import ConfigurationError


class Secret:
    """Wraps a credential. Reading it requires an explicit `.reveal()`."""
    __slots__ = ("_v",)

    def __init__(self, value: str):
        self._v = value

    def reveal(self) -> str:
        return self._v

    def __repr__(self) -> str:
        return "Secret(<redacted>)"

    __str__ = __repr__

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Secret) and self._v == other._v

    def __hash__(self) -> int:
        return hash(self._v)

    def __reduce__(self):  # pickling/copying would persist the plaintext
        raise TypeError("Secret cannot be pickled")


@runtime_checkable
class SecretProvider(Protocol):
    def get(self, name: str) -> Secret: ...


class EnvSecretProvider:
    def __init__(self, prefix: str = ""):
        self.prefix = prefix

    def get(self, name: str) -> Secret:
        v = os.environ.get(self.prefix + name)
        if not v:
            raise ConfigurationError(f"secret '{name}' is not configured")
        return Secret(v)
