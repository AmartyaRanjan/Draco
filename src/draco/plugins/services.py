"""Explicit service container. Names are unique; protected services cannot be replaced by plugins."""
from __future__ import annotations

from typing import Any, TypeVar

from draco.core.errors import DuplicateRegistration, MissingDependency, PolicyViolation

T = TypeVar("T")
CORE_OWNER = "core"


class ServiceRegistry:
    def __init__(self) -> None:
        self._services: dict[str, Any] = {}
        self._owners: dict[str, str] = {}
        self._protected: set[str] = set()

    def provide(self, name: str, service: Any, *, owner: str = CORE_OWNER, protected: bool = False) -> None:
        if name in self._services:
            raise DuplicateRegistration(f"service '{name}' already provided by '{self._owners[name]}'")
        self._services[name] = service
        self._owners[name] = owner
        if protected:
            self._protected.add(name)

    def replace(self, name: str, service: Any, *, owner: str) -> Any:
        """Swap an implementation. Only the owner (or core) may replace, never a protected service
        from a plugin. Returns the previous implementation."""
        if name not in self._services:
            raise MissingDependency(f"cannot replace unknown service '{name}'")
        if name in self._protected and owner != CORE_OWNER:
            raise PolicyViolation(f"service '{name}' is protected and cannot be replaced by '{owner}'")
        if owner not in (CORE_OWNER, self._owners[name]):
            raise PolicyViolation(f"service '{name}' is owned by '{self._owners[name]}', not '{owner}'")
        prev, self._services[name] = self._services[name], service
        return prev

    def get(self, name: str, expected: type[T] | None = None) -> T:
        try:
            svc = self._services[name]
        except KeyError:
            raise MissingDependency(f"service '{name}' is not registered") from None
        if expected is not None and not isinstance(svc, expected):
            raise MissingDependency(f"service '{name}' is not a {expected.__name__}")
        return svc

    def has(self, name: str) -> bool:
        return name in self._services

    def owner_of(self, name: str) -> str | None:
        return self._owners.get(name)

    def remove_owned_by(self, owner: str) -> list[str]:
        gone = [n for n, o in self._owners.items() if o == owner and n not in self._protected]
        for n in gone:
            del self._services[n], self._owners[n]
        return gone

    def names(self) -> list[str]:
        return sorted(self._services)
