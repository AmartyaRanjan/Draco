"""One event bus for lifecycle, hook and audit events.

Handler failure semantics are explicit: a normal handler (telemetry) never breaks the run, its
error is recorded; a `critical` handler (policy) runs first and its exception propagates.
"""
from __future__ import annotations

import inspect
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("draco.events")


@dataclass(frozen=True)
class Event:
    name: str
    execution_id: str | None = None
    parent_execution_id: str | None = None
    correlation_id: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)


Handler = Callable[[Event], Awaitable[None] | None]


@dataclass
class _Sub:
    pattern: str
    handler: Handler
    critical: bool


class EventBus:
    def __init__(self) -> None:
        self._subs: list[_Sub] = []
        self.handler_errors: list[tuple[str, str, Exception]] = []

    def subscribe(self, pattern: str, handler: Handler, *, critical: bool = False) -> Callable[[], None]:
        """`pattern` is an exact name, a prefix ending in `*` ('tool.*') or '*'. Returns unsubscribe."""
        sub = _Sub(pattern, handler, critical)
        self._subs.append(sub)

        def unsubscribe() -> None:
            if sub in self._subs:
                self._subs.remove(sub)
        return unsubscribe

    @staticmethod
    def _match(pattern: str, name: str) -> bool:
        if pattern == "*":
            return True
        if pattern.endswith("*"):
            return name.startswith(pattern[:-1])
        return pattern == name

    async def emit(self, event: Event) -> None:
        matching = [s for s in self._subs if self._match(s.pattern, event.name)]
        for sub in [s for s in matching if s.critical]:
            res = sub.handler(event)  # exceptions propagate: fail closed
            if inspect.isawaitable(res):
                await res
        for sub in [s for s in matching if not s.critical]:
            try:
                res = sub.handler(event)
                if inspect.isawaitable(res):
                    await res
            except Exception as exc:  # telemetry must not break a run
                self.handler_errors.append((event.name, sub.pattern, exc))
                logger.warning("event_handler_failed", extra={"event": event.name, "error": type(exc).__name__})
