"""Deterministic execution accounting. Limits are enforced here, never by the model."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from draco.core.errors import BudgetExhausted


@dataclass(frozen=True)
class BudgetLimits:
    max_model_calls: int | None = None
    max_tool_calls: int | None = None
    max_duration_s: float | None = None
    max_depth: int | None = 8
    max_retries: int | None = None
    max_tokens: int | None = None
    max_concurrency: int | None = None

    def __post_init__(self) -> None:
        for name, v in self.__dict__.items():
            if v is not None and v <= 0:
                raise ValueError(f"{name} must be positive, got {v}")


class BudgetTracker:
    """Shared by an execution and all of its children, so nested work draws from one budget."""

    def __init__(self, limits: BudgetLimits | None = None):
        self.limits = limits or BudgetLimits()
        self._lock = threading.Lock()
        self.model_calls = 0
        self.tool_calls = 0
        self.retries = 0
        self.tokens = 0
        self.started = time.monotonic()

    def _charge(self, attr: str, amount: int, limit: int | None, resource: str) -> None:
        with self._lock:
            new = getattr(self, attr) + amount
            if limit is not None and new > limit:
                raise BudgetExhausted(resource, limit)
            setattr(self, attr, new)

    def charge_model_call(self, tokens: int = 0) -> None:
        self._charge("model_calls", 1, self.limits.max_model_calls, "model_calls")
        if tokens:
            self._charge("tokens", tokens, self.limits.max_tokens, "tokens")

    def add_tokens(self, tokens: int) -> None:
        self._charge("tokens", tokens, self.limits.max_tokens, "tokens")

    def charge_tool_call(self) -> None:
        self._charge("tool_calls", 1, self.limits.max_tool_calls, "tool_calls")

    def charge_retry(self) -> None:
        self._charge("retries", 1, self.limits.max_retries, "retries")

    def check_depth(self, depth: int) -> None:
        if self.limits.max_depth is not None and depth > self.limits.max_depth:
            raise BudgetExhausted("depth", self.limits.max_depth)

    def check_duration(self) -> None:
        lim = self.limits.max_duration_s
        if lim is not None and time.monotonic() - self.started > lim:
            raise BudgetExhausted("duration", lim)

    def remaining_duration(self) -> float | None:
        lim = self.limits.max_duration_s
        return None if lim is None else max(0.0, lim - (time.monotonic() - self.started))

    def snapshot(self) -> dict[str, float | int]:
        return {"model_calls": self.model_calls, "tool_calls": self.tool_calls, "retries": self.retries,
                "tokens": self.tokens, "elapsed_s": round(time.monotonic() - self.started, 6)}
