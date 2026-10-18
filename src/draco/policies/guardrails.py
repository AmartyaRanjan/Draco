"""Input/output guardrails: plain callables enforced by the runtime, failing closed."""
from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Any

from draco.core.context import ExecutionContext
from draco.core.errors import PolicyViolation

Guardrail = Callable[[ExecutionContext, Any], Awaitable[None] | None]


async def run_guardrails(guardrails: list[Guardrail], ctx: ExecutionContext, payload: Any) -> None:
    """Each guardrail raises PolicyViolation to block. Any other exception also blocks (fail closed)."""
    for g in guardrails:
        try:
            res = g(ctx, payload)
            if inspect.isawaitable(res):
                await res
        except PolicyViolation:
            raise
        except Exception as exc:
            raise PolicyViolation("guardrail failed and the request was blocked") from exc


def max_input_chars(limit: int) -> Guardrail:
    def check(ctx: ExecutionContext, payload: Any) -> None:
        if isinstance(payload, str) and len(payload) > limit:
            raise PolicyViolation(f"input exceeds {limit} characters")
    return check
