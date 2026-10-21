"""Native Python tools: a decorator that turns an async function into a registry `Capability`,
so they get the same validation, permissions, timeout, size cap and error contract as the
existing 21 capabilities."""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from app.agent.contracts import CapabilityError
from app.agent.registry import Capability, CapabilityContext, CapabilityRegistry
from pydantic import BaseModel, ConfigDict

from draco.core.context import ExecutionContext


class NoArgs(BaseModel):
    """Input model for tools that take no arguments."""
    model_config = ConfigDict(extra="forbid")


class AnyOutput(BaseModel):
    model_config = ConfigDict(extra="allow")


def native_tool(registry: CapabilityRegistry, *, name: str, description: str, input_model: type[BaseModel],
                output_model: type[BaseModel] = AnyOutput, group: str = "native",
                permissions: frozenset[str] | set[str] = frozenset({"read"}), requires_approval: bool = False,
                mutates_state: bool = False, parallel_safe: bool = True, timeout_s: float = 30.0,
                max_output_chars: int = 24_000, untrusted_output: bool = False, llm_exposed: bool = True,
                version: str = "1.0") -> Callable[[Callable[[Any, ExecutionContext], Awaitable[Any]]], Capability]:
    """Usage:
        @native_tool(registry, name="add", description="Add", input_model=AddArgs)
        async def add(args: AddArgs, ctx: ExecutionContext) -> dict: return {"sum": args.a + args.b}
    Raise `app.agent.contracts.CapabilityError(code, message, retryable)` for expected failures."""

    def deco(fn: Callable[[Any, ExecutionContext], Awaitable[Any]]) -> Capability:
        async def handler(args: BaseModel, cap_ctx: CapabilityContext) -> BaseModel | dict:
            out = await fn(args, cap_ctx.scratch["draco_ctx"])
            return out if isinstance(out, (BaseModel, dict)) else {"result": out}
        cap = Capability(name=name, group=group, description=description, input_model=input_model,
                         output_model=output_model, handler=handler, version=version,
                         permissions=frozenset(permissions), mutates_state=mutates_state,
                         requires_approval=requires_approval, parallel_safe=parallel_safe,
                         untrusted_output=untrusted_output, llm_exposed=llm_exposed, timeout_s=timeout_s,
                         max_output_chars=max_output_chars)
        registry.register(cap)
        return cap
    return deco


__all__ = ["native_tool", "AnyOutput", "NoArgs", "CapabilityError"]
