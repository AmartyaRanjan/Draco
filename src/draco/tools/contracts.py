"""Tool contracts: one result envelope, stable error codes, and the declarative `Tool` definition."""
from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field

from draco.core.context import ExecutionContext


class ErrorCode(StrEnum):
    invalid_input = "invalid_input"
    not_found = "not_found"
    denied = "denied"
    approval_required = "approval_required"
    not_configured = "not_configured"
    timeout = "timeout"
    upstream_error = "upstream_error"
    unsafe_url = "unsafe_url"
    limit_exceeded = "limit_exceeded"
    output_too_large = "output_too_large"
    conflict = "conflict"
    internal = "internal"


class ToolErrorInfo(BaseModel):
    code: ErrorCode
    message: str  # client-safe: never paths, secrets or stack traces
    retryable: bool = False


class ToolMeta(BaseModel):
    trace_id: str
    execution_id: str
    capability_group: str
    version: str
    duration_ms: float = 0.0
    output_chars: int = 0
    input_valid: bool = True
    started_at: float = Field(default_factory=time.time)
    extra: dict[str, Any] = {}


class ToolResult(BaseModel):
    """success = produced data; empty = ran, found nothing; pending = accepted, continues asynchronously;
    error/timeout = failed (see `error`)."""
    tool_name: str
    status: Literal["success", "empty", "pending", "error", "timeout"]
    data: dict[str, Any] | None = None
    metadata: ToolMeta
    error: ToolErrorInfo | None = None

    @property
    def ok(self) -> bool:
        return self.status in ("success", "empty", "pending")


class ToolFailure(Exception):
    """Raised by handlers for expected failures; mapped to ToolResult.error."""

    def __init__(self, code: ErrorCode, message: str, retryable: bool = False):
        super().__init__(message)
        self.code, self.message, self.retryable = code, message, retryable


class Empty(Exception):
    """Handler signal: ran successfully but found nothing."""

    def __init__(self, data: dict[str, Any] | None = None):
        super().__init__("empty")
        self.data = data or {}


class Outcome:
    """Optional handler return wrapper to set a non-default status (e.g. "pending") and extra metadata."""

    def __init__(self, data: Any, status: str = "success", extra: dict[str, Any] | None = None):
        self.data, self.status, self.extra = data, status, extra or {}


ToolHandler = Callable[[Any, ExecutionContext], Awaitable[BaseModel | dict[str, Any] | Outcome]]


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    input_model: type[BaseModel]
    handler: ToolHandler
    output_model: type[BaseModel] | None = None  # None: any JSON object
    group: str = "native"
    version: str = "1.0"
    permissions: frozenset[str] = frozenset({"read"})
    mutates_state: bool = False  # mutating tools are never auto-retried
    requires_approval: bool = False
    parallel_safe: bool = True
    untrusted_output: bool = False  # output is flagged and scanned for injected instructions
    llm_exposed: bool = True
    timeout_s: float = 30.0
    max_output_chars: int = 24_000

    def spec(self) -> dict[str, Any]:
        return {"type": "function", "function": {"name": self.name, "description": self.description,
                                                  "parameters": self.input_model.model_json_schema()}}

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "version": self.version, "group": self.group,
                "permissions": sorted(self.permissions), "mutates_state": self.mutates_state,
                "requires_approval": self.requires_approval, "parallel_safe": self.parallel_safe,
                "untrusted_output": self.untrusted_output, "llm_exposed": self.llm_exposed,
                "input_schema": self.input_model.model_json_schema()}


def new_id() -> str:
    return uuid.uuid4().hex
