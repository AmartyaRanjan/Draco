"""Shared execution contracts for every capability: one result envelope, stable error codes."""
import time
import uuid
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class ErrorCode(str, Enum):
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


class ToolError(BaseModel):
    code: ErrorCode
    message: str  # client-safe: never contains paths, secrets or stack traces
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
    extra: dict = {}  # capability-specific timings/ids (job id, artifact id, token usage ...)


class ToolResult(BaseModel):
    """success = produced data; empty = ran fine, found nothing; pending = accepted, work continues
    asynchronously (job id in data); error/timeout = failed (see `error`)."""
    tool_name: str
    status: Literal["success", "empty", "pending", "error", "timeout"]
    data: dict[str, Any] | None = None
    metadata: ToolMeta
    error: ToolError | None = None

    @property
    def ok(self) -> bool:
        return self.status in ("success", "empty", "pending")


class CapabilityError(Exception):
    """Raised by handlers for expected failures; mapped to ToolResult.error."""

    def __init__(self, code: ErrorCode, message: str, retryable: bool = False):
        super().__init__(message)
        self.code, self.message, self.retryable = code, message, retryable


class Empty(Exception):
    """Handler signal: ran successfully but found nothing. Carries an explanatory data dict."""

    def __init__(self, data: dict | None = None):
        super().__init__("empty")
        self.data = data or {}


class Outcome:
    """Optional handler return wrapper to set a non-default status (e.g. "pending") and extra
    metadata (job/artifact ids, token usage) without polluting the data payload."""

    def __init__(self, data, status: str = "success", extra: dict | None = None):
        self.data, self.status, self.extra = data, status, extra or {}


def new_id() -> str:
    return uuid.uuid4().hex
