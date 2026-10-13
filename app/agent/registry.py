"""Capability registry + the single execution wrapper every capability goes through.

A capability is a typed, independently testable unit. The wrapper gives all of them the same
validation, permission check, timeout, timing, size cap, error mapping and trace record.
"""
import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal

from pydantic import BaseModel, ValidationError

from app.agent.contracts import (CapabilityError, Empty, ErrorCode, Outcome, ToolError, ToolMeta,
                                 ToolResult, new_id)

logger = logging.getLogger("rag.agent")
Cost = Literal["fast", "medium", "slow"]


@dataclass
class CapabilityContext:
    """Per-run context handed to handlers. `services` is the Services container (providers)."""
    user_id: str
    trace_id: str
    services: Any
    evidence: Any  # EvidenceStore
    permissions: frozenset = frozenset({"read"})
    document_scope: list[str] | None = None
    pages: dict = field(default_factory=dict)  # fetched web pages, keyed by page_id (untrusted)
    scratch: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Capability:
    name: str
    group: str
    description: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    handler: Callable[[BaseModel, CapabilityContext], Awaitable[BaseModel | dict]]
    version: str = "1.0"
    permissions: frozenset = frozenset({"read"})
    cost: Cost = "fast"
    mutates_state: bool = False
    requires_approval: bool = False
    parallel_safe: bool = True
    untrusted_output: bool = False
    llm_exposed: bool = True  # False: orchestrator-only (e.g. summarize_evidence, ingest_approved_source)
    timeout_s: float = 30.0
    max_output_chars: int = 24_000  # hard cap on what may be returned to the LLM/client

    def spec(self) -> dict:
        return {"type": "function", "function": {
            "name": self.name, "description": self.description,
            "parameters": self.input_model.model_json_schema()}}

    def describe(self) -> dict:
        return {"name": self.name, "version": self.version, "group": self.group, "cost": self.cost,
                "permissions": sorted(self.permissions), "mutates_state": self.mutates_state,
                "requires_approval": self.requires_approval, "parallel_safe": self.parallel_safe,
                "untrusted_output": self.untrusted_output, "llm_exposed": self.llm_exposed,
                "input_schema": self.input_model.model_json_schema(),
                "output_schema": self.output_model.model_json_schema()}


@dataclass
class TraceRecord:
    """One row of the execution trace; contains sizes and ids, never document text."""
    execution_id: str
    trace_id: str
    tool_name: str
    group: str
    purpose: str
    status: str
    input_valid: bool
    duration_ms: float
    output_chars: int
    error_code: str | None
    retry_count: int = 0
    extra: dict = field(default_factory=dict)
    input_summary: dict = field(default_factory=dict)
    output_summary: str = ""
    started_at: float = field(default_factory=time.time)


class CapabilityRegistry:
    def __init__(self):
        self._caps: dict[str, Capability] = {}

    def register(self, cap: Capability) -> Capability:
        if cap.name in self._caps:
            raise ValueError(f"capability {cap.name} already registered")
        self._caps[cap.name] = cap
        return cap

    def get(self, name: str) -> Capability | None:
        return self._caps.get(name)

    def all(self) -> list[Capability]:
        return list(self._caps.values())

    def groups(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for c in self._caps.values():
            out.setdefault(c.group, []).append(c.name)
        return out

    def specs_for(self, groups: set[str], *, exclude: set[str] = frozenset()) -> list[dict]:
        """Static, task-filtered tool list for the model: only LLM-exposed capabilities of the
        selected groups. Orchestrator-only capabilities are never offered."""
        return [c.spec() for c in self._caps.values()
                if c.group in groups and c.llm_exposed and c.name not in exclude]

    async def execute(self, name: str, raw_args: Any, ctx: CapabilityContext, *, purpose: str = "",
                      sink: list | None = None, retry_count: int = 0) -> ToolResult:
        cap = self._caps.get(name)
        t0 = time.perf_counter()
        eid = new_id()

        def finish(status, data=None, error=None, valid=True, extra=None, summary=""):
            ms = round((time.perf_counter() - t0) * 1e3, 3)
            chars = len(json.dumps(data, default=str)) if data is not None else 0
            meta = ToolMeta(trace_id=ctx.trace_id, execution_id=eid,
                            capability_group=cap.group if cap else "unknown",
                            version=cap.version if cap else "-", duration_ms=ms, output_chars=chars,
                            input_valid=valid, extra=extra or {})
            res = ToolResult(tool_name=name, status=status, data=data, metadata=meta, error=error)
            rec = TraceRecord(eid, ctx.trace_id, name, meta.capability_group, purpose, status, valid, ms,
                              chars, error.code.value if error else None, retry_count, extra or {},
                              _safe_args(raw_args), summary or (error.message if error else ""),
                              started_at=meta.started_at)
            if sink is not None:
                sink.append(rec)
            logger.info("capability_executed", extra={"tool": name, "status": status, "duration_ms": ms,
                                                      "trace_id": ctx.trace_id, "input_valid": valid,
                                                      "output_chars": chars,
                                                      "error": error.code.value if error else None})
            return res

        if cap is None:
            return finish("error", error=ToolError(code=ErrorCode.not_found,
                                                   message=f"Unknown capability '{name}'."), valid=False)
        if not cap.permissions <= ctx.permissions:
            return finish("error", error=ToolError(code=ErrorCode.denied,
                                                   message="Not permitted for this user."))
        try:
            if isinstance(raw_args, str):
                raw_args = json.loads(raw_args or "{}")
            args = cap.input_model.model_validate(raw_args or {})
        except (ValidationError, json.JSONDecodeError, TypeError) as exc:
            msg = ("Invalid arguments: " + "; ".join(
                f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
                   if isinstance(exc, ValidationError) else "Arguments are not valid JSON.")
            return finish("error", error=ToolError(code=ErrorCode.invalid_input, message=msg[:600]),
                          valid=False)
        status, extra = "success", None
        try:
            out = await asyncio.wait_for(cap.handler(args, ctx), min(cap.timeout_s, 600))
            if isinstance(out, Outcome):
                status, extra, out = out.status, out.extra, out.data
            data = cap.output_model.model_validate(out).model_dump(mode="json")
        except Empty as e:
            return finish("empty", data=_dump(cap, e.data), summary="no results")
        except asyncio.TimeoutError:
            return finish("timeout", error=ToolError(code=ErrorCode.timeout, retryable=True,
                                                     message=f"{name} timed out."))
        except CapabilityError as e:
            return finish("error", error=ToolError(code=e.code, message=e.message, retryable=e.retryable))
        except asyncio.CancelledError:
            raise
        except ValidationError:
            logger.error("capability_output_invalid", extra={"tool": name})
            return finish("error", error=ToolError(code=ErrorCode.internal, message="Capability returned an invalid result."))
        except PermissionError:
            return finish("error", error=ToolError(code=ErrorCode.denied, message="Not permitted."))
        except Exception as exc:  # unexpected: never leak internals
            logger.error("capability_failed", extra={"tool": name, "error": type(exc).__name__,
                                                     "trace_id": ctx.trace_id})
            return finish("error", error=ToolError(code=ErrorCode.internal, message=f"{name} failed unexpectedly."))
        size = len(json.dumps(data, default=str))
        if size > cap.max_output_chars:
            return finish("error", error=ToolError(code=ErrorCode.output_too_large,
                                                   message=f"Result too large ({size} chars); narrow the request."))
        return finish(status, data=data, extra=extra, summary=_summary(data))


def _dump(cap, data):
    try:
        return cap.output_model.model_validate(data).model_dump(mode="json")
    except ValidationError:
        return data


def _summary(data) -> str:
    if not isinstance(data, dict):
        return ""
    return str(data.get("summary") or ", ".join(f"{k}={len(v)}" for k, v in data.items()
                                                 if isinstance(v, list)))[:200]


_SECRET_KEYS = ("key", "token", "password", "secret", "authorization")


def _safe_args(raw) -> dict:
    """Trace-safe copy of the arguments: secrets dropped, long values truncated."""
    if not isinstance(raw, dict):
        return {}
    out = {}
    for k, v in raw.items():
        if any(s in k.lower() for s in _SECRET_KEYS):
            out[k] = "<redacted>"
        elif isinstance(v, str):
            out[k] = v[:200]
        elif isinstance(v, (int, float, bool)) or v is None:
            out[k] = v
        else:
            out[k] = f"<{type(v).__name__} len={len(v) if hasattr(v, '__len__') else '?'}>"
    return out
