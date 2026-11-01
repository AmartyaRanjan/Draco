"""Optional LangGraph checkpoint configuration and SQLite saver helpers."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CheckpointConfig:
    """Canonical LangGraph thread identity for a Draco execution."""

    execution_id: str
    tenant_id: str | None = None
    user_id: str | None = None
    session_id: str | None = None
    correlation_id: str | None = None
    checkpoint_ns: str = ""
    recursion_limit: int = 25

    def __post_init__(self) -> None:
        if not self.execution_id:
            raise ValueError("execution_id must not be empty")
        if self.recursion_limit < 1:
            raise ValueError("recursion_limit must be positive")

    def to_runnable_config(self) -> dict[str, Any]:
        metadata = {
            key: value
            for key, value in {
                "draco_tenant_id": self.tenant_id,
                "draco_user_id": self.user_id,
                "draco_session_id": self.session_id,
                "draco_correlation_id": self.correlation_id,
            }.items()
            if value is not None
        }
        return {
            "configurable": {
                "thread_id": self.execution_id,
                "checkpoint_ns": self.checkpoint_ns,
            },
            "recursion_limit": self.recursion_limit,
            "metadata": metadata,
        }

    @classmethod
    def from_execution_context(cls, ctx: Any, *, recursion_limit: int = 25) -> CheckpointConfig:
        """Build from a trusted Draco ExecutionContext."""
        return cls(
            execution_id=ctx.execution_id,
            tenant_id=ctx.tenant_id,
            user_id=ctx.user_id,
            session_id=ctx.session_id,
            correlation_id=ctx.correlation_id,
            recursion_limit=recursion_limit,
        )


def checkpoint_config(
    execution_id: str,
    *,
    tenant_id: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    correlation_id: str | None = None,
    checkpoint_ns: str = "",
    recursion_limit: int = 25,
) -> dict[str, Any]:
    """Create a LangGraph RunnableConfig with stable execution and identity metadata."""
    return CheckpointConfig(
        execution_id=execution_id,
        tenant_id=tenant_id,
        user_id=user_id,
        session_id=session_id,
        correlation_id=correlation_id,
        checkpoint_ns=checkpoint_ns,
        recursion_limit=recursion_limit,
    ).to_runnable_config()


def build_sqlite_checkpointer(path: str | Path) -> Any:
    """Create LangGraph's optional SQLite saver for durable graph state.

    The returned saver owns its SQLite connection; close ``saver.conn`` when
    its lifecycle ends. Install ``draco[agents]`` to enable this helper.
    """
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError as exc:
        raise ImportError(
            "SQLite checkpointing requires the optional agents dependencies; "
            "install with: pip install 'draco[agents]'"
        ) from exc

    db_path = str(path)
    if db_path != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    try:
        saver = SqliteSaver(conn)
        saver.setup()
    except Exception:
        conn.close()
        raise
    return saver


def resume_command(decision: str) -> Any:
    """Create the validated resume payload used by the approval interrupt contract."""
    if decision not in ("approve", "reject"):
        raise ValueError("decision must be 'approve' or 'reject'")
    try:
        from langgraph.types import Command
    except ImportError as exc:
        raise ImportError(
            "LangGraph is required for resume_command; install with: pip install 'draco[agents]'"
        ) from exc
    return Command(resume={"decision": decision})


def is_interrupted(result: Any) -> bool:
    """Return whether a LangGraph result carries at least one interrupt."""
    return isinstance(result, dict) and bool(result.get("__interrupt__"))


def extract_interrupts(result: Any) -> list[Any]:
    """Return interrupt values, not LangGraph's wrapper objects."""
    if not is_interrupted(result):
        return []
    return [getattr(item, "value", item) for item in result["__interrupt__"]]
