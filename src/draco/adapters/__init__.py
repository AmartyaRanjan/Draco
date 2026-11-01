"""Adapters translate external components into Draco contracts.

Application adapters are loaded on access so importing framework adapters does
not require the starter application or its runtime dependencies.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_APP_EXPORTS = {
    "agent_runner_workload": ("app_agent", "agent_runner_workload"),
    "ingestion_workload": ("app_ingestion", "ingestion_workload"),
    "rag_workload": ("app_retrieval", "rag_workload"),
    "retrieval_workload": ("app_retrieval", "retrieval_workload"),
    "map_app_error": ("errors", "map_app_error"),
}
__all__ = list(_APP_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _APP_EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    return getattr(import_module(f"{__name__}.{module_name}"), attribute)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
