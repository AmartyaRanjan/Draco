from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from draco.core.context import ExecutionContext

# A workload is anything the harness can run: (context, input) -> output. Existing pipelines are
# adapted to this shape in draco.adapters; they stay callable on their own.
Workload = Callable[[ExecutionContext, Any], Awaitable[Any]]
