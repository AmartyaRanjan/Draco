"""Translate the existing `app` exceptions into Draco's typed errors, keeping the cause."""
from __future__ import annotations

from draco.core.errors import ContractError, DracoError, ExecutionTimeout, ProviderError

_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


def map_app_error(exc: BaseException) -> DracoError | None:
    """Return a DracoError for known app exceptions, else None (caller re-raises the original)."""
    name = type(exc).__name__
    status = getattr(exc, "status_code", None)
    if name in ("RetrievalError", "InferenceError"):
        stage = getattr(exc, "stage", None) or getattr(exc, "code", None)
        details = {"stage": stage, "status_code": status}
        if status == 422:
            return ContractError(str(exc), details=details)
        if status == 504:
            return ExecutionTimeout(str(exc), details=details)
        return ProviderError(str(exc), retryable=status in _RETRYABLE_STATUS, details=details)
    return None
