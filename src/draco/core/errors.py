"""Typed failure categories. Every error says whether a retry could help and keeps its cause."""
from __future__ import annotations

from typing import Any


class DracoError(Exception):
    code = "draco_error"
    retryable = False

    def __init__(self, message: str, *, retryable: bool | None = None, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        if retryable is not None:
            self.retryable = retryable
        self.details = details or {}


class ConfigurationError(DracoError):
    code = "configuration"


class ContractError(DracoError):
    """Input/output failed validation against a declared contract."""
    code = "contract"


class RoutingError(DracoError):
    code = "routing"


class ToolError(DracoError):
    code = "tool"


class ProviderError(DracoError):
    """A model, retrieval backend or other upstream dependency failed."""
    code = "provider"


class PolicyViolation(DracoError):
    code = "policy"


class ExecutionTimeout(DracoError):
    code = "timeout"
    retryable = True


class ExecutionCancelled(DracoError):
    code = "cancelled"


class PersistenceError(DracoError):
    code = "persistence"
    retryable = True


class BudgetExhausted(DracoError):
    code = "budget"

    def __init__(self, resource: str, limit: float, message: str | None = None):
        super().__init__(message or f"{resource} budget exhausted (limit {limit})",
                         details={"resource": resource, "limit": limit})
        self.resource, self.limit = resource, limit


class PluginError(DracoError):
    code = "plugin"


class DuplicateRegistration(PluginError):
    code = "duplicate_registration"


class MissingDependency(PluginError):
    code = "missing_dependency"


class DependencyCycle(PluginError):
    code = "dependency_cycle"
