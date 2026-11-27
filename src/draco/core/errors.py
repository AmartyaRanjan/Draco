"""Typed failure categories. Every error says whether a retry could help and keeps its cause."""
from __future__ import annotations

from typing import Any


class DracoError(Exception):
    """Base class for every failure Draco raises. `code` is a stable machine-readable category, `retryable` says
    whether trying again could help (set honestly: it drives retries and HTTP status mapping), `details`
    carries structured context, and the original exception stays available as `__cause__`.
    """
    code = "draco_error"
    retryable = False

    def __init__(self, message: str, *, retryable: bool | None = None, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        if retryable is not None:
            self.retryable = retryable
        self.details = details or {}


class ConfigurationError(DracoError):
    """Invalid or missing configuration. The message names the setting."""
    code = "configuration"


class ContractError(DracoError):
    """Input/output failed validation against a declared contract."""
    code = "contract"


class RoutingError(DracoError):
    """No valid destination for a request: unknown agent, a handoff loop, or too many hops."""
    code = "routing"


class ToolError(DracoError):
    """A tool failed in a way the caller should see."""
    code = "tool"


class ProviderError(DracoError):
    """A model, retrieval backend or other upstream dependency failed."""
    code = "provider"


class PolicyViolation(DracoError):
    """The action is not permitted: denied by policy, authorization or tenant isolation. Never retryable."""
    code = "policy"


class ApprovalRequired(PolicyViolation):
    """The call is permitted only with an approval the policy does not grant. A PolicyViolation, so existing
    handlers keep working; callers that must tell 'needs approval' from 'denied' test the type, not the message."""


class ExecutionTimeout(DracoError):
    """The execution passed its deadline. Retryable by default."""
    code = "timeout"
    retryable = True


class ExecutionCancelled(DracoError):
    """The execution was cancelled by the caller or by its parent."""
    code = "cancelled"


class PersistenceError(DracoError):
    """The state store failed. Retryable by default."""
    code = "persistence"
    retryable = True


class BudgetExhausted(DracoError):
    """A budget limit was reached. `resource` names which one and `limit` is its ceiling."""
    code = "budget"

    def __init__(self, resource: str, limit: float, message: str | None = None):
        super().__init__(message or f"{resource} budget exhausted (limit {limit})",
                         details={"resource": resource, "limit": limit})
        self.resource, self.limit = resource, limit


class PluginError(DracoError):
    """A plugin could not be registered, started or stopped."""
    code = "plugin"


class DuplicateRegistration(PluginError):
    """A plugin id or service name was registered twice."""
    code = "duplicate_registration"


class MissingDependency(PluginError):
    """A plugin requires a service that nothing provides."""
    code = "missing_dependency"


class DependencyCycle(PluginError):
    """Plugins require each other in a cycle, so no start order exists."""
    code = "dependency_cycle"
