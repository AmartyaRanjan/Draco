"""Validated framework settings. Provider settings (vLLM, Qdrant ...) stay in the `app` settings."""
from __future__ import annotations

import os
from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from draco.core.budget import BudgetLimits
from draco.core.errors import ConfigurationError


class DracoSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    default_timeout_s: float | None = Field(default=120.0, gt=0)
    max_model_calls: int | None = Field(default=None, gt=0)
    max_tool_calls: int | None = Field(default=None, gt=0)
    max_depth: int | None = Field(default=8, gt=0)
    max_retries: int | None = Field(default=None, gt=0)
    max_tokens: int | None = Field(default=None, gt=0)
    max_concurrency: int = Field(default=8, gt=0)
    max_concurrency_per_tenant: int | None = Field(default=None, gt=0)  # root executions in flight per tenant; excess is rejected
    state_path: str | None = None  # SQLite file; None = in-memory stores
    trace_path: str | None = None  # JSONL file; None = in-memory trace only
    redact_keys: tuple[str, ...] = ("key", "token", "password", "secret", "authorization")
    pii_keys: tuple[str, ...] = ("user_id", "tenant_id", "subject", "resource_tenant")  # hashed in traces and persisted events
    require_identity: bool = False  # True: every protected operation without a trusted Identity is denied
    max_payload_chars: int = Field(default=2000, gt=0)

    def limits(self) -> BudgetLimits:
        return BudgetLimits(max_model_calls=self.max_model_calls, max_tool_calls=self.max_tool_calls,
                            max_duration_s=self.default_timeout_s, max_depth=self.max_depth,
                            max_retries=self.max_retries, max_tokens=self.max_tokens,
                            max_concurrency=self.max_concurrency)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None, prefix: str = "DRACO_") -> DracoSettings:
        env = os.environ if env is None else env
        raw: dict[str, object] = {}
        for name in cls.model_fields:
            key = prefix + name.upper()
            if key in env and env[key] != "":
                v = env[key]
                raw[name] = tuple(x.strip() for x in v.split(",")) if name in ("redact_keys", "pii_keys") else v
        try:
            return cls.model_validate(raw)
        except ValidationError as exc:
            problems = "; ".join(f"{prefix}{'.'.join(map(str, e['loc'])).upper()}: {e['msg']}"
                                 for e in exc.errors())
            raise ConfigurationError(f"invalid configuration: {problems}") from exc
