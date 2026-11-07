"""Retention: turn `DracoSettings.execution_retention_days` / `event_retention_days` into a store purge.

Run it periodically (cron, a background task, or `scripts/draco_retention.py`). Waiting and running
executions, and their descendants, are never purged. Purging a completed execution also forgets its
idempotency key, so a retry with that key after the retention window runs again."""
from __future__ import annotations

import time
from typing import Any

from draco.core.config import DracoSettings
from draco.core.errors import ConfigurationError
from draco.state.stores import PurgeResult, RetentionStore

DAY_S = 86400.0


def cutoffs(settings: DracoSettings, *, now: float | None = None) -> tuple[float | None, float | None]:
    """(older_than, events_older_than) epoch-second cutoffs for the configured retention windows."""
    now = time.time() if now is None else now
    ex = None if settings.execution_retention_days is None else now - settings.execution_retention_days * DAY_S
    ev = None if settings.event_retention_days is None else now - settings.event_retention_days * DAY_S
    return ex, ev


async def apply_retention(store: Any, settings: DracoSettings, *, now: float | None = None) -> PurgeResult:
    """Purge `store` according to `settings`. A no-op when no retention window is configured."""
    older_than, events_older_than = cutoffs(settings, now=now)
    if older_than is None and events_older_than is None:
        return PurgeResult()
    if not isinstance(store, RetentionStore):
        raise ConfigurationError(f"{type(store).__name__} does not support retention (no purge())")
    return await store.purge(older_than=older_than, events_older_than=events_older_than)
