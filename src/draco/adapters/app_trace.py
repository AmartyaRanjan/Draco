"""Bridge Draco execution events into the app's existing per-request `Trace` records, so there is
one tracing system: the app Trace keeps its stage timings, Draco adds identity and lifecycle."""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("rag.draco")


class AppLogExporter:
    """Exporter that writes each record through the app's structured logger (no payload data:
    the Tracer already redacts and truncates)."""

    def export(self, record: dict[str, Any]) -> None:
        logger.info(record["event"], extra={"draco": {k: record[k] for k in
                                                      ("execution_id", "parent_execution_id", "correlation_id")},
                                            "data": record["data"]})
