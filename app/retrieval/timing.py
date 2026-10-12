"""Monotonic stage timing and structured request records."""
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

logger = logging.getLogger("rag.retrieval")

# Model-loading durations, kept apart from steady-state request timings.
COLD_START_MS: dict[str, float] = {}


def now_ns() -> int:
    return time.perf_counter_ns()


@dataclass
class Trace:
    request_id: str
    stages: dict = field(default_factory=dict)  # name -> {ms, status, error?}
    metrics: dict = field(default_factory=dict)
    status: str = "ok"
    failed_stage: str | None = None
    _t0: int = field(default_factory=now_ns)

    @contextmanager
    def stage(self, name: str):
        """Record elapsed time even when the body raises."""
        start = now_ns()
        try:
            yield
        except BaseException as exc:
            self.record(name, start, "error", type(exc).__name__)
            if self.failed_stage is None:
                self.failed_stage = name
            raise
        else:
            self.record(name, start, "ok")

    def record(self, name, start_ns, status="ok", error=None, end_ns=None):
        ms = ((end_ns or now_ns()) - start_ns) / 1e6
        entry = {"ms": round(ms, 3), "status": status}
        if error:
            entry["error"] = error
        self.stages[name] = entry

    def set(self, **kw):
        self.metrics.update(kw)

    def finish(self, status="ok"):
        self.status = status
        self.stages["total"] = {"ms": round((now_ns() - self._t0) / 1e6, 3), "status": status}
        # No query text or document content is logged.
        logger.info("retrieval_request", extra={"request_id": self.request_id, "trace": self.as_dict()})
        return self

    def as_dict(self):
        return {
            "request_id": self.request_id,
            "status": self.status,
            "failed_stage": self.failed_stage,
            "stages": self.stages,
            "metrics": self.metrics,
        }
