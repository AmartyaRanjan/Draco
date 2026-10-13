"""Structured JSON logging with request correlation, and Prometheus metrics."""
import contextvars
import json
import logging
import sys
import time

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

_STD = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message", "asctime", "taskName"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + f".{int(record.msecs):03d}Z",
               "level": record.levelname, "logger": record.name, "msg": record.getMessage(),
               "request_id": request_id_var.get()}
        for k, v in record.__dict__.items():
            if k not in _STD:
                out[k] = v
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        return json.dumps(out, default=str, ensure_ascii=False)


class _RequestIdText(logging.Formatter):
    def format(self, record):
        record.request_id = request_id_var.get()
        return super().format(record)


def configure_logging(level: str = "INFO", as_json: bool = True) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if as_json else
                         _RequestIdText("%(asctime)s %(levelname)s %(name)s [%(request_id)s] %(message)s"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    for noisy in ("httpx", "httpcore", "opensearch", "urllib3", "elastic_transport"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    # uvicorn's own access log is replaced by the middleware's structured one
    logging.getLogger("uvicorn.access").disabled = True


REGISTRY = CollectorRegistry()
HTTP_REQUESTS = Counter("http_requests_total", "HTTP requests", ["method", "route", "status"], registry=REGISTRY)
HTTP_LATENCY = Histogram("http_request_duration_seconds", "Time to first response byte", ["method", "route"],
                         buckets=(.01, .05, .1, .25, .5, 1, 2.5, 5, 10, 20, 40, 80), registry=REGISTRY)
HTTP_INFLIGHT = Gauge("http_requests_in_flight", "Requests being handled", registry=REGISTRY)
RATE_LIMIT_REJECTIONS = Counter("rate_limit_rejections_total", "Requests answered 429", ["policy"], registry=REGISTRY)
RATE_LIMIT_BACKEND_ERRORS = Counter("rate_limit_backend_errors_total", "Redis failures; limiter fell back to memory",
                                    registry=REGISTRY)
LLM_SLOT_REJECTIONS = Counter("llm_slot_rejections_total", "Requests answered 503 because the LLM was saturated",
                              registry=REGISTRY)
LLM_INFLIGHT = Gauge("llm_requests_in_flight", "Requests holding an LLM slot", registry=REGISTRY)
READY = Gauge("dependency_up", "1 if the dependency passed its last readiness probe", ["dependency"], registry=REGISTRY)


def render_metrics() -> tuple[bytes, str]:
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
