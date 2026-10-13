"""One pure-ASGI middleware (not BaseHTTPMiddleware: that buffers and mishandles SSE disconnects) that adds,
in order: request id, per-address flood limit, body-size cap, security headers, access log and metrics."""
import json
import logging
import time
import uuid

from app.core import observability as obs
from app.core import ratelimit
from app.core.http_settings import get_platform_settings

logger = logging.getLogger("rag.access")
_SKIP_LIMIT = ("/health", "/metrics")
_SECURITY = [(b"x-content-type-options", b"nosniff"), (b"x-frame-options", b"DENY"),
             (b"referrer-policy", b"no-referrer"), (b"cache-control", b"no-store")]


class _TooLarge(Exception):
    pass


async def _send_json(send, status, detail, headers=()):
    body = json.dumps({"detail": detail}).encode()
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode()), *headers]})
    await send({"type": "http.response.body", "body": body})


class PlatformMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        s = get_platform_settings()
        headers = dict(scope["headers"])
        rid = headers.get(b"x-request-id", b"").decode()[:64]
        if not rid or not all(c.isalnum() or c in "-_" for c in rid):
            rid = uuid.uuid4().hex
        token = obs.request_id_var.set(rid)
        scope.setdefault("state", {})
        method, path = scope["method"], scope["path"]
        status, t0, started = 500, time.perf_counter(), False
        extra = [(b"x-request-id", rid.encode())]
        obs.HTTP_INFLIGHT.inc()

        async def send_wrapper(message):
            nonlocal status, started
            if too_large and not started:
                # The app saw a failed body read (FastAPI turns it into a 400); the real cause is the size cap.
                if message["type"] == "http.response.body":
                    return
                started, status = True, 413
                body = json.dumps({"detail": f"Request body exceeds {limit} bytes"}).encode()
                await send({"type": "http.response.start", "status": 413, "headers": [
                    (b"content-type", b"application/json"), (b"content-length", str(len(body)).encode()), *extra, *_SECURITY]})
                await send({"type": "http.response.body", "body": body})
                return
            if too_large:
                return
            if message["type"] == "http.response.start":
                status, started = message["status"], True
                rl = scope["state"].get("ratelimit")
                have = {k.lower() for k, _ in message.get("headers", [])}
                hdrs = list(message.get("headers", [])) + extra + [h for h in _SECURITY if h[0] not in have]
                if rl and rl.limit and b"x-ratelimit-limit" not in have:
                    hdrs += [(b"x-ratelimit-limit", str(rl.limit).encode()), (b"x-ratelimit-remaining", str(rl.remaining).encode())]
                message = {**message, "headers": hdrs}
            await send(message)

        limit = s.max_upload_bytes if (method == "POST" and path == "/documents") else s.max_body_bytes
        seen, too_large = 0, False

        async def counting_receive():
            nonlocal seen, too_large
            msg = await receive()
            if msg["type"] == "http.request":
                seen += len(msg.get("body", b""))
                if seen > limit:
                    too_large = True
                    raise _TooLarge
            return msg

        try:
            declared = headers.get(b"content-length")
            if declared and declared.isdigit() and int(declared) > limit:
                return await _send_json(send_wrapper, 413, f"Request body exceeds {limit} bytes")
            if not path.startswith(_SKIP_LIMIT):
                ip = ratelimit.client_ip_from({k.decode(): v.decode() for k, v in scope["headers"]},
                                              scope["client"][0] if scope.get("client") else None)
                scope["state"]["client_ip"] = ip
                d = await ratelimit.check("ip", ip)
                if not d.allowed:
                    exc = ratelimit.reject("ip", d)
                    return await _send_json(send_wrapper, 429, exc.detail, [(k.lower().encode(), v.encode()) for k, v in exc.headers.items()])
            await self.app(scope, counting_receive, send_wrapper)
        except _TooLarge:
            if not started:
                await _send_json(send_wrapper, 413, f"Request body exceeds {limit} bytes")
        except Exception:
            logger.exception("unhandled_error", extra={"path": path})
            if not started:
                await _send_json(send_wrapper, 500, "Internal server error")
        finally:
            route = getattr(scope.get("route"), "path", None) or "unmatched"
            dur = time.perf_counter() - t0
            obs.HTTP_INFLIGHT.dec()
            obs.HTTP_REQUESTS.labels(method, route, str(status)).inc()
            obs.HTTP_LATENCY.labels(method, route).observe(dur)
            if not path.startswith(_SKIP_LIMIT):
                logger.info("request", extra={"method": method, "path": path, "route": route, "status": status,
                                              "duration_ms": round(dur * 1000, 1), "client_ip": scope["state"].get("client_ip")})
            obs.request_id_var.reset(token)
