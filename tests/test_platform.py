"""HTTP platform layer: rate limiting, auth lockout, body caps, request ids, health, metrics, LLM admission."""
import asyncio
import dataclasses
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.api import chat, health
from app.core import concurrency, ratelimit
from app.core.http_settings import PlatformSettings, get_platform_settings
from app.db import registry
from app.main import app
from tests.test_inference import isettings

KEYS = {"good-key": "alice", "other-key": "bob"}


def settings(**kw) -> PlatformSettings:
    base = dataclasses.replace(get_platform_settings(), rate_limit_enabled=True, metrics_token=None, ready_cache_s=0,
                               policies={"heavy": (60, 3), "upload": (60, 3), "read": (60, 3), "ip": (6000, 1000), "authfail": (60, 3)})
    return dataclasses.replace(base, **kw)


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class MemoryLimiterTests(unittest.TestCase):
    def hit(self, lim, key="k", cost=1):
        return asyncio.run(lim.hit("p", key, 60, 3, cost))  # 1 token/s, burst 3

    def test_burst_then_refill(self):
        clock = Clock()
        lim = ratelimit.MemoryLimiter(clock)
        self.assertEqual([self.hit(lim).allowed for _ in range(4)], [True, True, True, False])
        d = self.hit(lim)
        self.assertFalse(d.allowed)
        self.assertAlmostEqual(d.retry_after_s, 1.0, places=2)
        clock.t += 1.0
        self.assertTrue(self.hit(lim).allowed)

    def test_buckets_are_per_key_and_capped_at_burst(self):
        clock = Clock()
        lim = ratelimit.MemoryLimiter(clock)
        for _ in range(3):
            self.hit(lim, "a")
        self.assertFalse(self.hit(lim, "a").allowed)
        self.assertTrue(self.hit(lim, "b").allowed)
        clock.t += 3600  # a long idle period must not bank more than `burst` tokens
        self.assertEqual([self.hit(lim, "a").allowed for _ in range(4)], [True, True, True, False])

    def test_peek_does_not_spend(self):
        lim = ratelimit.MemoryLimiter(Clock())
        for _ in range(10):
            self.assertTrue(self.hit(lim, cost=0).allowed)
        for _ in range(3):
            self.hit(lim)
        self.assertFalse(self.hit(lim, cost=0).allowed)


class RedisFailoverTests(unittest.TestCase):
    def test_unreachable_redis_falls_back_to_memory_and_trips_breaker(self):
        lim = ratelimit.RedisLimiter("redis://127.0.0.1:1/0")  # nothing listens here
        self.assertTrue(asyncio.run(lim.hit("p", "k", 60, 2)).allowed)
        self.assertGreater(lim._down_until, 0)  # breaker open: later calls skip the connect attempt
        self.assertTrue(asyncio.run(lim.hit("p", "k", 60, 2)).allowed)
        self.assertFalse(asyncio.run(lim.hit("p", "k", 60, 2)).allowed)  # still limited, by the in-memory bucket

    def test_real_redis_shares_one_budget(self):
        import redis
        try:
            redis.Redis.from_url(get_platform_settings().redis_url, socket_connect_timeout=0.3).ping()
        except Exception:
            self.skipTest("Redis not reachable")
        import uuid
        key = uuid.uuid4().hex

        async def run():
            a, b = ratelimit.RedisLimiter(get_platform_settings().redis_url), ratelimit.RedisLimiter(get_platform_settings().redis_url)
            r = [(await x.hit("t", key, 60, 3)).allowed for x in (a, b, a, b)]  # two "replicas", one bucket
            return r, await b.hit("t", key, 60, 3, cost=0)
        results, peek = asyncio.run(run())
        self.assertEqual(results, [True, True, True, False])
        self.assertFalse(peek.allowed)


class ApiTestBase(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.stack.enter_context(patch.object(registry, "DB_PATH", Path(self.tmp.name) / "documents.db"))
        registry.init_db()
        self.cfg = settings()
        for mod in ("app.core.ratelimit", "app.core.concurrency", "app.core.middleware", "app.api.health"):
            self.stack.enter_context(patch(f"{mod}.get_platform_settings", lambda: self.cfg))
        self.stack.enter_context(patch("app.api.chat.get_inference_settings", lambda: isettings(api_keys=KEYS)))
        ratelimit.set_limiter(ratelimit.MemoryLimiter())
        self.addCleanup(ratelimit.set_limiter, None)
        concurrency._sem = None
        self.addCleanup(setattr, concurrency, "_sem", None)
        self.client = TestClient(app)

    H = {"x-api-key": "good-key"}


class RateLimitApiTests(ApiTestBase):
    def test_per_user_limit_returns_429_with_retry_after(self):
        codes = [self.client.get("/documents/x", headers=self.H).status_code for _ in range(5)]
        self.assertEqual(codes[:3], [404, 404, 404])  # reached the handler (document missing), budget of 3
        self.assertEqual(codes[3:], [429, 429])
        r = self.client.get("/documents/x", headers=self.H)
        self.assertGreaterEqual(int(r.headers["retry-after"]), 1)
        self.assertEqual(r.headers["x-ratelimit-remaining"], "0")

    def test_limits_are_per_user(self):
        for _ in range(4):
            self.client.get("/documents/x", headers=self.H)
        self.assertEqual(self.client.get("/documents/x", headers={"x-api-key": "other-key"}).status_code, 404)

    def test_remaining_header_on_success(self):
        r = self.client.get("/documents/x", headers=self.H)
        self.assertEqual((r.headers["x-ratelimit-limit"], r.headers["x-ratelimit-remaining"]), ("3", "2"))

    def test_policies_are_independent(self):
        for _ in range(4):
            self.client.get("/documents/x", headers=self.H)  # exhaust "read"
        r = self.client.post("/query", headers=self.H, json={"query": ""})
        self.assertEqual(r.status_code, 422)  # "heavy" bucket untouched; request reached validation

    def test_documents_and_query_now_require_a_key(self):
        self.assertEqual(self.client.get("/documents/x").status_code, 401)
        self.assertEqual(self.client.post("/query", json={"query": "hi"}).status_code, 401)
        self.assertEqual(self.client.post("/documents", files={"file": ("a.pdf", b"%PDF-1")}).status_code, 401)

    def test_failed_auth_lockout_blocks_even_a_valid_key(self):
        codes = [self.client.get("/documents/x", headers={"x-api-key": f"bad{i}"}).status_code for i in range(5)]
        self.assertEqual(codes, [401, 401, 401, 429, 429])
        r = self.client.get("/documents/x", headers=self.H)
        self.assertEqual(r.status_code, 429)  # brute-force guard: no oracle for guessing the key during lockout

    def test_ip_flood_limit(self):
        self.cfg = settings(policies={**self.cfg.policies, "ip": (60, 2)})
        codes = [self.client.get("/health/ready" if False else "/documents/x", headers=self.H).status_code for _ in range(4)]
        self.assertEqual(codes[:2], [404, 404])
        self.assertEqual(codes[2], 429)

    def test_forwarded_for_ignored_unless_proxies_are_trusted(self):
        ip = lambda hops: (setattr(self, "cfg", settings(trusted_proxy_hops=hops)),
                           ratelimit.client_ip_from({"x-forwarded-for": "6.6.6.6, 10.0.0.9"}, "10.0.0.1"))[1]
        self.assertEqual(ip(0), "10.0.0.1")   # spoofable header is not trusted by default
        self.assertEqual(ip(1), "10.0.0.9")   # the entry our own proxy appended
        self.assertEqual(ip(5), "10.0.0.1")   # fewer entries than hops: use the peer

    def test_disabled_limiter_lets_everything_through(self):
        self.cfg = settings(rate_limit_enabled=False)
        self.assertTrue(all(self.client.get("/documents/x", headers=self.H).status_code == 404 for _ in range(10)))


class MiddlewareTests(ApiTestBase):
    def test_request_id_generated_echoed_and_sanitised(self):
        r = self.client.get("/health")
        self.assertEqual(len(r.headers["x-request-id"]), 32)
        self.assertEqual(self.client.get("/health", headers={"x-request-id": "trace-123"}).headers["x-request-id"], "trace-123")
        bad = self.client.get("/health", headers={"x-request-id": "a b\"c"}).headers["x-request-id"]
        self.assertNotEqual(bad, "a b\"c")

    def test_security_headers_present_once(self):
        r = self.client.get("/health")
        self.assertEqual(r.headers["x-content-type-options"], "nosniff")
        self.assertEqual(r.headers["x-frame-options"], "DENY")

    def test_oversized_json_body_rejected_by_content_length(self):
        self.cfg = settings(max_body_bytes=2048, max_upload_bytes=4096)
        r = self.client.post("/query", headers=self.H, json={"query": "x" * 3000})
        self.assertEqual(r.status_code, 413)

    def test_oversized_chunked_body_rejected_without_content_length(self):
        self.cfg = settings(max_body_bytes=2048, max_upload_bytes=4096)
        r = self.client.post("/query", headers={**self.H, "content-type": "application/json"},
                             content=iter([b'{"query": "', b"x" * 3000, b'"}']))
        self.assertEqual(r.status_code, 413)

    def test_upload_gets_the_larger_cap(self):
        self.cfg = settings(max_body_bytes=2048, max_upload_bytes=1 << 20)
        with patch("app.api.routes.ingestion_queue"), patch("app.api.routes.register_upload", return_value=None), \
                patch("app.api.routes.submit_document", return_value="job-1"), \
                patch("app.api.routes.UPLOAD_DIR", __import__("pathlib").Path(__import__("tempfile").mkdtemp())):
            r = self.client.post("/documents", headers=self.H, files={"file": ("a.pdf", b"%PDF-1.7 " + b"x" * 10_000)})
        self.assertEqual(r.status_code, 202)

    def test_unhandled_exception_is_a_clean_500_with_request_id(self):
        with patch("app.api.routes.get_document_for_owner", side_effect=RuntimeError("boom secret")):
            r = TestClient(app, raise_server_exceptions=False).get("/documents/x", headers=self.H)
        self.assertEqual(r.status_code, 500)
        self.assertNotIn("boom", r.text)
        self.assertIn("x-request-id", r.headers)


class LlmAdmissionTests(ApiTestBase):
    def test_saturated_model_gives_503_not_500(self):
        self.cfg = settings(llm_max_concurrent=1, llm_queue_wait_s=0.05)

        async def scenario():
            await concurrency.acquire_llm_slot()  # someone else holds the only slot
            try:
                with self.assertRaises(HTTPException) as ctx:
                    await concurrency.acquire_llm_slot()
                return ctx.exception
            finally:
                concurrency.release_llm_slot()
        exc = asyncio.run(scenario())
        self.assertEqual((exc.status_code, exc.headers["Retry-After"]), (503, "5"))

    def test_route_maps_saturation_to_503(self):
        async def busy():
            raise HTTPException(503, "The model is at capacity. Retry shortly.", headers={"Retry-After": "5"})
        with patch("app.api.query.concurrency.llm_slot", side_effect=lambda: _Raises(busy)):
            r = self.client.post("/query", headers=self.H, json={"query": "hi"})
        self.assertEqual(r.status_code, 503)
        self.assertEqual(r.headers["retry-after"], "5")

    def test_slot_released_on_error(self):
        self.cfg = settings(llm_max_concurrent=1, llm_queue_wait_s=0.05)

        async def scenario():
            for _ in range(3):
                with self.assertRaises(RuntimeError):
                    async with concurrency.llm_slot():
                        raise RuntimeError
            async with concurrency.llm_slot():
                return True
        self.assertTrue(asyncio.run(scenario()))


class _Raises:
    def __init__(self, fn):
        self.fn = fn

    async def __aenter__(self):
        await self.fn()

    async def __aexit__(self, *a):
        return False


class OpsEndpointTests(ApiTestBase):
    def test_liveness_is_dependency_free(self):
        self.assertEqual(self.client.get("/health").json(), {"status": "ok"})

    def test_ready_reports_each_dependency_and_503_when_one_is_down(self):
        def boom():
            raise ConnectionError("down")

        async def llm_ok():
            return None
        with patch.object(health, "_redis", lambda: None), patch.object(health, "_qdrant", boom), \
                patch.object(health, "_opensearch", lambda: None), patch.object(health, "_llm", llm_ok):
            r = self.client.get("/health/ready")
        self.assertEqual(r.status_code, 503)
        body = r.json()
        self.assertEqual(body["status"], "unavailable")
        self.assertTrue(body["checks"]["redis"]["ok"])
        self.assertEqual(body["checks"]["qdrant"], {"ok": False, "ms": body["checks"]["qdrant"]["ms"], "error": "ConnectionError"})

    def test_ready_ok_when_all_up(self):
        async def llm_ok():
            return None
        with patch.object(health, "_redis", lambda: None), patch.object(health, "_qdrant", lambda: None), \
                patch.object(health, "_opensearch", lambda: None), patch.object(health, "_llm", llm_ok):
            self.assertEqual(self.client.get("/health/ready").status_code, 200)

    def test_metrics_exposed_and_count_requests_by_route_template(self):
        self.client.get("/documents/abc", headers=self.H)
        body = self.client.get("/metrics").text
        self.assertIn('http_requests_total{method="GET",route="/documents/{document_id}",status="404"}', body)
        self.assertNotIn("abc", body)  # no per-id label cardinality

    def test_metrics_token_enforced_when_configured(self):
        self.cfg = settings(metrics_token="s3cret")
        self.assertEqual(self.client.get("/metrics").status_code, 401)
        self.assertEqual(self.client.get("/metrics", headers={"authorization": "Bearer nope"}).status_code, 401)
        self.assertEqual(self.client.get("/metrics", headers={"authorization": "Bearer s3cret"}).status_code, 200)


class SettingsValidationTests(unittest.TestCase):
    def test_production_refuses_unsafe_flags(self):
        prod = dataclasses.replace(get_platform_settings(), env="production", cors_origins=("https://a.example",), rate_limit_enabled=True)
        prod.validate()
        with self.assertRaises(ValueError):
            prod.validate(auth_disabled=True)
        with self.assertRaises(ValueError):
            dataclasses.replace(prod, cors_origins=("*",)).validate()
        with self.assertRaises(ValueError):
            dataclasses.replace(prod, rate_limit_enabled=False).validate()

    def test_bad_values_rejected(self):
        base = get_platform_settings()
        for kw in ({"env": "staging"}, {"llm_max_concurrent": 0}, {"max_body_bytes": 10}, {"max_upload_bytes": 1024}):
            with self.assertRaises(ValueError, msg=kw):
                dataclasses.replace(base, **kw).validate()

    def test_policy_parsing(self):
        from app.core.http_settings import _policy
        with patch.dict("os.environ", {"RATE_LIMIT_X": "30, 10"}):
            self.assertEqual(_policy("X", 1, 1), (30.0, 10.0))
        with patch.dict("os.environ", {"RATE_LIMIT_X": "oops"}), self.assertRaises(ValueError):
            _policy("X", 1, 1)
        with patch.dict("os.environ", {"RATE_LIMIT_X": "0,5"}), self.assertRaises(ValueError):
            _policy("X", 1, 1)


class LogFormatTests(unittest.TestCase):
    def test_json_log_carries_request_id_and_extras(self):
        import json
        import logging
        from app.core import observability as obs
        rec = logging.LogRecord("n", logging.INFO, "f", 1, "hello %s", ("w",), None)
        rec.stage = "dense"
        tok = obs.request_id_var.set("rid-1")
        try:
            out = json.loads(obs.JsonFormatter().format(rec))
        finally:
            obs.request_id_var.reset(tok)
        self.assertEqual((out["msg"], out["request_id"], out["stage"], out["level"]), ("hello w", "rid-1", "dense", "INFO"))


if __name__ == "__main__":
    unittest.main()
