"""API-key scopes, expiry and the auth audit log (app/core/auth.py), and their wiring onto routers in app/main.py."""
import asyncio
import datetime as dt
import logging
import unittest
from unittest.mock import patch

from fastapi import HTTPException
from starlette.requests import Request

from app.core import auth, ratelimit
from app.core.auth import DEFAULT_SCOPES, ApiKey, parse_api_keys, require_scope
from app.inference.config import _keys
from tests.test_inference import isettings
from tests.test_platform import ApiTestBase

TODAY = dt.datetime.now(dt.UTC).date()
PAST = (TODAY - dt.timedelta(days=1)).isoformat()
FUTURE = (TODAY + dt.timedelta(days=30)).isoformat()


def request(method="GET", path="/documents/x", key=None, ip="10.0.0.1"):
    headers = [(b"x-api-key", key.encode())] if key else []
    return Request({"type": "http", "method": method, "path": path, "headers": headers, "query_string": b"",
                    "client": (ip, 1234), "server": ("test", 80), "scheme": "http", "root_path": "", "state": {}})


class ParseTests(unittest.TestCase):
    def test_legacy_entries_keep_all_non_admin_scopes_and_never_expire(self):
        keys = parse_api_keys("k1:alice, k2:bob")
        self.assertEqual(keys["k1"], ApiKey("alice", DEFAULT_SCOPES, None))
        self.assertNotIn("admin", keys["k2"].scopes)
        self.assertEqual(_keys("k1:alice,k2:bob"), {"k1": "alice", "k2": "bob"})  # InferenceSettings.api_keys unchanged

    def test_scopes_and_expiry_in_either_order(self):
        keys = parse_api_keys(f"a:u1:read+chat,b:u2:expires=2027-01-31,c:u3:expires=2027-01-31:agent,d:u4:admin:expires={FUTURE}")
        self.assertEqual(keys["a"].scopes, {"read", "chat"})
        self.assertIsNone(keys["a"].expires)
        self.assertEqual((keys["b"].scopes, keys["b"].expires), (DEFAULT_SCOPES, dt.date(2027, 1, 31)))
        self.assertEqual(keys["c"].scopes, {"agent"})
        self.assertTrue(keys["d"].allows("upload"))  # admin implies every scope

    def test_malformed_entries_fail_without_echoing_the_key(self):
        for raw in ("secretkey", "secretkey:", "secretkey:u:reed", "secretkey:u:expires=31-01-2027",
                    "secretkey:u:read:chat", "secretkey:u:expires=2027-01-01:expires=2027-01-02", "secretkey:a,secretkey:b"):
            with self.subTest(raw=raw), self.assertRaises(ValueError) as cm:
                parse_api_keys(raw)
            self.assertNotIn("secretkey", str(cm.exception))

    def test_expiry_is_exclusive_at_utc_midnight(self):
        k = ApiKey("u", DEFAULT_SCOPES, dt.date(2027, 1, 31))
        self.assertFalse(k.expired(dt.datetime(2027, 1, 30, 23, 59, tzinfo=dt.UTC)))
        self.assertTrue(k.expired(dt.datetime(2027, 1, 31, 0, 0, tzinfo=dt.UTC)))

    def test_unknown_scope_in_guard_definition_is_a_bug(self):
        with self.assertRaises(ValueError):
            require_scope("write")


class GuardTests(unittest.TestCase):
    KEYS = parse_api_keys(f"rk:reader:read,ck:chatter:chat+read,ek:old:expires={PAST},ak:root:admin,lk:legacy")

    def setUp(self):
        p = patch.object(auth, "get_api_keys", lambda: self.KEYS)
        p.start()
        self.addCleanup(p.stop)
        ratelimit.set_limiter(ratelimit.MemoryLimiter())
        self.addCleanup(ratelimit.set_limiter, None)

    def run_guard(self, guard, req):
        return asyncio.run(guard(req))

    def test_scope_allowed_denied_and_admin(self):
        g = require_scope("chat")
        self.assertIsNone(self.run_guard(g, request("POST", "/chat", "ck")))
        self.assertIsNone(self.run_guard(g, request("POST", "/chat", "ak")))
        self.assertIsNone(self.run_guard(g, request("POST", "/chat", "lk")))  # legacy key: unchanged behaviour
        with self.assertRaises(HTTPException) as cm:
            self.run_guard(g, request("POST", "/chat", "rk"))
        self.assertEqual(cm.exception.status_code, 403)

    def test_method_specific_scope(self):
        g = require_scope("read", {"POST": "upload"})
        self.assertIsNone(self.run_guard(g, request("GET", "/documents/x", "rk")))
        with self.assertRaises(HTTPException) as cm:
            self.run_guard(g, request("POST", "/documents", "rk"))
        self.assertEqual(cm.exception.status_code, 403)
        self.assertIsNone(self.run_guard(g, request("POST", "/documents", "lk")))

    def test_expired_key_is_401(self):
        with self.assertRaises(HTTPException) as cm:
            self.run_guard(require_scope("read"), request(key="ek"))
        self.assertEqual((cm.exception.status_code, cm.exception.detail), (401, "API key expired"))

    def test_unknown_and_missing_keys_are_left_to_the_route_auth(self):
        g = require_scope("read")
        self.assertIsNone(self.run_guard(g, request(key="nope")))
        self.assertIsNone(self.run_guard(g, request()))

    def test_auth_disabled_skips_the_guard(self):
        with patch.dict("os.environ", {"APP_AUTH_DISABLED": "true"}):
            self.assertIsNone(self.run_guard(require_scope("admin"), request(key="rk")))

    def test_audit_events_and_no_secrets(self):
        g = require_scope("chat")
        with self.assertLogs("rag.audit", level="INFO") as logs:
            self.run_guard(g, request("POST", "/chat", "ck"))
            self.run_guard(g, request("POST", "/chat", "guessed-secret-value"))
            for key in ("rk", "ek"):
                with self.assertRaises(HTTPException):
                    self.run_guard(g, request("POST", "/chat", key))
        events = [(r.event, getattr(r, "user", None), getattr(r, "reason", None)) for r in logs.records]
        self.assertEqual(events, [("auth_success", "chatter", None), ("auth_failure", None, "unknown_key"),
                                  ("auth_scope_denied", "reader", None), ("auth_expired", "old", None)])
        self.assertEqual(logs.records[0].key_id, auth.key_id("ck"))
        for r in logs.records:
            blob = repr(r.__dict__)
            self.assertNotIn("guessed-secret-value", blob)
            self.assertNotIn("'ck'", blob)
            self.assertEqual(r.client_ip, "10.0.0.1")


class GuardApiTests(ApiTestBase):
    """End to end through app.main: router-level scope guards run before each route's own auth dependency."""

    def setUp(self):
        super().setUp()
        keys = parse_api_keys(f"good-key:alice:read,up-key:alice:upload,old-key:alice:expires={PAST}")
        self.stack.enter_context(patch.object(auth, "get_api_keys", lambda: keys))
        self.stack.enter_context(patch("app.api.chat.get_inference_settings",
                                       lambda: isettings(api_keys={k: v.user for k, v in keys.items()})))

    def test_read_scope_reaches_handler(self):
        self.assertEqual(self.client.get("/documents/x", headers=self.H).status_code, 404)

    def test_missing_scope_is_403_on_each_router(self):
        pdf = {"file": ("a.pdf", b"%PDF-1.4\n", "application/pdf")}
        self.assertEqual(self.client.post("/documents", headers=self.H, files=pdf).status_code, 403)
        self.assertEqual(self.client.post("/chat", headers=self.H, json={"message": "hi"}).status_code, 403)
        self.assertEqual(self.client.post("/agent/chat", headers=self.H, json={"message": "hi"}).status_code, 403)
        self.assertEqual(self.client.get("/documents/x", headers={"x-api-key": "up-key"}).status_code, 403)

    def test_expired_key_is_rejected_and_counts_toward_lockout(self):
        codes = [self.client.get("/documents/x", headers={"x-api-key": "old-key"}).status_code for _ in range(5)]
        self.assertEqual(codes, [401, 401, 401, 429, 429])
        self.assertEqual(self.client.get("/documents/x", headers=self.H).status_code, 429)

    def test_unknown_key_still_handled_by_route_auth(self):
        self.assertEqual(self.client.get("/documents/x", headers={"x-api-key": "bad"}).status_code, 401)

    def test_health_needs_no_key(self):
        self.assertEqual(self.client.get("/health").status_code, 200)


if __name__ == "__main__":
    logging.basicConfig()
    unittest.main()
