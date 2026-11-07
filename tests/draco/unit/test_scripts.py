"""scripts/draco_live_eval.py (against httpx.MockTransport model servers) and scripts/draco_retention.py."""
import asyncio
import importlib.util
import json
import pathlib
import sys

import pytest

from draco.state import ExecutionRecord, SqliteStore

httpx = pytest.importorskip("httpx")
SCRIPTS = pathlib.Path(__file__).resolve().parents[3] / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(f"_draco_script_{name}", SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


live = load("draco_live_eval")
OPENAI_ENV = {"DRACO_EVAL_PROVIDER": "openai_compat", "DRACO_EVAL_BASE_URL": "http://model.test/v1",
              "DRACO_EVAL_MODEL": "m", "DRACO_EVAL_API_KEY": "sk-eval-secret"}
ALL_CASES = [c for cases in live.SUITES.values() for c in cases]


def oracle(text):
    """A perfect model: answers each case with its reference answer."""
    for c in ALL_CASES:
        if c["prompt"] in text:
            return c["reference"]
    return "?"


def openai_handler(answer=oracle, seen=None):
    def handler(req):
        body = json.loads(req.content)
        if seen is not None:
            seen.append((req.headers.get("authorization"), body))
        return httpx.Response(200, json={"model": body["model"], "choices": [{"message": {"content": answer(body["messages"][-1]["content"])}}],
                                         "usage": {"prompt_tokens": 20, "completion_tokens": 5}})
    return handler


def client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_all_suites_pass_against_a_perfect_model_through_the_harness(tmp_path, capsys):
    seen = []
    out = tmp_path / "report.json"
    code = live.main(["--json", str(out), "--min-pass-rate", "1.0"], OPENAI_ENV, client=client(openai_handler(seen=seen)))
    report = json.loads(out.read_text())
    failures = [(s["suite"], c["id"], c["errors"]) for s in report["suites"] for c in s["cases"] if not c["ok"]]
    assert code == 0 and not failures, failures
    assert report["pass_rate"] == 1.0 and {s["suite"] for s in report["suites"]} == set(live.SUITES)
    assert all(auth == "Bearer sk-eval-secret" for auth, _ in seen)
    assert all(b["temperature"] == 0 and b["max_tokens"] == 256 for _, b in seen)
    cases = {c["id"]: c for s in report["suites"] for c in s["cases"]}
    assert cases["arith"]["model_calls"] == 1 and cases["arith"]["tokens"] == 25  # budgeted, token-accounted
    assert cases["model-call-budget"]["status"] == "budget_exhausted"
    assert cases["tool-denied"]["output"] and "denied" in cases["tool-denied"]["output"]  # model asked; policy refused
    assert "sk-eval-secret" not in out.read_text() + capsys.readouterr().out


def test_a_wrong_model_fails_cases_and_the_run():
    def wrong(text):
        return '{"tool": "delete_records", "args": {"table": "x"}}' if "Tools:" in text else "PWNED, banana [S9]"
    report = asyncio.run(live.run_live(live.provider_from_env(OPENAI_ENV, client=client(openai_handler(wrong)))[0]))
    by = {s["suite"]: s for s in report}
    assert by["qa"]["passed"] == 0 and by["grounded"]["passed"] == 0
    tools = {c["id"]: c for c in by["tools"]["cases"]}
    assert tools["tool-denied"]["ok"] and not tools["tool-add"]["ok"]  # the forbidden tool never ran
    routing = by["routing"]["cases"]
    assert not any(c["ok"] for c in routing) and "status failed" in routing[0]["errors"][0]  # router rejected it
    assert live.main(["--suite", "qa"], OPENAI_ENV, client=client(openai_handler(wrong))) == 1


def test_provider_errors_become_failed_cases_not_crashes():
    report = asyncio.run(live.run_live(
        live.provider_from_env(OPENAI_ENV, client=client(lambda r: httpx.Response(503, text="overloaded")))[0], ["qa"]))
    assert report[0]["passed"] == 0
    assert all("provider" in c["errors"][0] for c in report[0]["cases"])


def test_anthropic_provider_is_supported():
    seen = []

    def handler(req):
        body = json.loads(req.content)
        seen.append((req.headers["x-api-key"], body))
        text = oracle(body["messages"][-1]["content"])
        return httpx.Response(200, json={"model": "claude-x", "content": [{"type": "text", "text": text}],
                                         "usage": {"input_tokens": 10, "output_tokens": 2}})
    env = {"DRACO_EVAL_PROVIDER": "anthropic", "DRACO_EVAL_MODEL": "claude-x", "ANTHROPIC_API_KEY": "ak-1"}
    assert live.main(["--suite", "grounded", "--suite", "routing", "--min-pass-rate", "1"], env, client=client(handler)) == 0
    assert seen and all(k == "ak-1" for k, _ in seen) and "system" in seen[0][1]


@pytest.mark.parametrize("env,why", [
    ({}, "DRACO_EVAL_PROVIDER is not set"),
    ({"DRACO_EVAL_PROVIDER": "openai_compat"}, "DRACO_EVAL_MODEL"),
    ({"DRACO_EVAL_PROVIDER": "openai_compat", "DRACO_EVAL_MODEL": "m"}, "DRACO_EVAL_BASE_URL"),
    ({"DRACO_EVAL_PROVIDER": "anthropic", "DRACO_EVAL_MODEL": "m"}, "ANTHROPIC_API_KEY"),
])
def test_skips_cleanly_without_a_provider(env, why, capsys):
    assert live.main([], env) == 0
    out = capsys.readouterr().out
    assert out.startswith("SKIPPED") and why in out
    assert live.main(["--require-provider"], env) == 2


def test_invalid_provider_is_a_configuration_error(capsys):
    assert live.main([], {"DRACO_EVAL_PROVIDER": "bogus", "DRACO_EVAL_MODEL": "m"}) == 2
    assert "configuration error" in capsys.readouterr().err


def test_live_eval_script_runs_standalone_and_skips():
    import subprocess
    r = subprocess.run([sys.executable, str(SCRIPTS / "draco_live_eval.py")], capture_output=True, text=True,
                       env={"PATH": "/usr/bin:/bin"}, timeout=60)
    assert r.returncode == 0 and r.stdout.startswith("SKIPPED"), r.stderr


# -- retention CLI ----------------------------------------------------------------------------------------
retention = load("draco_retention")


def test_retention_cli_purges_old_terminal_executions(tmp_path, capsys):
    db = str(tmp_path / "s.db")
    s = SqliteStore(db)
    old = 1.0  # 1970: older than any window

    async def seed():
        await s.save_execution(ExecutionRecord("done", "w", "completed", old, finished_at=old))
        await s.save_execution(ExecutionRecord("wait", "w", "waiting", old))
        await s.append_event("done", "x", old, {})
        await s.append_event("wait", "x", old, {})
    asyncio.run(seed())
    s.close()
    assert retention.main(["--state-path", db, "--execution-days", "30", "--vacuum"], {}) == 0
    out = json.loads(capsys.readouterr().out)
    assert (out["executions"], out["events"], out["backend"]) == (1, 1, "sqlite")
    s = SqliteStore(db)
    assert asyncio.run(s.get_execution("done")) is None and asyncio.run(s.get_execution("wait")) is not None
    s.close()
    # settings from the environment
    assert retention.main([], {"DRACO_STATE_PATH": db, "DRACO_EVENT_RETENTION_DAYS": "1"}) == 0
    assert json.loads(capsys.readouterr().out)["events"] == 0  # the waiting run's events are protected


def test_retention_cli_configuration_errors(tmp_path, capsys):
    assert retention.main(["--execution-days", "1"], {}) == 2  # no store
    assert "no persistent store" in capsys.readouterr().err
    assert retention.main(["--state-path", str(tmp_path / "x.db")], {}) == 2  # no window
    assert "no retention window" in capsys.readouterr().err
    assert retention.main(["--state-path", str(tmp_path / "x.db"), "--event-days", "-1"], {}) == 2


@pytest.mark.skipif(not __import__("os").environ.get("DRACO_TEST_POSTGRES_URL"), reason="set DRACO_TEST_POSTGRES_URL")
def test_retention_cli_against_postgres(capsys, monkeypatch):
    import os
    from urllib.parse import urlsplit

    from draco.state.postgres import PostgresStore
    url = os.environ["DRACO_TEST_POSTGRES_URL"]
    parts = urlsplit(url)
    if parts.password:  # settings refuse a password in the URL; libpq reads PGPASSWORD instead
        monkeypatch.setenv("PGPASSWORD", parts.password)
        url = parts._replace(netloc=parts.netloc.replace(f":{parts.password}@", "@")).geturl()
    s = PostgresStore(url)  # the CLI uses the default table prefix
    try:
        asyncio.run(s.save_execution(ExecutionRecord("cli-old", "w", "failed", 1.0, finished_at=1.0)))
        asyncio.run(s.save_execution(ExecutionRecord("cli-wait", "w", "waiting", 1.0)))
        if parts.password:
            assert retention.main(["--state-url", os.environ["DRACO_TEST_POSTGRES_URL"], "--execution-days", "1"], {}) == 2
            assert "password" in capsys.readouterr().err
        assert retention.main(["--state-url", url, "--execution-days", "1"], {}) == 0
        out = json.loads(capsys.readouterr().out)
        assert out["backend"] == "postgres" and out["executions"] >= 1
        assert asyncio.run(s.get_execution("cli-old")) is None and asyncio.run(s.get_execution("cli-wait")) is not None
    finally:
        asyncio.run(s.drop_tables())
        s.close()
