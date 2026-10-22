"""End-to-end: every shipped example runs in a fresh interpreter and prints what the docs claim."""
import os
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[3]
EXPECT = {
    "a_simple": ["status\": \"completed", "trace events: ['execution.started', 'security.allowed', 'execution.finished']"],
    "b_rag": ["execution_id == retrieval request_id: True", "status=budget_exhausted", "degraded=['bm25']"],
    "c_multi_agent": ["path: ['triage', 'math']", "- auditor [completed]", "unknown agent -> failed routing"],
    "d_tool": ["'fahrenheit': 212.0", "'denied'", "'invalid_input'"],
    "e_recovery": ["'charge card' not repeated", "same execution id: True", "attempts: 2"],
    "f_plugin": ["started in order: ['quota', 'audit']", "quota exceeded for alice", "quota service present? False"],
    "g_secure_tenants": ["{'notes': ['acme roadmap']}", "cross-tenant access denied", "no identity established",
                         "contradict the authenticated identity"],
    "h_langgraph": ["status: waiting", "same execution id: True"],
}


@pytest.mark.parametrize("name", sorted(EXPECT))
def test_example_runs(name):
    if name == "h_langgraph":
        pytest.importorskip("langgraph")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), str(ROOT), *sys.path])}
    r = subprocess.run([sys.executable, "-m", f"examples.{name}"], capture_output=True, text=True, cwd=ROOT, env=env, timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
    for needle in EXPECT[name]:
        assert needle in r.stdout, f"{needle!r} not in output:\n{r.stdout}"
