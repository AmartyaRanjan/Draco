"""The documentation's runnable examples are executed, so the docs cannot drift from the code.

Markers (HTML comments, invisible on the site) go on the line before a fenced block:

    <!-- doctest -->            the next ```python block is a complete program: it must exit 0
    <!-- doctest requires=langgraph -->   same, but skipped where that package is not installed
    <!-- output -->             the next fenced block is the exact stdout of the program above

Blocks without a marker are illustrations and are not run. Keep runnable blocks self-contained and
deterministic (no ids, timings or paths in what they print).
"""
from __future__ import annotations

import os
import pathlib
import re
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[3]
DOCS = ROOT / "docs"
FENCE = re.compile(r"^```(\w*)[^\n]*\n(.*?)^```[ \t]*$", re.S | re.M)


def blocks(path: pathlib.Path):
    """Yield (line, code, expected_output_or_None, required_module_or_None) for every doctest block in a page."""
    text = path.read_text()
    for m in re.finditer(r"<!-- doctest(?: requires=([\w.]+))? -->[ \t]*\n" + FENCE.pattern, text, re.S | re.M):
        code = m.group(3)
        expected = None
        after = re.match(r"\s*<!-- output -->[ \t]*\n" + FENCE.pattern, text[m.end():], re.S | re.M)
        if after:
            expected = after.group(2)
        yield text.count("\n", 0, m.start()) + 1, code, expected, m.group(1)


CASES = [(p, line, code, exp, req) for p in sorted(DOCS.rglob("*.md")) for line, code, exp, req in blocks(p)]


def test_the_docs_contain_runnable_examples():
    assert len(CASES) >= 10, "expected the documentation to carry runnable, tested examples"


@pytest.mark.parametrize("path,line,code,expected,requires", CASES,
                         ids=[f"{p.relative_to(DOCS)}:{line}" for p, line, *_ in CASES])
def test_doc_example_runs(path, line, code, expected, requires):
    if requires:
        pytest.importorskip(requires)
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")}
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT, env=env, timeout=60)
    where = f"{path.relative_to(ROOT)}:{line}"
    assert r.returncode == 0, f"{where} failed:\n{r.stderr[-2000:]}"
    if expected is not None:
        got = "\n".join(s.rstrip() for s in r.stdout.strip().splitlines())
        want = "\n".join(s.rstrip() for s in expected.strip().splitlines())
        assert got == want, f"{where} printed different output than the docs show:\n--- docs\n{want}\n--- actual\n{got}"
