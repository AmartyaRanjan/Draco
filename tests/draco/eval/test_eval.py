import pytest

from tests.draco.eval.runner import SUITES, run_suite


@pytest.mark.parametrize("suite", sorted(SUITES))
async def test_suite_passes_all_cases(suite):
    out = await run_suite(suite)
    assert not out["failures"], out["failures"]
    assert out["total"] >= 7
