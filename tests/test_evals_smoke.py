"""Smoke test for the eval harness wiring.

This runs ONE case with N=1 under ``@pytest.mark.eval``. Default ``pytest``
skips it (pyproject's addopts includes ``-m "not eval"``); run with
``uv run pytest -m eval`` to include it.

This test is not a pass-rate gate — it only verifies:
1. Cases can be loaded.
2. The driver can drive the agent in-process.
3. The grader produces a structured result.

Pass-rate enforcement is intentionally left to ``python -m evals``.
"""

from __future__ import annotations

import os

import pytest


@pytest.mark.eval
@pytest.mark.anyio
async def test_smoke_single_case():
    # Skip if we can't reach the Anthropic API (no key set).
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set; smoke eval needs live API")

    from pathlib import Path

    from evals.case import load_case
    from evals.runner import run_cases

    case_path = (
        Path(__file__).resolve().parent.parent
        / "evals"
        / "cases"
        / "starter-ltx-text-to-video.yaml"
    )
    case = load_case(case_path)

    summaries = await run_cases(
        [case],
        runs_override=1,
        output_dir=Path("/tmp/eval-smoke"),
    )
    assert len(summaries) == 1
    summary = summaries[0]
    assert len(summary.runs) == 1
    # We don't assert pass here — the smoke test is about wiring, not
    # agent quality. But we do assert the run produced *some* result
    # structure (either a proposal or a recorded failure).
    run = summary.runs[0]
    assert run.drive is not None
    assert run.drive.trace, "driver produced an empty SSE trace"
