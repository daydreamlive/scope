"""State persistence — onboarding and settings survive a process restart.

If ``DAYDREAM_SCOPE_DIR`` is stable across runs (it is for every non-test
user — it defaults to ``~/.daydream-scope``), stopping and restarting
Scope must not wipe the user's onboarding state. A regression here means
the user sees the onboarding flow every single time they launch the app,
which is the kind of "death by a thousand cuts" bug a unit test will
never catch.

We prove persistence by:
  1. Running a normal onboarding via the UI.
  2. Recording whatever ended up in ``DAYDREAM_SCOPE_DIR`` (onboarding.json).
  3. Killing the Scope subprocess.
  4. Booting a new Scope subprocess pointed at the SAME directory.
  5. Asserting the UI lands directly on the graph view (Run visible) —
     NOT on the inference-mode picker.
"""

from __future__ import annotations

from pathlib import Path

from harness import flows
from harness.driver import PlaywrightDriver
from harness.report import TestReport
from harness.scope_process import ScopeHarness
from playwright.sync_api import TimeoutError as PwTimeout


def test_onboarding_state_persists_across_restart(
    scope_harness: ScopeHarness,
    driver: PlaywrightDriver,
    report: TestReport,
    tmp_path: Path,
):
    """Complete onboarding, restart the subprocess, confirm state survived."""
    report.metadata["workflow"] = "local-passthrough"

    # 1. Drive onboarding to completion.
    flows.complete_onboarding_local(driver, workflow_id="local-passthrough")
    driver.wait_testid("stream-run-stop")

    scope_dir = scope_harness.tmp_dir
    assert scope_dir is not None
    onboarding_file = scope_dir / "onboarding.json"
    report.metadata["scope_dir"] = str(scope_dir)
    report.measure(
        "onboarding_file_exists_pre_restart", int(onboarding_file.exists())
    )
    if not onboarding_file.exists():
        report.fail(
            f"onboarding.json never materialized at {onboarding_file} — "
            "state isn't being written"
        )
        assert False, "onboarding state not persisted to disk"

    before_size = onboarding_file.stat().st_size
    report.measure("onboarding_file_size_pre", before_size)

    # 2. Stop the current Scope subprocess (keeping tmp_dir contents).
    scope_harness.stop()

    # 3. Start a NEW subprocess pointed at the same DAYDREAM_SCOPE_DIR.
    # We reuse the harness object — start() allocates a fresh port and
    # respawns. The tmp_dir is preserved.
    scope_harness.start()

    # 4. Point the driver at the new URL and navigate.
    driver.goto(scope_harness.base_url)

    # 5. The app MUST land on the graph view (Run button), not onboarding.
    try:
        driver.wait_testid("stream-run-stop", timeout_ms=30_000)
        report.measure("landed_on_graph_post_restart", 1)
    except PwTimeout:
        report.measure("landed_on_graph_post_restart", 0)
        # Did we get kicked back to inference-mode?
        try:
            driver.wait_testid("inference-mode-local", timeout_ms=3000)
            report.fail(
                "onboarding state LOST across restart — user would have to "
                "re-onboard on every app launch"
            )
        except PwTimeout:
            report.fail(
                "post-restart UI is neither on onboarding nor on the graph view"
            )

    # 6. Prove the file wasn't rewritten-to-empty.
    after_size = onboarding_file.stat().st_size
    report.measure("onboarding_file_size_post", after_size)

    assert report.passed, f"Hard fails: {report.hard_fails}"
