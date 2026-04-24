"""Chaos — rapid Stop/Run toggling after a successful first frame.

Simulates the user who can't decide if they like what they see: for N
seconds, every 500–2000ms, click Stop then Run again. Asserts that no
click produces a retry, no session closes unexpectedly, and every Run
produces a new frame within a generous timeout.

This is exactly the pattern that exposes failure mode #2 — Scope-server ↔
remote-inference bad interactions when a session is torn down and brought
back up quickly.

Note for future test authors: this file is a reference implementation
of chaos-style tests under the `@scenario` decorator. See
`product-tests/WRITING_TESTS.md` for the cookbook and
`product-tests/_templates/chaos.py.tpl` for a blank template.
"""

from __future__ import annotations

import pytest
from harness.scenario import scenario


@scenario(
    mode="local",
    workflow="local-passthrough",
    feature="lifecycle",
    marks=(pytest.mark.chaos,),
)
def test_rapid_stop_start_local(ctx):
    """Onboard, Run, hammer Stop/Run for 30s; every Run must land a frame."""
    ctx.metadata("chaos_seed", ctx.chaos_seed)

    ctx.complete_onboarding()
    ctx.run_and_wait_first_frame(timeout_ms=60_000)

    toggles = {"count": 0, "frames_after_run": 0}

    def toggle_stop_start():
        # ctx.toggle_run auto-marks initiated stops on the Stop side.
        ctx.toggle_run()  # Stop
        ctx.sleep(200)
        ctx.toggle_run()  # Run
        try:
            ctx.driver.wait_first_frame(timeout_ms=20_000)
            toggles["frames_after_run"] += 1
        except Exception:
            pass
        toggles["count"] += 1

    chaos = ctx.chaos()
    chaos.register("toggle_stop_start", weight=1.0, fn=toggle_stop_start)
    chaos.run(duration_sec=30.0)

    ctx.measure("toggle_count", toggles["count"])
    ctx.measure("frames_landed_after_run", toggles["frames_after_run"])
    if toggles["count"] > 0 and toggles["frames_after_run"] < toggles["count"]:
        ctx.report.fail(
            f"only {toggles['frames_after_run']}/{toggles['count']} Run clicks produced a frame"
        )
