"""Onboarding smoke — local inference, passthrough workflow, first frame.

The "if this is red, ship nothing" gate. Drives real UI through Playwright
against a real Scope subprocess and asserts:
  - Onboarding completes without a single error toast
  - Stream starts and a video frame renders
  - RetryProbe sees zero retry/drop events at teardown
  - FailureWatcher sees zero unexpected session closes

This scenario uses the `local-passthrough` starter workflow so it runs
CPU-only and fits within the PR gate's 25-minute budget.

Note for future test authors: this file is deliberately minimal as a
reference implementation for the `@scenario` decorator. See
`product-tests/WRITING_TESTS.md` for the cookbook.
"""

from __future__ import annotations

from harness import baselines
from harness.scenario import scenario


@scenario(mode="local", workflow="local-passthrough", feature="onboarding")
def test_onboarding_local_passthrough(ctx):
    """Cold-start → pick local → decline telemetry → pick Camera Preview → Run → first frame."""
    ctx.complete_onboarding()

    first_ms = ctx.run_and_wait_first_frame(timeout_ms=90_000)
    baselines.check(
        ctx.report, "local", "passthrough", "first_frame_time_ms", int(first_ms)
    )
    # Default gates (zero retries, zero unexpected closes, zero UI errors) and
    # the clean stop are all applied by the @scenario teardown.
