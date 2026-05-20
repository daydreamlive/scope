"""<<SCENARIO_NAME>> — <<one-line user journey being validated>>.

This lives under scenarios/ because it's a happy-path guard that must
stay green on every PR. Keep the body tight: one user journey, one
clear signal. Anything more specific belongs in regression/.

Success (in addition to the default gates — zero retries, zero
unexpected closes, zero UI errors):
  - <<explicit success signal 1>>
  - <<explicit success signal 2>>
"""

from __future__ import annotations

from harness import baselines
from harness.scenario import scenario


@scenario(
    mode="local",
    workflow="local-passthrough",
)
def test_<<scenario_slug>>(ctx):
    """<<One-line test description shown in pytest output.>>"""
    ctx.complete_onboarding()

    first_ms = ctx.run_and_wait_first_frame(timeout_ms=90_000)

    # Enforce the baseline for this scenario. Grow the baselines file with
    # a representative p95 from a clean run, not an optimistic best case.
    baselines.check(
        ctx.report, ctx.mode, "<<scenario_slug>>", "first_frame_time_ms", int(first_ms)
    )

    # Add scenario-specific measurements here. The decorator auto-populates
    # retry_count, unexpected_close_count, ui_error_events.
    #   ctx.measure("my_dimension", value)
