"""UI multimodal — onboarding tour tooltip sits over the Run button.

The tour popover (``frontend/src/components/onboarding/TourPopover.tsx``)
points the user at the Run/Stop button when they first land on the
graph. If the popover mispositions (common regression when the Run
button moves or the portal's anchor calculation changes), the user
hits a dead-end: they don't know where the tour is pointing and the
"Next" button leads nowhere visually useful.

The testids ``tour-next`` and ``stream-run-stop`` still exist when the
popover is mispositioned. Only a visual check catches it. This test
screenshots the full page with the tour visible and asks the
multimodal reviewer whether the popover's arrow points at the
Run/Stop control.

Requires ``SCOPE_MULTIMODAL_EVAL=1`` + ``ANTHROPIC_API_KEY``.
"""

from __future__ import annotations

import pytest
from harness import flows, testids
from harness.scenario import scenario


@scenario(
    mode="local",
    workflow="local-passthrough",
    feature=("ui", "onboarding"),
    marks=(pytest.mark.multimodal,),
)
def test_tour_popover_points_at_run_button(ctx):
    """Complete onboarding; tour popover is visible; it points at Run."""
    # complete_onboarding_local lands on the graph with the tour popover
    # visible (the tour fires on first landing). We do NOT click
    # tour-next yet — we want the first-step popover up for the check.
    flows.complete_onboarding_local(ctx.driver, workflow_id="local-passthrough")

    # Wait for both the popover's Next button and the Run/Stop button
    # to be present; they anchor the visual check.
    ctx.wait(testids.TOUR_NEXT, timeout_ms=15_000)
    ctx.wait(testids.STREAM_RUN_STOP)

    full = ctx.screenshot(name="tour_popover_full.png")

    verdict = ctx.multimodal_check(
        full,
        question=(
            "Is the onboarding tour popover visibly positioned adjacent "
            "to — and pointing at (via arrow, highlight, or callout) — "
            "the Run/Stop button in the toolbar? The popover should NOT "
            "obscure the button itself, but its pointer/arrow should "
            "clearly indicate the button as the tour's current target."
        ),
        must_contain=[
            "tour popover is visible on screen",
            "a Run or Stop button is visible on screen",
            "popover arrow or highlight points at the Run/Stop button",
        ],
    )

    ctx.metadata("multimodal_status", verdict.status)
    ctx.metadata("multimodal_reasoning", verdict.reasoning)

    if verdict.status == "fail":
        ctx.report.fail(
            f"tour popover placement check failed: {verdict.reasoning} "
            f"(missing: {verdict.missing_required or 'n/a'})"
        )

    # Clean-up: advance or skip the tour so teardown can stop the stream.
    try:
        ctx.click(testids.TOUR_SKIP)
    except Exception:
        pass
