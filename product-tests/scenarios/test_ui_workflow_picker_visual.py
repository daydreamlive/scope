"""UI multimodal — workflow picker shows three distinct starter cards.

Reference implementation of the "look at it like a human would" pattern
for UI correctness. Reaches the workflow picker step of onboarding,
screenshots the picker's testid-scoped container, and asks Claude with
vision whether three distinct workflow cards are present with
thumbnails.

This catches the bug class that passes every machine check: the
existing ``workflow-card-*`` testids still exist, the component still
renders, but CSS regressions clip the third card, the thumbnail image
fails to load, or the cards collapse on narrow viewports. None of those
fail a DOM query. All of them fail a "does this look right?" review.

Requires ``SCOPE_MULTIMODAL_EVAL=1`` + ``ANTHROPIC_API_KEY``. Without
them the verdict is ``uncertain`` with a "disabled" reason — the test
still runs and captures the screenshot artifact, it just doesn't assert.
"""

from __future__ import annotations

import pytest
from harness import testids
from harness.scenario import scenario


@scenario(
    mode="local",
    workflow="local-passthrough",
    feature="ui",
    marks=(pytest.mark.multimodal,),
)
def test_workflow_picker_shows_three_cards(ctx):
    """Onboarding → workflow picker step; assert picker shows 3 cards."""
    # Advance onboarding up to the workflow picker step, then stop so we
    # can screenshot the picker itself. We reuse the inference-mode +
    # telemetry portions of ``complete_onboarding_local`` inline so we
    # don't blow past the picker.
    ctx.driver.goto(ctx.base_url)
    ctx.wait(testids.inference_mode("local"))
    ctx.click(testids.inference_mode("local"))
    ctx.click(testids.INFERENCE_MODE_CONTINUE)
    ctx.wait(testids.TELEMETRY_DECLINE)
    ctx.click(testids.TELEMETRY_DECLINE)

    # Now on the workflow picker. Screenshot the full page so the
    # multimodal reviewer sees surrounding layout (captures layout bugs
    # like "third card overflows the viewport").
    ctx.wait(testids.workflow_card("local-passthrough"), timeout_ms=15_000)
    full = ctx.screenshot(name="workflow_picker_full.png")
    # Also capture a scoped shot of one card for fine-grained evidence
    # during triage. Any of the three works as a reference.
    card = ctx.screenshot_testid(
        testids.workflow_card("local-passthrough"),
        name="workflow_card_local.png",
    )

    verdict = ctx.multimodal_check(
        [full, card],
        question=(
            "Does the workflow picker UI show exactly three distinct "
            "starter workflow cards, each with a visible thumbnail image "
            "and a readable title, all fully rendered within the viewport "
            "(not clipped, not overlapping, not collapsed)?"
        ),
        must_contain=[
            "three workflow cards arranged in a row",
            "each card has a visible thumbnail image",
            "each card has a readable title",
        ],
    )

    ctx.metadata("multimodal_status", verdict.status)
    ctx.metadata("multimodal_reasoning", verdict.reasoning)
    if verdict.observations:
        ctx.metadata("multimodal_observations", " | ".join(verdict.observations))

    if verdict.status == "fail":
        ctx.report.fail(
            f"multimodal UI check failed: {verdict.reasoning} "
            f"(missing: {verdict.missing_required or 'n/a'})"
        )
    # status == "uncertain" is a skip (usually because eval is disabled);
    # status == "pass" falls through to the auto-teardown gates.
