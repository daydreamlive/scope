"""Chaos — <<ONE_LINE_DESCRIPTION_OF_THE_CHAOTIC_BEHAVIOR>>.

Why this test exists: simulates the real-world user pattern
"<<USER_PATTERN, e.g. 'switches inputs every few seconds while Scope is
still mid-load'>>". This pattern is exactly the kind of sequence that
exposes <<FAILURE_MODE, e.g. 'bad interactions between Scope-server and
remote-inference during teardown'>> — a failure mode that unit tests
cannot catch because it depends on timing and state transitions across
the whole stack.

Every action is deterministic under ``--chaos-seed``; ticks are logged
to ``timeline.jsonl`` so failures are reproducible.
"""

from __future__ import annotations

import pytest

from harness.scenario import scenario


@scenario(
    mode="local",
    workflow="local-passthrough",
    marks=(pytest.mark.chaos,),
)
def test_<<chaos_slug>>(ctx):
    """<<One-line description of the chaotic loop.>>"""
    ctx.complete_onboarding()
    ctx.run_and_wait_first_frame(timeout_ms=60_000)

    counters = {"fires": 0}

    def action_one():
        # Replace with the action you want to sample. Use ctx helpers,
        # NOT raw driver/page calls, so stop-events are properly attributed.
        ctx.toggle_run()
        counters["fires"] += 1

    chaos = ctx.chaos()
    chaos.register("<<action_one_name>>", weight=1.0, fn=action_one)
    # Register more actions with different weights if the chaos should
    # sample from a distribution:
    #   chaos.register("spam_param", weight=0.5, fn=lambda: ctx.set_parameter("k", "v"))

    chaos.run(duration_sec=30.0)
    ctx.measure("chaos_fires", counters["fires"])

    # Optional chaos-specific invariants (the default gates still apply):
    #   if counters["fires"] == 0:
    #       ctx.report.fail("chaos driver never fired — check the weight/duration")
