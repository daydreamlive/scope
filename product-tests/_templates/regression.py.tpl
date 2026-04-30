"""Regression for #<<PR_OR_ISSUE>>: <<ONE_LINE_SYMPTOM>>.

Context (fill this in — future maintainers will thank you):
- What the user did:   <<reproduction steps in plain English>>
- What should happen:  <<expected outcome>>
- What did happen:     <<observed symptom, incl. any log line patterns>>
- Root cause:          <<one-line root cause from the PR>>
- Fix:                 <<PR title / brief description of the fix>>

Keep this file narrow: one bug, one repro. If the repro needs a different
mode / workflow, write a second regression file.
"""

from __future__ import annotations

from harness.scenario import scenario


@scenario(
    mode="local",  # or "cloud" — keep PR-ring tests on "local" unless the bug is cloud-specific
    workflow="local-passthrough",  # override if the bug only reproduces on a specific workflow
)
def test_pr_<<PR_OR_ISSUE>>_<<SHORT_SLUG>>(ctx):
    """Reproduces the pre-fix failure; asserts the gates stay green on the fix."""
    ctx.complete_onboarding()
    ctx.run_and_wait_first_frame()

    # -- reproduction steps below --
    # Replace this block with the precise actions that reproduced the bug.
    # Common building blocks:
    #   ctx.set_parameter(name, value)       -> POST /api/v1/session/parameters
    #   ctx.click("stream-run-stop")         -> click a data-testid
    #   ctx.toggle_run()                     -> rapid stop/run toggle
    #   ctx.sleep(ms)                        -> deterministic settle
    #   ctx.metrics()                        -> read session metrics
    # See WRITING_TESTS.md for the full ctx surface.
    pass

    # No explicit assertion needed — the @scenario teardown fails the test if:
    #   * any retry counter ticked (/api/v1/_debug/retry_stats)
    #   * an unexpected session_close fired (not preceded by an initiated stop)
    #   * a UI error toast appeared
    # If this regression needs a more specific check, add it here:
    #   assert ctx.metrics()["sessions"], "session went dark unexpectedly"
