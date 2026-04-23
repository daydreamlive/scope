"""Chaos — WiFi drops mid-stream (browser flips to offline, then back).

Playwright's ``context.set_offline(True)`` sets ``navigator.onLine=false``
and blocks network requests originating from the browser. For localhost
WebRTC the media path itself is unaffected, but the frontend's online
handlers, any pending fetch/XHR to the Scope HTTP API, and any
reconnect logic that keys off ``navigator.onLine`` all fire.

Real-world analog: user's router reboots, laptop switches WiFi networks,
ethernet cable knocked loose. The contract: when online comes back,
the stream keeps going or recovers without a retry counter tick.
"""

from __future__ import annotations

import time

import pytest
from harness import flows, gates
from harness.driver import PlaywrightDriver
from harness.failure_watcher import FailureWatcher
from harness.report import TestReport
from harness.retry_probe import RetryProbe
from harness.scope_process import ScopeHarness


@pytest.mark.chaos
def test_network_offline_cycle_local(
    scope_harness: ScopeHarness,
    driver: PlaywrightDriver,
    retry_probe: RetryProbe,
    failure_watcher: FailureWatcher,
    report: TestReport,
):
    """Run a stream; go offline, wait, come back online. 3 cycles."""
    report.metadata["workflow"] = "local-passthrough"

    flows.complete_onboarding_local(driver, workflow_id="local-passthrough")
    first_ms = flows.start_stream_and_wait_first_frame(driver, timeout_ms=60_000)
    report.measure("first_frame_time_ms", int(first_ms))

    t_before = driver.page.evaluate(
        """() => document.querySelector('[data-testid="sink-video"]')?.currentTime ?? 0"""
    )

    CYCLES = 3
    for i in range(CYCLES):
        # Flip offline. The browser dispatches an 'offline' event, and any
        # pending fetches either fail or queue depending on the browser.
        driver.context.set_offline(True)
        report.metadata[f"cycle_{i}_offline_at_sec"] = round(time.monotonic(), 2)
        time.sleep(3.0)
        driver.context.set_offline(False)
        report.metadata[f"cycle_{i}_online_at_sec"] = round(time.monotonic(), 2)
        time.sleep(3.0)

    # Stream must still advance. If the frontend's online handler triggered
    # a reconnect cascade that tore down and rebuilt the session, the retry
    # counters will tick and the gate below will catch it.
    t_after = driver.page.evaluate(
        """() => document.querySelector('[data-testid="sink-video"]')?.currentTime ?? 0"""
    )
    advance = float(t_after) - float(t_before)
    report.measure("video_currenttime_advance_sec", int(advance * 1000) / 1000.0)
    report.measure("offline_cycles", CYCLES)

    # We spent ~18s cycling. Expect real advance; <5s means the video froze.
    if advance < 5.0:
        report.fail(
            f"video.currentTime advanced only {advance:.2f}s across "
            f"{CYCLES} offline/online cycles — stream likely froze"
        )

    gates.enforce_all_gates(report, retry_probe, failure_watcher, driver)

    failure_watcher.mark_initiated_stop()
    flows.stop_stream(driver)

    assert report.passed, f"Hard fails: {report.hard_fails}"
