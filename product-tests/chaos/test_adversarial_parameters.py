"""Chaos — adversarial parameter payloads against /session/parameters.

Users paste things. Timelines serialize things. Third-party UIs ship
whatever JSON they feel like. The server MUST stay alive: reject junk
cleanly (4xx with a JSON error, or silently drop unknown keys), never
crash, never forcibly close the session, and never tick a retry
counter.

We explicitly do NOT assert 2xx here. Many of these payloads *should*
be rejected. The assertion is about the blast radius: a bad payload is
a user error, not a system failure.
"""

from __future__ import annotations

import pytest
import requests
from harness import flows, gates
from harness.driver import PlaywrightDriver
from harness.failure_watcher import FailureWatcher
from harness.report import TestReport
from harness.retry_probe import RetryProbe
from harness.scope_process import ScopeHarness

# Each payload goes through requests.post as-is (or encoded to JSON).
# The harness POSTs each and records (status, exception).
ADVERSARIAL_PAYLOADS: list[tuple[str, object]] = [
    ("empty_dict", {}),
    ("null_value", {"k": None}),
    ("huge_string_1mb", {"prompt": "A" * 1_000_000}),
    ("deeply_nested", {"a": {"b": {"c": {"d": {"e": {"f": {"g": 1}}}}}}}),
    ("unicode_soup", {"k": "🔥" * 1000 + "\u202e" + "مرحبا" + "\x00"}),
    ("wrong_type_list", {"prompt_interpolation_method": ["linear", "nearest"]}),
    ("wrong_type_num", {"prompt_interpolation_method": 42}),
    ("bool_where_str", {"prompt_interpolation_method": True}),
    ("negative_float", {"some_float": -1e308}),
    ("nan_like", {"some_float": "NaN"}),
    ("inf_like", {"some_float": "Infinity"}),
    ("special_keys", {"": "empty key", " ": "space key", "__proto__": "pollute"}),
    ("control_chars", {"k": "line\nbreak\tand\x07bell"}),
    ("sql_injection", {"k": "'; DROP TABLE users; --"}),
    ("path_traversal", {"k": "../../../etc/passwd"}),
    ("html_like", {"k": "<script>alert(1)</script>"}),
    ("very_long_key", {"X" * 10_000: "v"}),
    ("many_keys", {f"k{i}": i for i in range(1000)}),
]


@pytest.mark.chaos
def test_adversarial_parameters_local(
    scope_harness: ScopeHarness,
    driver: PlaywrightDriver,
    retry_probe: RetryProbe,
    failure_watcher: FailureWatcher,
    report: TestReport,
):
    """Fire each adversarial payload at /session/parameters; session must survive."""
    report.metadata["workflow"] = "local-passthrough"

    flows.complete_onboarding_local(driver, workflow_id="local-passthrough")
    first_ms = flows.start_stream_and_wait_first_frame(driver, timeout_ms=60_000)
    report.measure("first_frame_time_ms", int(first_ms))

    base = scope_harness.base_url
    endpoint = f"{base}/api/v1/session/parameters"

    results: dict[str, int | str] = {}
    crashes: list[str] = []

    for name, payload in ADVERSARIAL_PAYLOADS:
        try:
            r = requests.post(endpoint, json=payload, timeout=5.0)
            results[name] = r.status_code
            # 5xx is a server bug; 4xx is a user error (fine).
            if r.status_code >= 500:
                crashes.append(f"{name}: {r.status_code} body={r.text[:120]}")
        except requests.exceptions.Timeout:
            crashes.append(f"{name}: TIMEOUT")
            results[name] = "timeout"
        except Exception as e:
            crashes.append(f"{name}: {type(e).__name__}: {e}")
            results[name] = f"err:{type(e).__name__}"

    # Server must still be healthy.
    try:
        health = requests.get(f"{base}/health", timeout=3.0)
        assert health.status_code == 200, f"health returned {health.status_code}"
    except Exception as e:
        crashes.append(f"health check failed: {e}")

    # Session must still be alive — prove it by sending a well-formed param.
    try:
        r = requests.post(endpoint, json={"k": "recovery-check"}, timeout=3.0)
        if r.status_code >= 400:
            crashes.append(f"sane payload rejected post-adversarial: {r.status_code}")
    except Exception as e:
        crashes.append(f"sane payload errored post-adversarial: {e}")

    # And a frame must still be flowing.
    try:
        driver.wait_first_frame(timeout_ms=15_000)
    except Exception:
        crashes.append("video frame did not recover after adversarial payloads")

    report.metadata["adversarial_results"] = results
    report.measure("adversarial_5xx_or_crash", len(crashes))
    if crashes:
        report.fail(f"adversarial payloads caused {len(crashes)} failures: {crashes}")

    gates.enforce_all_gates(report, retry_probe, failure_watcher, driver)

    failure_watcher.mark_initiated_stop()
    flows.stop_stream(driver)

    assert report.passed, f"Hard fails: {report.hard_fails}"
