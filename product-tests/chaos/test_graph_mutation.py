"""Chaos — concurrent graph mutations while a UI session is live.

The user clicks Run (UI-owned session). Then a scripted client starts
mutating: POST a new graph, POST a malformed graph, POST a graph with a
dangling edge. The server must either (a) apply cleanly, (b) reject
cleanly with a 4xx, or (c) keep the current session intact.

Not acceptable: 5xx, crash, or an in-flight graph swap that leaves the
session in a broken state where neither frames flow nor stop works.
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


def _valid_graph(pipeline_id: str) -> dict:
    return {
        "input_mode": "camera",
        "graph": {
            "nodes": [
                {"id": "input", "type": "source", "source_mode": "camera"},
                {"id": "pipe", "type": "pipeline", "pipeline_id": pipeline_id},
                {"id": "output", "type": "sink"},
            ],
            "edges": [
                {
                    "from": "input",
                    "from_port": "video",
                    "to_node": "pipe",
                    "to_port": "video",
                    "kind": "stream",
                },
                {
                    "from": "pipe",
                    "from_port": "video",
                    "to_node": "output",
                    "to_port": "video",
                    "kind": "stream",
                },
            ],
        },
    }


MUTATIONS: list[tuple[str, dict]] = [
    ("swap_to_gray", _valid_graph("gray")),
    ("swap_to_passthrough", _valid_graph("passthrough")),
    (
        "dangling_edge",
        {
            "input_mode": "camera",
            "graph": {
                "nodes": [
                    {"id": "input", "type": "source", "source_mode": "camera"},
                    {"id": "pipe", "type": "pipeline", "pipeline_id": "passthrough"},
                ],
                "edges": [
                    {
                        "from": "pipe",
                        "from_port": "video",
                        "to_node": "nonexistent",
                        "to_port": "video",
                        "kind": "stream",
                    },
                ],
            },
        },
    ),
    (
        "duplicate_node_id",
        {
            "input_mode": "camera",
            "graph": {
                "nodes": [
                    {"id": "dup", "type": "source", "source_mode": "camera"},
                    {"id": "dup", "type": "pipeline", "pipeline_id": "passthrough"},
                    {"id": "output", "type": "sink"},
                ],
                "edges": [],
            },
        },
    ),
    ("empty_graph", {"input_mode": "camera", "graph": {"nodes": [], "edges": []}}),
    ("unknown_pipeline", _valid_graph("definitely-does-not-exist-9000")),
    (
        "cyclic_graph",
        {
            "input_mode": "camera",
            "graph": {
                "nodes": [
                    {"id": "a", "type": "pipeline", "pipeline_id": "passthrough"},
                    {"id": "b", "type": "pipeline", "pipeline_id": "passthrough"},
                ],
                "edges": [
                    {
                        "from": "a",
                        "from_port": "video",
                        "to_node": "b",
                        "to_port": "video",
                        "kind": "stream",
                    },
                    {
                        "from": "b",
                        "from_port": "video",
                        "to_node": "a",
                        "to_port": "video",
                        "kind": "stream",
                    },
                ],
            },
        },
    ),
]


@pytest.mark.chaos
@pytest.mark.graph
def test_graph_mutation_local(
    scope_harness: ScopeHarness,
    driver: PlaywrightDriver,
    retry_probe: RetryProbe,
    failure_watcher: FailureWatcher,
    report: TestReport,
):
    """Stream, then submit 7 varied graphs via HTTP; server must not crash."""
    report.metadata["workflow"] = "local-passthrough"

    flows.complete_onboarding_local(driver, workflow_id="local-passthrough")
    first_ms = flows.start_stream_and_wait_first_frame(driver, timeout_ms=60_000)
    report.measure("first_frame_time_ms", int(first_ms))

    base = scope_harness.base_url
    results: dict[str, int | str] = {}
    crashes: list[str] = []

    for name, body in MUTATIONS:
        try:
            r = requests.post(f"{base}/api/v1/session/start", json=body, timeout=10.0)
            results[name] = r.status_code
            if r.status_code >= 500:
                crashes.append(f"{name}: {r.status_code} body={r.text[:160]}")
        except requests.exceptions.Timeout:
            crashes.append(f"{name}: TIMEOUT")
            results[name] = "timeout"
        except Exception as e:
            crashes.append(f"{name}: {type(e).__name__}: {e}")
            results[name] = f"err:{type(e).__name__}"

    report.metadata["mutation_results"] = results
    report.measure("mutation_5xx_or_crash", len(crashes))
    if crashes:
        report.fail(f"graph mutations caused {len(crashes)} failures: {crashes}")

    # Server must still be healthy.
    try:
        h = requests.get(f"{base}/health", timeout=3.0)
        assert h.status_code == 200
    except Exception as e:
        report.fail(f"health check failed post-mutation: {e}")

    # And a sane graph must still be acceptable.
    try:
        r = requests.post(
            f"{base}/api/v1/session/start",
            json=_valid_graph("passthrough"),
            timeout=10.0,
        )
        if r.status_code >= 500:
            report.fail(
                f"valid graph rejected post-chaos: {r.status_code} {r.text[:160]}"
            )
    except Exception as e:
        report.fail(f"valid graph errored post-chaos: {e}")

    gates.enforce_all_gates(report, retry_probe, failure_watcher, driver)

    failure_watcher.mark_initiated_stop()
    flows.stop_stream(driver)

    assert report.passed, f"Hard fails: {report.hard_fails}"
