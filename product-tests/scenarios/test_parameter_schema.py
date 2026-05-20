"""Parameter schema coverage — every declared param round-trips cleanly.

Scope exposes pipeline parameter descriptors at ``/api/v1/pipelines/schemas``
as JSON Schema. This test takes that schema as source-of-truth and:

  1. Sends each parameter at its declared default — must round-trip.
  2. Sends numeric params at both bounds (min, max when declared).
  3. Sends enum params at each allowed value.
  4. Sends out-of-range values — server must return 4xx, not 5xx.

What this catches: a recent refactor broke enum validation, someone
tightened a min without updating the schema, a pipeline now 500s on a
valid-per-schema value. The schema is the contract with the frontend;
this test enforces it.
"""

from __future__ import annotations

import time
from typing import Any

import pytest
import requests
from harness import flows
from harness.failure_watcher import FailureWatcher
from harness.report import TestReport
from harness.retry_probe import RetryProbe
from harness.scope_process import ScopeHarness


def _start_passthrough(base_url: str) -> None:
    # Direct-HTTP tests must load the pipeline themselves — UI-driven
    # tests get this via onboarding. Without it, FrameProcessor fails
    # with "Pipeline passthrough not loaded".
    flows.http_load_pipeline_and_wait(base_url, ["passthrough"])
    r = requests.post(
        f"{base_url}/api/v1/session/start",
        json={"pipeline_id": "passthrough", "input_mode": "camera"},
        timeout=20.0,
    )
    assert r.status_code == 200, f"session/start failed: {r.status_code} {r.text[:200]}"


def _get_schema(base_url: str, pipeline_id: str) -> dict[str, Any]:
    r = requests.get(f"{base_url}/api/v1/pipelines/schemas", timeout=10.0)
    r.raise_for_status()
    return (
        r.json()
        .get("pipelines", {})
        .get(pipeline_id, {})
        .get("config_schema", {})
        .get("properties", {})
    )


def _extract_type(prop: dict[str, Any]) -> str | None:
    """Handle the Pydantic-style anyOf wrappings."""
    if "type" in prop:
        return prop["type"]
    for any_of in prop.get("anyOf", []):
        t = any_of.get("type")
        if t and t != "null":
            return t
    return None


def _extract_enum(prop: dict[str, Any]) -> list[str] | None:
    if "enum" in prop:
        return prop["enum"]
    for any_of in prop.get("anyOf", []):
        if "enum" in any_of:
            return any_of["enum"]
    return None


def _extract_bounds(prop: dict[str, Any]) -> tuple[float | None, float | None]:
    lo = prop.get("minimum")
    hi = prop.get("maximum")
    if lo is None and hi is None:
        for any_of in prop.get("anyOf", []):
            if "minimum" in any_of:
                lo = any_of["minimum"]
            if "maximum" in any_of:
                hi = any_of["maximum"]
    return lo, hi


def _post_param(base_url: str, payload: dict[str, Any]) -> int:
    r = requests.post(
        f"{base_url}/api/v1/session/parameters", json=payload, timeout=5.0
    )
    return r.status_code


@pytest.mark.params
def test_parameter_schema_roundtrip_passthrough(
    scope_harness: ScopeHarness,
    retry_probe: RetryProbe,
    failure_watcher: FailureWatcher,
    report: TestReport,
):
    base = scope_harness.base_url
    _start_passthrough(base)
    time.sleep(1.0)  # let the frame processor spin up

    schema = _get_schema(base, "passthrough")
    report.measure("schema_param_count", len(schema))
    if not schema:
        report.fail("/api/v1/pipelines/schemas returned empty schema for passthrough")

    valid_failures: list[str] = []
    out_of_range_5xx: list[str] = []
    enum_failures: list[str] = []

    for name, prop in schema.items():
        # -- 1. Default value must round-trip.
        if "default" in prop and prop["default"] is not None:
            code = _post_param(base, {name: prop["default"]})
            if code >= 400:
                valid_failures.append(f"{name}@default -> {code}")

        t = _extract_type(prop)
        enum_vals = _extract_enum(prop)
        lo, hi = _extract_bounds(prop)

        # -- 2. Enum: every allowed value must be accepted.
        if enum_vals:
            for v in enum_vals:
                code = _post_param(base, {name: v})
                if code >= 400:
                    enum_failures.append(f"{name}={v} -> {code}")

        # -- 3. Numeric bounds: accept min and max.
        if t in {"number", "integer"}:
            for v in (lo, hi):
                if v is None:
                    continue
                code = _post_param(base, {name: v})
                if code >= 400:
                    valid_failures.append(f"{name}@{v} -> {code}")

            # -- 4. Out-of-range: must be rejected or silently ignored — but
            # MUST NOT 5xx. 4xx = fine (server validated), 2xx = also fine
            # (server accepted lax input).
            if lo is not None:
                below = lo - 1 if t == "integer" else lo - 0.5
                code = _post_param(base, {name: below})
                if code >= 500:
                    out_of_range_5xx.append(f"{name}={below} -> {code}")
            if hi is not None:
                above = hi + 1 if t == "integer" else hi + 0.5
                code = _post_param(base, {name: above})
                if code >= 500:
                    out_of_range_5xx.append(f"{name}={above} -> {code}")

    report.measure("valid_param_failures", len(valid_failures))
    report.measure("enum_param_failures", len(enum_failures))
    report.measure("out_of_range_5xx", len(out_of_range_5xx))
    report.metadata["valid_failures_samples"] = valid_failures[:10]
    report.metadata["enum_failures_samples"] = enum_failures[:10]
    report.metadata["out_of_range_5xx_samples"] = out_of_range_5xx[:10]

    if valid_failures:
        report.fail(f"schema-valid params rejected: {valid_failures[:5]}")
    if enum_failures:
        report.fail(f"enum values rejected: {enum_failures[:5]}")
    if out_of_range_5xx:
        report.fail(
            f"out-of-range params produced 5xx (should be 4xx): {out_of_range_5xx[:5]}"
        )

    failure_watcher.mark_initiated_stop()
    requests.post(f"{base}/api/v1/session/stop", timeout=10.0)

    from harness import gates

    gates.enforce_zero_retries(report, retry_probe)
    gates.enforce_zero_unexpected_closes(report, failure_watcher)

    assert report.passed, f"Hard fails: {report.hard_fails}"
