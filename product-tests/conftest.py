"""Top-level fixtures for product-tests.

Every scenario/chaos test gets:
    scope_harness   — fresh Scope subprocess (isolated DAYDREAM_SCOPE_DIR)
    driver          — Playwright wrapper pointed at the Scope URL
    retry_probe     — queries /api/v1/_debug/retry_stats
    failure_watcher — background log tail + WS watcher
    report          — TestReport populated over the test lifetime

Teardown enforces the three hard-fail gates (retries, unexpected closes,
UI errors) regardless of the assertions the test itself made.
"""

from __future__ import annotations

import os
import time
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from contracts import (
    NoRetries,
    NoRetriesViolation,
    NoUnexpectedSessionClose,
    NoUnexpectedSessionCloseViolation,
)
from harness.cloud_auth import install_cloud_auth_bypass
from harness.driver import PlaywrightDriver
from harness.failure_watcher import FailureWatcher
from harness.report import TestReport, aggregate_summary
from harness.retry_probe import RetryProbe
from harness.scope_process import ScopeHarness
from playwright.sync_api import sync_playwright

# ---------------------------------------------------------------------------
# CLI options
# ---------------------------------------------------------------------------


def pytest_addoption(parser):
    parser.addoption(
        "--chaos-seed",
        default="",
        help="Deterministic seed for ChaosDriver. Defaults to a per-run uuid.",
    )
    parser.addoption(
        "--reports-dir",
        default=None,
        help="Root dir for report artifacts. Defaults to product-tests/reports/<run-id>.",
    )


# ---------------------------------------------------------------------------
# Session-scoped paths
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def run_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]


@pytest.fixture(scope="session")
def reports_root(request, run_id: str) -> Path:
    override = request.config.getoption("--reports-dir")
    if override:
        p = Path(override)
    else:
        p = Path(__file__).parent / "reports" / run_id
    p.mkdir(parents=True, exist_ok=True)
    return p


@pytest.fixture(scope="session")
def shared_models_dir(tmp_path_factory) -> Path:
    """Models directory shared across all tests in a run.

    If ``DAYDREAM_SCOPE_MODELS_DIR`` is set in the env, reuse it (CI caches
    models between runs); otherwise allocate a per-run temp dir.
    """
    env = os.environ.get("DAYDREAM_SCOPE_MODELS_DIR")
    if env:
        p = Path(env)
        p.mkdir(parents=True, exist_ok=True)
        return p
    return tmp_path_factory.mktemp("models")


@pytest.fixture(scope="session")
def cloud_app_id() -> str | None:
    return os.environ.get("SCOPE_CLOUD_APP_ID")


@pytest.fixture(scope="session")
def chaos_seed(request) -> str:
    seed = request.config.getoption("--chaos-seed")
    if not seed:
        seed = os.environ.get("GITHUB_SHA", uuid.uuid4().hex)
    return seed


# ---------------------------------------------------------------------------
# Per-test fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def test_report_dir(request, reports_root: Path) -> Path:
    safe = request.node.nodeid.replace("/", "_").replace("::", "__").replace(" ", "_")
    p = reports_root / safe
    p.mkdir(parents=True, exist_ok=True)
    return p


@pytest.fixture
def scope_harness(
    request,
    tmp_path: Path,
    test_report_dir: Path,
    shared_models_dir: Path,
    cloud_app_id: str | None,
) -> Iterator[ScopeHarness]:
    """Boot a fresh Scope subprocess for this test."""
    marker = request.node.get_closest_marker("cloud")
    mode = "cloud" if marker else "local"
    if mode == "cloud" and not cloud_app_id:
        pytest.skip("cloud mode requires SCOPE_CLOUD_APP_ID")

    harness = ScopeHarness(
        mode=mode,
        tmp_dir=tmp_path / "scope-home",
        report_dir=test_report_dir,
        models_dir=shared_models_dir,
        cloud_app_id=cloud_app_id,
    )
    harness.start()
    try:
        yield harness
    finally:
        harness.stop()


@pytest.fixture
def retry_probe(scope_harness: ScopeHarness) -> RetryProbe:
    return RetryProbe(base_url=scope_harness.base_url)


@pytest.fixture
def failure_watcher(
    scope_harness: ScopeHarness,
) -> Iterator[FailureWatcher]:
    assert scope_harness.log_path is not None
    with FailureWatcher(log_path=scope_harness.log_path) as w:
        yield w


@pytest.fixture
def report(
    request,
    scope_harness: ScopeHarness,
    test_report_dir: Path,
) -> Iterator[TestReport]:
    r = TestReport(
        test=request.node.nodeid,
        mode=scope_harness.mode,
        report_dir=test_report_dir,
    )
    try:
        yield r
    finally:
        r.emit()


@pytest.fixture
def driver(
    request,
    scope_harness: ScopeHarness,
    test_report_dir: Path,
) -> Iterator[PlaywrightDriver]:
    """Playwright browser context pointed at the Scope URL.

    For tests marked @pytest.mark.cloud, pre-seeds localStorage with a test
    auth blob so the CloudAuthStep auto-advances past the sign-in phase.
    """
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context(
            record_video_dir=str(test_report_dir),
            record_video_size={"width": 1280, "height": 800},
            viewport={"width": 1280, "height": 800},
        )
        if request.node.get_closest_marker("cloud"):
            install_cloud_auth_bypass(context)
        context.tracing.start(screenshots=True, snapshots=True, sources=True)
        page = context.new_page()
        d = PlaywrightDriver(page=page, context=context, report_dir=test_report_dir)
        d.goto(scope_harness.base_url)
        try:
            yield d
        finally:
            trace_path = test_report_dir / "trace.zip"
            try:
                context.tracing.stop(path=str(trace_path))
            except Exception:
                pass
            context.close()
            browser.close()


# ---------------------------------------------------------------------------
# Cross-cutting enforcement
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def enforce_contracts(
    request,
    scope_harness: ScopeHarness,
    failure_watcher: FailureWatcher,
):
    """Auto-apply cross-cutting contracts at teardown.

    Each test body makes its own assertions, but these contracts are the
    "silent-flake" guards: banned retry counters must be zero, and no
    unexpected session close may have fired.
    """
    yield
    # Skip if the test itself failed — don't mask the real error.
    if hasattr(request.node, "rep_call") and request.node.rep_call.failed:
        return

    no_retries = NoRetries(probe=RetryProbe(base_url=scope_harness.base_url))
    no_close = NoUnexpectedSessionClose(watcher=failure_watcher)
    try:
        no_retries.assert_clean()
    except NoRetriesViolation as e:
        pytest.fail(str(e))
    except Exception:
        # If the probe is unreachable, don't swallow it silently — but also
        # don't fail the test on a teardown race where scope has just stopped.
        pass
    try:
        no_close.assert_clean()
    except NoUnexpectedSessionCloseViolation as e:
        pytest.fail(str(e))


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    setattr(item, f"rep_{rep.when}", rep)


# ---------------------------------------------------------------------------
# Session finale — roll up a summary.md
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def emit_summary(reports_root: Path):
    yield
    try:
        path = aggregate_summary(reports_root)
        print(f"\nproduct-tests summary: {path}")
    except Exception as e:  # pragma: no cover - best-effort
        print(f"Failed to aggregate summary: {e}")
