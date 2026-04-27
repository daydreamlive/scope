"""PlaywrightDriver — thin convenience wrapper over the Playwright sync API.

Design principles:
- retries=0 is the default. If a click or wait fails, the test fails.
- Deterministic waits on data-testid; no sleep-based timing.
- Video + trace recording on by default so failures are debuggable.

Tests can still use the raw ``page`` fixture for things the wrapper doesn't
cover — the wrapper is for common patterns, not a replacement.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from playwright.sync_api import BrowserContext, Page
from playwright.sync_api import TimeoutError as PwTimeout

DEFAULT_TIMEOUT_MS = 15_000


@dataclass
class PlaywrightDriver:
    page: Page
    context: BrowserContext
    report_dir: Path

    def goto(self, url: str, *, timeout_ms: int = DEFAULT_TIMEOUT_MS) -> None:
        self.page.goto(url, timeout=timeout_ms, wait_until="domcontentloaded")

    def wait_testid(self, testid: str, *, timeout_ms: int = DEFAULT_TIMEOUT_MS) -> None:
        """Wait for a testid to be visible. Fail loud if it isn't."""
        self.page.wait_for_selector(
            f'[data-testid="{testid}"]', state="visible", timeout=timeout_ms
        )

    def click_testid(
        self, testid: str, *, timeout_ms: int = DEFAULT_TIMEOUT_MS
    ) -> None:
        self.wait_testid(testid, timeout_ms=timeout_ms)
        self.page.locator(f'[data-testid="{testid}"]').click()

    def click_all_tour_steps(self, *, max_steps: int = 20) -> None:
        """Walk the tour by clicking Next/Done until the popover disappears."""
        for _ in range(max_steps):
            try:
                self.page.wait_for_selector(
                    '[data-testid="tour-next"]', state="visible", timeout=2000
                )
            except PwTimeout:
                return
            self.page.locator('[data-testid="tour-next"]').click()
            # brief settle for position animation
            self.page.wait_for_timeout(150)

    def wait_first_frame(self, *, timeout_ms: int = 60_000) -> float:
        """Wait until the sink <video> is playing with frames. Returns ms elapsed."""
        start = time.monotonic()
        # First wait for the element to exist.
        self.page.wait_for_selector(
            '[data-testid="sink-video"]', state="attached", timeout=timeout_ms
        )
        # Then poll until it has non-zero video dimensions and currentTime > 0.
        deadline = time.monotonic() + timeout_ms / 1000.0
        while time.monotonic() < deadline:
            ready = self.page.evaluate(
                """() => {
                    const v = document.querySelector('[data-testid="sink-video"]');
                    if (!v) return false;
                    return v.readyState >= 2
                        && v.videoWidth > 0
                        && v.currentTime > 0;
                }"""
            )
            if ready:
                return (time.monotonic() - start) * 1000
            self.page.wait_for_timeout(100)
        raise PwTimeout(f"no video frame within {timeout_ms}ms")

    def error_toast_count(self) -> int:
        """Rough proxy: count of elements containing an error-ish class."""
        return self.page.locator(
            "[role='alert'], [data-testid*='error-'], .sonner-toast-error"
        ).count()

    def save_trace(self, name: str = "trace.zip") -> Path:
        path = self.report_dir / name
        self.context.tracing.stop(path=str(path))
        return path
