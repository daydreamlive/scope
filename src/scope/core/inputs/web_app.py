"""Web App input source implementation.

Uses Playwright for headless browser frame capture.
Install with: uv sync --extra web && uv run playwright install chromium
"""

from __future__ import annotations

import io
import logging
import threading
from typing import ClassVar

import numpy as np

from .interface import InputSource, InputSourceInfo

logger = logging.getLogger(__name__)


class WebAppInputSource(InputSource):
    """Input source that captures frames from a local HTML file or URL.

    Uses Playwright's headless Chromium browser to render and capture frames.
    Install the optional dependency with:
        uv sync --extra web && uv run playwright install chromium
    """

    source_id: ClassVar[str] = "web_app"
    source_name: ClassVar[str] = "Web App"
    source_description: ClassVar[str] = (
        "Capture frames from a local HTML file or URL using a headless browser. "
        "Requires Playwright: uv sync --extra web && uv run playwright install chromium"
    )

    def __init__(self):
        self._browser = None
        self._page = None
        self._playwright = None
        self._lock = threading.Lock()
        self._connected = False

    @classmethod
    def is_available(cls) -> bool:
        """Check if Playwright is installed."""
        try:
            import playwright  # noqa: F401

            return True
        except ImportError:
            return False

    def list_sources(self, timeout_ms: int = 5000) -> list[InputSourceInfo]:
        """No discovery needed — user provides URL or file path directly."""
        return []

    def connect(self, identifier: str) -> bool:
        """Connect to a web app by file path or URL.

        Args:
            identifier: Local file path or http(s):// URL.
        """
        try:
            from playwright.sync_api import sync_playwright

            self._playwright = sync_playwright().start()
            self._browser = self._playwright.chromium.launch(headless=True)
            self._page = self._browser.new_page()

            # Local files need file:// prefix
            if identifier.startswith(("http://", "https://", "file://")):
                url = identifier
            else:
                url = f"file://{identifier}"

            self._page.goto(url, wait_until="networkidle", timeout=30000)
            self._connected = True
            logger.info(f"WebApp connected to '{identifier}'")
            return True
        except Exception as e:
            logger.error(f"WebApp connect failed: {e}")
            self._connected = False
            self._cleanup()
            return False

    def receive_frame(self, timeout_ms: int = 100) -> np.ndarray | None:
        """Capture a screenshot and return it as an RGB numpy array."""
        if not self._connected or self._page is None:
            return None
        try:
            from PIL import Image

            with self._lock:
                png_bytes = self._page.screenshot(type="png", timeout=timeout_ms)

            img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
            return np.array(img, dtype=np.uint8)
        except Exception as e:
            logger.error(f"WebApp receive_frame failed: {e}")
            return None

    def disconnect(self):
        """Disconnect and release browser resources."""
        self._connected = False
        self._cleanup()

    def _cleanup(self):
        try:
            if self._page:
                self._page.close()
        except Exception:
            pass
        try:
            if self._browser:
                self._browser.close()
        except Exception:
            pass
        try:
            if self._playwright:
                self._playwright.stop()
        except Exception:
            pass
        self._page = None
        self._browser = None
        self._playwright = None
