"""ScopeHarness — boot and tear down a Scope server subprocess per test.

Each test gets a fresh Scope subprocess with an isolated DAYDREAM_SCOPE_DIR
so onboarding state is truly virgin. Models are shared across tests via
DAYDREAM_SCOPE_MODELS_DIR to avoid re-downloading multi-GB weights.

Retry instrumentation is enabled (SCOPE_TEST_INSTRUMENTATION=1) so the
RetryProbe can observe counters via /api/v1/_debug/retry_stats.

Usage (fixture wraps this):

    harness = ScopeHarness(mode="local", workflow="mythical-creature",
                           tmp_dir=tmp_path, report_dir=report_dir)
    harness.start()
    try:
        ...drive the UI...
    finally:
        harness.stop()
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests


def _find_free_port() -> int:
    """Bind 0, read the assigned port, close. Race-prone but good enough."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@dataclass
class ScopeHarness:
    """Lifecycle manager for a per-test Scope subprocess."""

    mode: str = "local"  # "local" or "cloud"
    tmp_dir: Path | None = None
    report_dir: Path | None = None
    models_dir: Path | None = None  # shared across tests
    cloud_app_id: str | None = None
    extra_env: dict[str, str] = field(default_factory=dict)

    port: int = 0
    process: subprocess.Popen | None = None
    log_path: Path | None = None
    _log_fh = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self, boot_timeout: float = 120.0) -> None:
        """Spawn Scope, wait for /health to return 200."""
        if self.tmp_dir is None:
            raise RuntimeError("tmp_dir is required for isolation")
        if self.report_dir is None:
            raise RuntimeError("report_dir is required for log capture")

        self.port = _find_free_port()

        env = os.environ.copy()
        env["DAYDREAM_SCOPE_DIR"] = str(self.tmp_dir)
        env["SCOPE_TEST_INSTRUMENTATION"] = "1"
        # Disable pipelines that require GPU weights in CPU-only CI rings.
        env.setdefault("CUDA_VISIBLE_DEVICES", "")
        if self.models_dir is not None:
            env["DAYDREAM_SCOPE_MODELS_DIR"] = str(self.models_dir)
        if self.mode == "cloud":
            if not self.cloud_app_id:
                raise RuntimeError(
                    "cloud mode requires cloud_app_id (via SCOPE_CLOUD_APP_ID)"
                )
            env["SCOPE_CLOUD_APP_ID"] = self.cloud_app_id
        for k, v in self.extra_env.items():
            env[k] = v

        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.report_dir / "scope.log"
        self._log_fh = open(self.log_path, "w", buffering=1)

        cmd = [
            "uv",
            "run",
            "daydream-scope",
            "--port",
            str(self.port),
        ]
        self.process = subprocess.Popen(
            cmd,
            stdout=self._log_fh,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=_repo_root(),
        )

        deadline = time.time() + boot_timeout
        last_err: Exception | None = None
        while time.time() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"Scope exited during boot (rc={self.process.returncode}); "
                    f"see {self.log_path}"
                )
            try:
                r = requests.get(f"{self.base_url}/health", timeout=2.0)
                if r.status_code == 200:
                    return
            except Exception as e:
                last_err = e
            time.sleep(0.5)

        self.stop()
        raise RuntimeError(
            f"Scope did not become healthy within {boot_timeout}s on port "
            f"{self.port} (last error: {last_err}); see {self.log_path}"
        )

    def stop(self) -> None:
        """Terminate cleanly, escalate to kill after a brief grace period."""
        if self.process is None:
            return
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5.0)
        self.process = None
        if self._log_fh is not None:
            self._log_fh.close()
            self._log_fh = None


def _repo_root() -> Path:
    """Find the repo root (directory containing pyproject.toml) from this file."""
    p = Path(__file__).resolve()
    for candidate in [p, *p.parents]:
        if (candidate / "pyproject.toml").exists() and (candidate / "src").exists():
            return candidate
    # Fallback: current working dir.
    return Path.cwd()


def on_windows() -> bool:
    return sys.platform.startswith("win")
