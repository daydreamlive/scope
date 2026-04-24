"""`@scenario` — the low-friction way to write a product-test.

The problem it solves: a hand-rolled scenario test requires five fixture
declarations, five harness imports, knowledge that ``failure_watcher.
mark_initiated_stop()`` must be called before every graceful stop, and
a manual ``gates.enforce_all_gates()`` call at teardown. That's a lot of
surface area to get right for what should be a one-screen regression repro.

Instead, decorate a function that takes one argument (``ctx``):

    from harness.scenario import scenario

    @scenario(mode="local", workflow="local-passthrough")
    def test_pr_1234_parameter_spam_crash(ctx):
        '''Regression for #1234: rapid parameter updates crashed the session.'''
        ctx.complete_onboarding()
        ctx.run_and_wait_first_frame()
        for _ in range(200):
            ctx.set_parameter("__prompt", "a test prompt")
        # teardown auto-asserts: zero retries, zero unexpected closes,
        # zero UI errors, stream stopped cleanly.

What the decorator does for you:
  1. Pulls in the canonical fixtures (scope_harness, driver, retry_probe,
     failure_watcher, report, test_report_dir).
  2. Applies the ``cloud`` marker when ``mode="cloud"`` so the fixture
     layer wires cloud auth + skips when SCOPE_CLOUD_APP_ID is unset.
  3. On teardown (even if the body raises): marks an initiated stop,
     stops the stream if still running, populates dimensions via
     ``gates.enforce_all_gates``, and asserts ``report.passed``.

``ctx`` exposes a small set of high-level actions (the 80% of what every
test needs). For anything else, reach through to ``ctx.driver``,
``ctx.harness``, or ``ctx.page`` directly — the wrapper is a shortcut,
not a cage.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import requests

from . import flows, gates
from .chaos import ChaosDriver
from .driver import PlaywrightDriver
from .failure_watcher import FailureWatcher
from .report import TestReport
from .retry_probe import RetryProbe
from .scope_process import ScopeHarness


@dataclass
class ScenarioContext:
    """High-level driver for product-tests.

    Each attribute is a safe default; reach through to the underlying
    ``driver`` / ``harness`` / ``page`` when you need something this
    wrapper doesn't cover.
    """

    harness: ScopeHarness
    driver: PlaywrightDriver
    retry_probe: RetryProbe
    failure_watcher: FailureWatcher
    report: TestReport
    test_report_dir: Path
    chaos_seed: str
    workflow: str | None = None
    _streaming: bool = field(default=False, init=False)
    _torn_down: bool = field(default=False, init=False)

    # ------------------------------------------------------------------
    # Convenience accessors — read-only passthroughs.
    # ------------------------------------------------------------------

    @property
    def page(self):
        """The Playwright ``Page``. For raw UI access."""
        return self.driver.page

    @property
    def base_url(self) -> str:
        """Scope server base URL (http://127.0.0.1:<port>)."""
        return self.harness.base_url

    @property
    def mode(self) -> str:
        """``"local"`` or ``"cloud"``."""
        return self.harness.mode

    # ------------------------------------------------------------------
    # Report shortcuts.
    # ------------------------------------------------------------------

    def measure(self, name: str, value: float | int) -> None:
        """Record a dimension on the report."""
        self.report.measure(name, value)

    def metadata(self, key: str, value: str) -> None:
        """Stash an arbitrary string on the report metadata."""
        self.report.metadata[key] = value

    # ------------------------------------------------------------------
    # High-level actions — the 80% of what most tests need.
    # ------------------------------------------------------------------

    def complete_onboarding(self, workflow_id: str | None = None) -> None:
        """Walk the new-user flow and land on the graph view.

        Automatically dispatches on ``mode`` (local vs cloud). If
        ``workflow_id`` is omitted, falls back to ``self.workflow`` from
        the ``@scenario`` decorator, then to a mode-appropriate default.
        """
        wf = workflow_id or self.workflow
        if self.mode == "cloud":
            target = wf or "starter-mythical-creature"
            self.metadata("workflow", target)
            flows.complete_onboarding_cloud(self.driver, workflow_id=target)
        else:
            target = wf or "local-passthrough"
            self.metadata("workflow", target)
            flows.complete_onboarding_local(self.driver, workflow_id=target)

    def run_and_wait_first_frame(self, *, timeout_ms: int = 60_000) -> float:
        """Click Run; block until a video frame renders. Returns ms elapsed.

        Records ``first_frame_time_ms`` on the report.
        """
        ms = flows.start_stream_and_wait_first_frame(self.driver, timeout_ms=timeout_ms)
        self._streaming = True
        self.measure("first_frame_time_ms", int(ms))
        return ms

    def stop_stream(self) -> None:
        """Mark an initiated stop, then click Stop. Idempotent."""
        self.failure_watcher.mark_initiated_stop()
        flows.stop_stream(self.driver)
        self._streaming = False

    def toggle_run(self) -> None:
        """Click the Run/Stop button once. Marks the stop side as initiated.

        For chaos-style rapid-toggle loops. If you want to assert a
        first-frame landed after each Run, call
        ``self.driver.wait_first_frame()`` yourself.
        """
        if self._streaming:
            self.failure_watcher.mark_initiated_stop()
        self.driver.click_testid("stream-run-stop")
        self._streaming = not self._streaming

    def set_parameter(self, name: str, value: object) -> int:
        """POST a single parameter update via the HTTP API.

        Returns the HTTP status code so tests can assert on rejection
        behavior without catching exceptions.
        """
        r = requests.post(
            f"{self.base_url}/api/v1/session/parameters",
            json={name: value},
            timeout=5.0,
        )
        return r.status_code

    def get_parameters(self) -> dict:
        """Read the current runtime parameter state."""
        r = requests.get(f"{self.base_url}/api/v1/session/parameters", timeout=5.0)
        r.raise_for_status()
        return r.json().get("parameters", {})

    def metrics(self) -> dict:
        """Fetch ``/api/v1/session/metrics`` (fps, VRAM, frame stats)."""
        r = requests.get(f"{self.base_url}/api/v1/session/metrics", timeout=5.0)
        r.raise_for_status()
        return r.json()

    def click(self, testid: str, *, timeout_ms: int = 15_000) -> None:
        """Shortcut for ``driver.click_testid``."""
        self.driver.click_testid(testid, timeout_ms=timeout_ms)

    def wait(self, testid: str, *, timeout_ms: int = 15_000) -> None:
        """Shortcut for ``driver.wait_testid``."""
        self.driver.wait_testid(testid, timeout_ms=timeout_ms)

    def chaos(self) -> ChaosDriver:
        """A ChaosDriver seeded from the run's ``--chaos-seed``.

        Register actions on the returned driver and call ``.run()`` to
        sample them for a bounded duration.
        """
        return ChaosDriver(seed=self.chaos_seed, report_dir=self.test_report_dir)

    def sleep(self, ms: int) -> None:
        """Deterministic browser-side sleep. Prefer real waits over this."""
        self.driver.page.wait_for_timeout(ms)

    # ------------------------------------------------------------------
    # Recording + frame capture (Slice 5 — media quality).
    #
    # These helpers target the class of bug where the output is
    # statistically broken: wrong framerate, synthesized-looking
    # timestamps, black / frozen / all-one-color frames. The harness
    # saves every captured artifact into ``self.test_report_dir`` so it
    # auto-uploads on CI failure.
    # ------------------------------------------------------------------

    def start_recording(self, node_id: str = "record") -> None:
        """Start the headless recorder for a record node in the graph.

        Headless sessions have session_id ``"headless"``, so we POST to
        ``/api/v1/recordings/headless/start?node_id=<id>``. For graphs
        without a named record node, pass ``node_id=""``.
        """
        url = f"{self.base_url}/api/v1/recordings/headless/start"
        params = {"node_id": node_id} if node_id else {}
        r = requests.post(url, params=params, timeout=10.0)
        r.raise_for_status()

    def stop_and_download_recording(
        self,
        node_id: str = "record",
        *,
        filename: str | None = None,
    ) -> Path:
        """Stop the recorder and download the MP4 into the report dir.

        Returns the local ``Path`` to the downloaded file. Raises if the
        download returns non-2xx or a zero-byte response.
        """
        base = self.base_url
        stop_url = f"{base}/api/v1/recordings/headless/stop"
        dl_url = f"{base}/api/v1/recordings/headless"
        params = {"node_id": node_id} if node_id else {}
        r = requests.post(stop_url, params=params, timeout=15.0)
        r.raise_for_status()
        r = requests.get(dl_url, params=params, timeout=60.0)
        r.raise_for_status()
        if not r.content:
            raise RuntimeError(f"empty recording download from {dl_url}")
        out = self.test_report_dir / (
            filename or f"recording_{node_id or 'default'}.mp4"
        )
        out.write_bytes(r.content)
        return out

    def capture_live_frame(
        self,
        *,
        sink_node_id: str | None = None,
        filename: str | None = None,
    ) -> Path:
        """Snapshot the current frame from the live session as a JPEG.

        Saved into ``self.test_report_dir`` so it's an auto-uploaded
        artifact. ``sink_node_id`` targets a specific sink in a
        multi-sink graph; omit it for single-sink workflows.
        """
        url = f"{self.base_url}/api/v1/session/frame"
        params = {"sink_node_id": sink_node_id} if sink_node_id else {}
        r = requests.get(url, params=params, timeout=10.0)
        r.raise_for_status()
        if not r.content:
            raise RuntimeError(f"empty frame response from {url}")
        ts = int(time.time() * 1000)
        name = filename or f"frame_{sink_node_id or 'default'}_{ts}.jpg"
        out = self.test_report_dir / name
        out.write_bytes(r.content)
        return out

    # ------------------------------------------------------------------
    # Screenshot + multimodal verification (Slice 5 — visual quality).
    #
    # ``screenshot`` and ``screenshot_testid`` capture browser pixels
    # into the report dir so they're available both as failure artifacts
    # and as inputs to multimodal_check(). ``multimodal_check`` routes
    # any mix of sink frames, screenshots, and recorded-MP4 samples
    # through the Anthropic vision API when SCOPE_MULTIMODAL_EVAL=1.
    # ------------------------------------------------------------------

    def screenshot(self, name: str | None = None) -> Path:
        """Full-page browser screenshot into the report dir.

        ``name`` is an optional filename (``.png`` appended if missing).
        Returns the saved path.
        """
        ts = int(time.time() * 1000)
        fname = name or f"screenshot_{ts}.png"
        if not fname.endswith((".png", ".jpg", ".jpeg")):
            fname = f"{fname}.png"
        out = self.test_report_dir / fname
        self.driver.page.screenshot(path=str(out), full_page=True)
        return out

    def screenshot_testid(self, testid: str, name: str | None = None) -> Path:
        """Screenshot scoped to a single testid element.

        Critical for tooltip/modal/button-state checks where a
        full-page shot is too noisy for either a human reviewer or a
        multimodal eval to focus on. The element must already be visible
        — we don't auto-wait on testid here; call ``ctx.wait(testid)``
        first if needed.
        """
        ts = int(time.time() * 1000)
        fname = name or f"{testid}_{ts}.png"
        if not fname.endswith((".png", ".jpg", ".jpeg")):
            fname = f"{fname}.png"
        out = self.test_report_dir / fname
        locator = self.driver.page.locator(f'[data-testid="{testid}"]').first
        locator.screenshot(path=str(out))
        return out

    def multimodal_check(
        self,
        images: Path | list[Path],
        question: str,
        *,
        must_contain: list[str] | None = None,
    ):
        """Route images + question through the multimodal eval.

        Gated by ``SCOPE_MULTIMODAL_EVAL=1``. When disabled, returns a
        ``Verdict`` with ``status="uncertain"`` and a "disabled" reason,
        so tests marked ``@pytest.mark.multimodal`` still collect without
        burning API credit locally.

        The images argument can be any mix of:
          - sink frames from ``ctx.capture_live_frame()``
          - browser screenshots from ``ctx.screenshot()`` / ``screenshot_testid()``
          - recorded-MP4 sample frames from ``harness.media.sample_frames()``

        Returns the ``Verdict`` without auto-asserting. Callers decide
        whether an ``uncertain`` verdict is a skip, a warning, or a fail.
        """
        from . import visual_eval

        paths = [images] if isinstance(images, Path) else list(images)
        return visual_eval.eval_images(paths, question, must_contain=must_contain)

    def capture_sink_video_slice(
        self,
        seconds: float = 3.0,
        *,
        filename: str | None = None,
    ) -> Path:
        """Grab a short MP4 slice from the live MPEG-TS output.

        Uses ``ffmpeg`` to pull from ``/api/v1/session/output.ts`` for
        ``seconds`` seconds, remux to MP4, and save into the report dir.
        Useful when a single-frame capture wouldn't show the bug (e.g.
        stutter, intermittent artifacts).
        """
        import shutil
        import subprocess

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError(
                "ffmpeg not found on PATH. "
                "Install with `brew install ffmpeg` / `apt-get install ffmpeg`."
            )
        ts_url = f"{self.base_url}/api/v1/session/output.ts"
        ts = int(time.time() * 1000)
        out = self.test_report_dir / (filename or f"slice_{ts}.mp4")
        subprocess.check_call(
            [
                ffmpeg,
                "-loglevel",
                "error",
                "-y",
                "-t",
                f"{seconds:.3f}",
                "-i",
                ts_url,
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                str(out),
            ],
            timeout=seconds + 30.0,
        )
        return out

    # ------------------------------------------------------------------
    # Internal — teardown contract.
    # ------------------------------------------------------------------

    def _teardown(self, *, body_raised: bool) -> None:
        """Close the stream cleanly and enforce the gate checklist.

        Called from the decorator's ``finally`` block. If the test body
        already raised, we skip the gate assertion so pytest reports the
        original error instead of a secondary failure.
        """
        if self._torn_down:
            return
        self._torn_down = True

        # Clean stop if we're still streaming. Best-effort; a Scope that's
        # already dead shouldn't prevent gate reporting.
        if self._streaming:
            try:
                self.stop_stream()
            except Exception:
                pass

        # Small settle window so any session_closed fires during the
        # graceful-stop grace period, not after.
        time.sleep(0.5)

        # Populate dimensions even on body failure — the artifacts are
        # still valuable for the report.
        try:
            gates.enforce_all_gates(
                self.report, self.retry_probe, self.failure_watcher, self.driver
            )
        except Exception:
            # Never let a gate-check-crash mask the real test failure.
            pass

        # Opt-in triage pass on failure: point a multimodal reviewer at
        # whatever we captured during the test and write a plain-English
        # summary into the report dir. This turns "here's trace.zip, good
        # luck" into "the workflow picker rendered 2 cards instead of 3".
        failing = body_raised or not self.report.passed
        if failing and os.environ.get("SCOPE_MULTIMODAL_TRIAGE") == "1":
            try:
                self._write_triage_report()
            except Exception:
                # Never let triage mask the real failure.
                pass

        if body_raised:
            return
        assert self.report.passed, f"Hard fails: {self.report.hard_fails}"

    def _write_triage_report(self) -> None:
        """Collect captured images and ask the multimodal eval to describe
        the failure in plain English. Writes ``triage.md`` into the
        report dir. Gated upstream by ``SCOPE_MULTIMODAL_TRIAGE=1``.
        """
        from . import visual_eval

        if not visual_eval.is_enabled():
            # TRIAGE is on but EVAL isn't — skip gracefully.
            return

        exts = {".png", ".jpg", ".jpeg"}
        candidates = sorted(
            p
            for p in self.test_report_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in exts
        )
        if not candidates:
            return
        # Keep the payload bounded: take up to 8 images, evenly sampled.
        if len(candidates) > 8:
            step = max(1, len(candidates) // 8)
            candidates = candidates[::step][:8]

        context = (
            f"Test report dir: {self.test_report_dir.name}. "
            f"Hard fails: {self.report.hard_fails}."
        )
        verdict = visual_eval.triage(candidates, context=context)

        lines = [
            "# Triage — multimodal failure summary",
            "",
            f"- status: `{verdict.status}`",
            f"- hard_fails: {self.report.hard_fails}",
            "- images reviewed:",
            *[f"  - `{p.name}`" for p in candidates],
            "",
            "## Reasoning",
            "",
            verdict.reasoning or "(no reasoning returned)",
            "",
        ]
        if verdict.observations:
            lines += [
                "## Observations",
                "",
                *[f"- {o}" for o in verdict.observations],
                "",
            ]
        if verdict.missing_required:
            lines += [
                "## Missing required",
                "",
                *[f"- {m}" for m in verdict.missing_required],
                "",
            ]
        (self.test_report_dir / "triage.md").write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# @scenario decorator
# ---------------------------------------------------------------------------


# Canonical feature axis. Must match the marker list in pytest.ini so
# `pytest -m <feature>` works without the --strict-markers check firing.
# Tests can pass a single feature or a tuple; unknown features raise early
# rather than silently producing unregistered markers.
_CANONICAL_FEATURES = frozenset(
    {
        "onboarding",
        "recording",
        "params",
        "lifecycle",
        "networking",
        "input",
        "graph",
        "ui",
    }
)


def scenario(
    *,
    mode: str = "local",
    workflow: str | None = None,
    feature: str | tuple[str, ...] | None = None,
    marks: tuple = (),
) -> Callable:
    """Turn a ``def test_foo(ctx)`` function into a full-gated pytest test.

    Args:
        mode: ``"local"`` (default) or ``"cloud"``. Cloud tests auto-skip
            when ``SCOPE_CLOUD_APP_ID`` is unset and receive a test-only
            auth bypass in the browser.
        workflow: default workflow id for ``ctx.complete_onboarding()``.
            Override per-call if a single test switches workflows.
        feature: one or more feature-axis tags (e.g. ``"recording"`` or
            ``("ui", "onboarding")``). Applied as pytest markers so that
            ``pytest -m recording`` selects every recording-related test
            regardless of which folder it lives in. Canonical set:
            ``onboarding, recording, params, lifecycle, networking, input,
            graph, ui``.
        marks: additional pytest marks to apply (e.g. ``(pytest.mark.slow,)``).

    The decorated function MUST be named ``test_*`` per pytest's
    collection rules, take a single ``ctx`` argument, and live under
    ``product-tests/`` so the conftest fixtures are visible.
    """
    if mode not in {"local", "cloud"}:
        raise ValueError(f"mode must be 'local' or 'cloud', got {mode!r}")

    # Normalize feature to a tuple of validated strings.
    if feature is None:
        features: tuple[str, ...] = ()
    elif isinstance(feature, str):
        features = (feature,)
    else:
        features = tuple(feature)
    for f in features:
        if f not in _CANONICAL_FEATURES:
            raise ValueError(
                f"unknown feature {f!r}; canonical set: {sorted(_CANONICAL_FEATURES)}"
            )

    def decorator(user_fn: Callable) -> Callable:
        # The wrapper's parameters MUST match fixture names exactly so
        # pytest's fixture injection works. Do NOT rename these.
        def _impl(
            scope_harness: ScopeHarness,
            driver: PlaywrightDriver,
            retry_probe: RetryProbe,
            failure_watcher: FailureWatcher,
            report: TestReport,
            test_report_dir: Path,
            chaos_seed: str,
        ):
            ctx = ScenarioContext(
                harness=scope_harness,
                driver=driver,
                retry_probe=retry_probe,
                failure_watcher=failure_watcher,
                report=report,
                test_report_dir=test_report_dir,
                chaos_seed=chaos_seed,
                workflow=workflow,
            )
            body_raised = False
            try:
                user_fn(ctx)
            except Exception:
                body_raised = True
                raise
            finally:
                ctx._teardown(body_raised=body_raised)

        # Preserve the user's test name so pytest's node id is stable.
        # Critically, we do NOT use ``functools.wraps`` here: that sets
        # ``__wrapped__`` which makes ``inspect.signature(follow_wrapped=True)``
        # return ``(ctx)``, and pytest would then look for a ``ctx`` fixture
        # that doesn't exist. By manually copying only the identity attrs,
        # pytest sees ``_impl``'s real parameter list and injects fixtures.
        _impl.__name__ = user_fn.__name__
        _impl.__qualname__ = user_fn.__qualname__
        _impl.__doc__ = user_fn.__doc__
        _impl.__module__ = user_fn.__module__

        # Apply marks. The ``cloud`` mark is read by the scope_harness
        # fixture to enable cloud mode. Feature markers let
        # ``pytest -m <feature>`` slice across folders.
        wrapped: Callable = _impl
        if mode == "cloud":
            wrapped = pytest.mark.cloud(wrapped)
        for f in features:
            wrapped = getattr(pytest.mark, f)(wrapped)
        for m in marks:
            wrapped = m(wrapped)
        # Retain a back-reference so introspection tools / error messages
        # can surface the decorator's config.
        wrapped.__scenario_config__ = {  # type: ignore[attr-defined]
            "mode": mode,
            "workflow": workflow,
            "features": features,
        }
        return wrapped

    return decorator


__all__ = ["scenario", "ScenarioContext"]
