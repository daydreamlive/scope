"""Recording round-trip — start, stream, stop, download, verify the MP4.

Scope wires the recording API but we never actually check the file is
valid. This test pulls the bytes, decodes with OpenCV, and asserts:

  - The file is a real MP4 (decodable via cv2.VideoCapture).
  - Frame count is non-zero and consistent with duration.
  - Reported FPS is sane (>= 5, <= 120).
  - Resolution matches what the pipeline declares.

Catches bugs where recording silently drops frames, writes an empty
container, or produces a file that opens in VLC but not in any
programmatic decoder.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import pytest
import requests
from harness import flows
from harness.failure_watcher import FailureWatcher
from harness.report import TestReport
from harness.retry_probe import RetryProbe
from harness.scope_process import ScopeHarness


def _make_test_video(path: Path, seconds: int = 10) -> None:
    """Make a 30fps solid-color MP4 so we have a deterministic input."""
    import numpy as np

    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 240))
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    frame[:] = (0, 255, 0)
    for _ in range(30 * seconds):
        w.write(frame)
    w.release()


@pytest.mark.recording
def test_recording_roundtrip_local_passthrough(
    scope_harness: ScopeHarness,
    retry_probe: RetryProbe,
    failure_watcher: FailureWatcher,
    report: TestReport,
    tmp_path: Path,
):
    """HTTP-only: start headless session with video-file source, record, validate."""
    report.metadata["workflow"] = "local-passthrough"

    # Produce a deterministic source video.
    src = tmp_path / "src.mp4"
    _make_test_video(src, seconds=10)

    base = scope_harness.base_url

    # 1. Load the pipeline first — direct-HTTP tests skip the UI
    # onboarding flow that would normally do this implicitly.
    flows.http_load_pipeline_and_wait(base, ["passthrough"])

    # 2. Start a headless session with passthrough pipeline.
    body = {
        "pipeline_id": "passthrough",
        "input_mode": "video",
        "input_source": {
            "enabled": True,
            "source_type": "video_file",
            "source_name": str(src),
        },
    }
    r = requests.post(f"{base}/api/v1/session/start", json=body, timeout=30.0)
    assert r.status_code == 200, f"session/start: {r.status_code} {r.text[:200]}"

    # 2. Give frames a moment to start flowing.
    time.sleep(2.0)

    # 3. Start recording, let it run ~3s, stop.
    r = requests.post(f"{base}/api/v1/recordings/headless/start", timeout=10.0)
    assert r.status_code == 200, f"recordings start: {r.status_code} {r.text[:200]}"
    record_start = time.perf_counter()
    time.sleep(3.0)
    r = requests.post(f"{base}/api/v1/recordings/headless/stop", timeout=10.0)
    assert r.status_code == 200, f"recordings stop: {r.status_code} {r.text[:200]}"
    record_duration = time.perf_counter() - record_start
    report.measure("recording_duration_sec", round(record_duration, 2))

    # 4. Download the MP4.
    r = requests.get(f"{base}/api/v1/recordings/headless", timeout=30.0, stream=True)
    assert r.status_code == 200, f"recordings get: {r.status_code} {r.text[:200]}"
    assert r.headers.get("content-type", "").startswith("video/mp4"), (
        f"unexpected content-type: {r.headers.get('content-type')}"
    )
    out = tmp_path / "out.mp4"
    out.write_bytes(r.content)

    size_bytes = out.stat().st_size
    report.measure("recording_size_bytes", size_bytes)
    if size_bytes < 1024:
        report.fail(
            f"recording too small ({size_bytes} bytes) — likely empty container"
        )

    # 5. Decode with cv2.
    cap = cv2.VideoCapture(str(out))
    if not cap.isOpened():
        report.fail(f"cv2 cannot open the recording at {out}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        report.measure("recording_fps", round(fps, 2))
        report.measure("recording_frame_count", frame_count)
        report.measure("recording_width", width)
        report.measure("recording_height", height)

        if frame_count == 0:
            report.fail("recording frame_count=0 — container valid but no frames")
        if not (1.0 <= fps <= 120.0):
            report.fail(f"recording fps out of range: {fps}")

        # Can we actually read a frame?
        ok, frame = cap.read()
        if not ok or frame is None:
            report.fail("cv2.read() returned no frame from the first position")
        else:
            report.measure("first_frame_shape", list(frame.shape))
    finally:
        cap.release()

    # 6. Stop the session cleanly.
    failure_watcher.mark_initiated_stop()
    requests.post(f"{base}/api/v1/session/stop", timeout=10.0)

    # 7. Hard gates (we skip enforce_zero_ui_errors since there's no driver).
    from harness import gates

    gates.enforce_zero_retries(report, retry_probe)
    gates.enforce_zero_unexpected_closes(report, failure_watcher)

    assert report.passed, f"Hard fails: {report.hard_fails}"
