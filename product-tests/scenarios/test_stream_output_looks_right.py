"""Stream-output multimodal — captured sink frames don't look broken.

The "green in CI but the user sees garbage" class of bug: the stream
runs, WebRTC carries frames, metrics look healthy, but the actual
pixels are all black, all one color, frozen, or obviously artifacted.
No testid or numeric threshold catches this. A human glance does.

This test runs a passthrough session, samples five live frames from
the sink over a few seconds, and asks Claude with vision whether they
look like normal video output (varying content, reasonable contrast,
not-pathological).

Uses ``passthrough`` pipeline + a synthesized moving-gradient source
so the test is CPU-only but produces deterministic non-pathological
output: any all-black / all-one-color / frozen-frame verdict proves
the pipeline dropped the source, not that the source was empty to
begin with.

Requires ``SCOPE_MULTIMODAL_EVAL=1`` + ``ANTHROPIC_API_KEY``.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
import requests
from harness import media
from harness.scenario import scenario


def _make_gradient_video(path: Path, seconds: int = 20, fps: int = 30) -> None:
    """Write a small MP4 whose content varies per frame — so a
    passthrough pipeline's output is expected to differ frame-to-frame.
    All-one-color or frozen output is therefore diagnostic of a
    pipeline/sink bug, not a source bug.
    """
    import cv2

    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (320, 240))
    try:
        total = seconds * fps
        for i in range(total):
            frame = np.zeros((240, 320, 3), dtype=np.uint8)
            # Animated diagonal gradient so consecutive frames differ.
            shift = (i * 4) % 255
            for y in range(240):
                for x_block in range(0, 320, 32):
                    frame[y, x_block : x_block + 32] = (
                        (x_block + shift) % 255,
                        (y + shift) % 255,
                        (i * 3) % 255,
                    )
            # Also draw a frame counter so "frozen" is unambiguous.
            cv2.putText(
                frame,
                f"f{i:04d}",
                (8, 232),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            w.write(frame)
    finally:
        w.release()


@scenario(
    mode="local",
    workflow="local-passthrough",
    feature=("ui", "lifecycle"),
    marks=(pytest.mark.multimodal,),
)
def test_passthrough_sink_frames_look_right(ctx):
    """Run passthrough; capture 5 sink frames; verify they look like video."""
    # We drive this via HTTP, not the UI, so we get deterministic control
    # over the source. ``complete_onboarding_local`` would run too, but
    # we'd still want to override the source, so we skip the picker.
    src = ctx.test_report_dir / "gradient_source.mp4"
    _make_gradient_video(src, seconds=20, fps=30)

    start_body = {
        "pipeline_id": "passthrough",
        "input_mode": "video",
        "input_source": {
            "enabled": True,
            "source_type": "video_file",
            "source_name": str(src),
        },
    }
    r = requests.post(
        f"{ctx.base_url}/api/v1/session/start",
        json=start_body,
        timeout=30.0,
    )
    r.raise_for_status()
    ctx._streaming = True  # so teardown stops cleanly

    # Let frames flow for ~3s before sampling.
    time.sleep(3.0)

    captures: list[Path] = []
    for i in range(5):
        captures.append(ctx.capture_live_frame(filename=f"sink_{i}.jpg"))
        time.sleep(0.6)

    # Cheap pre-check: if every frame is black or monochrome, we know
    # the answer without calling the API. This also lets the test RED
    # on its own in environments where multimodal is disabled.
    all_bad = all(media.looks_black(p) or media.looks_monochrome(p) for p in captures)
    ctx.measure(
        "degenerate_live_frames",
        sum(1 for p in captures if media.looks_black(p) or media.looks_monochrome(p)),
    )
    if all_bad:
        ctx.report.fail(
            f"all {len(captures)} captured sink frames are black or "
            "single-color — pipeline is emitting garbage"
        )

    # Multimodal: is the output actually sensible?
    verdict = ctx.multimodal_check(
        captures,
        question=(
            "These are five frames captured from a live video stream "
            "driven by an animated gradient source. Do they look like "
            "frames of a normal video — varying content, reasonable "
            "contrast, a visible frame counter in the bottom-left — as "
            "opposed to all-black frames, all-one-color frames, or five "
            "identical (frozen) frames?"
        ),
        must_contain=[
            "varied pixel content across the frames (not all identical)",
            "not all black",
            "not a single solid color",
        ],
    )

    ctx.metadata("multimodal_status", verdict.status)
    ctx.metadata("multimodal_reasoning", verdict.reasoning)

    if verdict.status == "fail":
        ctx.report.fail(
            f"sink output multimodal check failed: {verdict.reasoning} "
            f"(missing: {verdict.missing_required or 'n/a'})"
        )
