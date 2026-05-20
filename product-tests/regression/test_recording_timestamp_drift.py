"""Regression — recorded output must carry real PTS, not synthesized ones.

Reported by a user in Discord: recordings taken from a cloud session show
a lower framerate and visible artifacts (stuttering, pixelation) vs. the
live WebRTC output. Root cause (per Rami's write-up in that thread):

  Most pipelines synthesize frame timestamps best-effort from a nominal
  FPS heuristic. Only ``passthrough`` and ``ltx`` forward real PTS from
  the source. The recording pipeline then **rewrites** incoming
  timestamps instead of passing through whatever the runner emitted —
  so any mismatch between the runner's actual cadence and its declared
  FPS produces a recording with the wrong clock. On cloud sessions where
  the runner's effective FPS oscillates, the recorded clock drifts. The
  recorded MP4 is a faithful record of the synthesized clock, not the
  real one, so playback stutters.

What this test asserts (all must pass):

  1. Recording decodes and reports a non-absurd frame count.
  2. Recorded mean FPS is within ±10% of the live ``session/metrics`` FPS.
  3. ``analyze_timing`` does NOT classify the PTS as ``looks_synthesized``
     (real PTS have small but nonzero jitter; synthesized ones don't).
  4. A sample of 5 evenly-spaced frames is captured into the report dir
     so a failed run has visual evidence for triage.

Nightly-ring by default because the happy-path workflow
``starter-mythical-creature`` is GPU-bound.

How to verify this test actually catches the bug: temporarily force the
recording pipeline to synthesize timestamps (e.g. rewrite PTS as
``frame_idx / nominal_fps`` in the recorder path), run this test, and
observe it red. Revert and it should green. That round-trip is the
demonstration the bug→test loop works on real user-reported issues.
"""

from __future__ import annotations

import time

import pytest
from harness import media
from harness.scenario import scenario


@scenario(
    mode="cloud",
    workflow="starter-mythical-creature",
    feature=("recording", "lifecycle"),
    marks=(pytest.mark.regression, pytest.mark.slow),
)
def test_recording_timestamp_passthrough(ctx):
    """Record 5s; assert recorded FPS matches live FPS and PTS look real."""
    ctx.complete_onboarding()
    ctx.run_and_wait_first_frame(timeout_ms=120_000)

    # Let the pipeline settle so the reported FPS is steady before we sample.
    ctx.sleep(2_000)

    pre_metrics = ctx.metrics()
    live_fps = float(pre_metrics.get("fps") or pre_metrics.get("frame_rate") or 0)
    ctx.measure("live_fps_pre_record", round(live_fps, 2))
    if live_fps <= 0:
        ctx.report.fail(
            "could not read live FPS from /api/v1/session/metrics — test cannot "
            "compare recorded vs. live framerate without a reference"
        )

    # Record ~5 seconds of the running session.
    ctx.start_recording(node_id="record")
    t0 = time.perf_counter()
    ctx.sleep(5_000)
    record_wall_sec = time.perf_counter() - t0
    mp4 = ctx.stop_and_download_recording(node_id="record")
    ctx.measure("recording_size_bytes", mp4.stat().st_size)

    # ffprobe the recorded PTS and compute a timing report.
    pts = media.ffprobe_pts(mp4)
    timing = media.analyze_timing(pts)
    ctx.measure("recorded_frame_count", timing.frame_count)
    ctx.measure("recorded_duration_sec", round(timing.duration_sec, 3))
    ctx.measure("recorded_mean_fps", round(timing.mean_fps, 2))
    ctx.measure("recorded_jitter_stddev_sec", round(timing.jitter_stddev_sec, 6))
    ctx.measure("recorded_jitter_p95_sec", round(timing.jitter_p95_sec, 6))
    ctx.metadata("looks_synthesized", "yes" if timing.looks_synthesized else "no")

    # -- Gate 1: frame count sanity.
    if timing.frame_count < 10:
        ctx.report.fail(
            f"recorded frame count suspiciously low: {timing.frame_count} "
            f"(recorded ~{record_wall_sec:.1f}s at live fps ~{live_fps:.1f} — "
            "expected tens to hundreds of frames)"
        )

    # -- Gate 2: recorded FPS matches live FPS within 10% (symmetric ratio).
    if timing.mean_fps > 0 and live_fps > 0:
        ratio = timing.mean_fps / live_fps
        ctx.measure("recorded_live_fps_ratio", round(ratio, 3))
        if not (0.9 <= ratio <= 1.1):
            ctx.report.fail(
                f"recorded FPS {timing.mean_fps:.2f} vs live FPS {live_fps:.2f} "
                f"(ratio {ratio:.2f}) — outside the ±10% acceptance window. "
                "This is the Discord bug: the recording clock doesn't match "
                "the live runner's actual frame cadence."
            )

    # -- Gate 3: PTS do not look synthesized.
    if timing.looks_synthesized:
        ctx.report.fail(
            "recorded PTS look synthesized (stddev/mean_delta < 1%). Real "
            "pipeline output has small-but-nonzero jitter; regenerated "
            "best-effort timestamps from an FPS heuristic do not. "
            "Expected the recorder to pass through the runner's real PTS."
        )

    # -- Gate 4: capture 5 evenly-spaced frames for visual evidence (artifacted
    # pixelation is hard to see in stats; a human or multimodal check can
    # look at these after a fail). Saved into the report dir as artifacts.
    samples = media.sample_frames(
        mp4, n=5, out_dir=ctx.test_report_dir / "recording_samples"
    )
    ctx.measure("frame_samples_extracted", len(samples))
    ctx.metadata("frame_sample_paths", "; ".join(str(p.name) for p in samples))

    # -- Gate 5: none of the samples should be black / monochrome (catches the
    # "recording container has frames but they're all green" class of bug).
    bad_samples = [
        p.name for p in samples if media.looks_black(p) or media.looks_monochrome(p)
    ]
    ctx.measure("degenerate_frame_samples", len(bad_samples))
    if bad_samples:
        ctx.report.fail(
            f"{len(bad_samples)}/{len(samples)} recorded frame samples are "
            f"black or single-color: {bad_samples}. This is the visible half "
            "of the artifact class users report (pixelation / stutter often "
            "surfaces as dropped-to-flat frames in the recorded output)."
        )
