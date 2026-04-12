from __future__ import annotations

from dataclasses import dataclass

# Defaults tuned for realtime media pacing.
DEFAULT_DRIFT_TOLERANCE_S = 0.010
DEFAULT_SOFT_REANCHOR_DRIFT_S = 0.50
DEFAULT_DISCONTINUITY_MIN_S = 0.25
DEFAULT_DISCONTINUITY_MULTIPLIER = 8.0
DEFAULT_EXPECTED_DELTA_ALPHA = 0.20


@dataclass(slots=True)
class MediaPacingConfig:
    drift_tolerance_s: float = DEFAULT_DRIFT_TOLERANCE_S
    soft_reanchor_drift_s: float = DEFAULT_SOFT_REANCHOR_DRIFT_S
    discontinuity_min_s: float = DEFAULT_DISCONTINUITY_MIN_S
    discontinuity_multiplier: float = DEFAULT_DISCONTINUITY_MULTIPLIER
    expected_delta_alpha: float = DEFAULT_EXPECTED_DELTA_ALPHA


@dataclass(slots=True)
class MediaPacingState:
    start_media_ts: float | None = None
    start_wall_monotonic: float | None = None
    prev_media_ts: float | None = None
    prev_wall_monotonic: float | None = None
    expected_media_delta: float | None = None


@dataclass(slots=True)
class MediaPacingDecision:
    sleep_s: float
    hard_reset: bool = False
    soft_reanchor: bool = False
    has_valid_ts: bool = False
    drift_s: float | None = None
    media_delta_s: float | None = None
    wall_delta_s: float | None = None
    stall_delta_s: float | None = None


def reset_pacing_state(state: MediaPacingState) -> None:
    state.start_media_ts = None
    state.start_wall_monotonic = None
    state.prev_media_ts = None
    state.prev_wall_monotonic = None
    state.expected_media_delta = None


def anchor_pacing_state(
    state: MediaPacingState,
    *,
    media_ts: float,
    now_monotonic: float,
    reset_expected_delta: bool = False,
) -> None:
    state.start_media_ts = media_ts
    state.start_wall_monotonic = now_monotonic
    state.prev_media_ts = media_ts
    state.prev_wall_monotonic = now_monotonic
    if reset_expected_delta:
        state.expected_media_delta = None


def _update_expected_media_delta(
    expected_media_delta: float | None,
    media_delta: float,
    *,
    alpha: float,
) -> float:
    if expected_media_delta is None:
        return media_delta
    return (1.0 - alpha) * expected_media_delta + alpha * media_delta


def compute_pacing_decision(
    state: MediaPacingState,
    *,
    media_ts: float | None,
    now_monotonic: float,
    config: MediaPacingConfig | None = None,
) -> MediaPacingDecision:
    cfg = config or MediaPacingConfig()
    # Missing timestamps mean we cannot align media time to wall time; reset
    # pacing state and hand off immediately.
    if media_ts is None:
        reset_pacing_state(state)
        return MediaPacingDecision(sleep_s=0.0, hard_reset=True, has_valid_ts=False)

    # First valid timestamp establishes the stream anchor for cumulative drift.
    if state.start_media_ts is None or state.start_wall_monotonic is None:
        anchor_pacing_state(state, media_ts=media_ts, now_monotonic=now_monotonic)
        return MediaPacingDecision(sleep_s=0.0, has_valid_ts=True, drift_s=0.0)

    media_delta = (
        media_ts - state.prev_media_ts if state.prev_media_ts is not None else None
    )
    wall_delta = (
        now_monotonic - state.prev_wall_monotonic
        if state.prev_wall_monotonic is not None
        else None
    )
    stall_delta = (
        wall_delta - media_delta
        if wall_delta is not None and media_delta is not None
        else None
    )

    # Non-monotonic media timestamps indicate a timeline break; hard-reset.
    if media_delta is None or media_delta <= 0:
        anchor_pacing_state(
            state,
            media_ts=media_ts,
            now_monotonic=now_monotonic,
            reset_expected_delta=True,
        )
        return MediaPacingDecision(
            sleep_s=0.0,
            hard_reset=True,
            has_valid_ts=True,
            media_delta_s=media_delta,
            wall_delta_s=wall_delta,
            stall_delta_s=stall_delta,
        )

    # Detect large media-time jumps relative to recent cadence and treat them as
    # discontinuities rather than attempting to "catch up" by pacing math.
    if state.expected_media_delta is not None:
        discontinuity_s = max(
            cfg.discontinuity_min_s,
            state.expected_media_delta * cfg.discontinuity_multiplier,
        )
        if media_delta >= discontinuity_s:
            anchor_pacing_state(
                state,
                media_ts=media_ts,
                now_monotonic=now_monotonic,
                reset_expected_delta=True,
            )
            return MediaPacingDecision(
                sleep_s=0.0,
                hard_reset=True,
                has_valid_ts=True,
                media_delta_s=media_delta,
                wall_delta_s=wall_delta,
                stall_delta_s=stall_delta,
            )

    media_elapsed = media_ts - state.start_media_ts
    wall_elapsed = now_monotonic - state.start_wall_monotonic
    drift = wall_elapsed - media_elapsed

    # For extreme positive drift (wall-clock far behind media timeline), soften
    # the debt instead of fast-forward draining bursty payloads.
    if drift >= cfg.soft_reanchor_drift_s:
        anchor_pacing_state(
            state,
            media_ts=media_ts,
            now_monotonic=now_monotonic,
            reset_expected_delta=False,
        )
        state.expected_media_delta = _update_expected_media_delta(
            state.expected_media_delta,
            media_delta,
            alpha=cfg.expected_delta_alpha,
        )
        return MediaPacingDecision(
            sleep_s=0.0,
            soft_reanchor=True,
            has_valid_ts=True,
            drift_s=drift,
            media_delta_s=media_delta,
            wall_delta_s=wall_delta,
            stall_delta_s=stall_delta,
        )

    # Core pacing: negative drift means we're ahead of media time, so sleep;
    # positive drift means we're behind, so send immediately.
    sleep_s = max(0.0, -drift)
    # Deadband: ignore tiny (<=10ms) drift to avoid 1-2ms micro-sleeps and
    # scheduler-noise jitter on realtime streams.
    if abs(drift) <= cfg.drift_tolerance_s:
        sleep_s = 0.0

    state.prev_media_ts = media_ts
    state.expected_media_delta = _update_expected_media_delta(
        state.expected_media_delta,
        media_delta,
        alpha=cfg.expected_delta_alpha,
    )
    return MediaPacingDecision(
        sleep_s=sleep_s,
        has_valid_ts=True,
        drift_s=drift,
        media_delta_s=media_delta,
        wall_delta_s=wall_delta,
        stall_delta_s=stall_delta,
    )
