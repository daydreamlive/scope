from __future__ import annotations

from scope.server.pacing import MediaPacingState, compute_pacing_decision


def _dispatch(state: MediaPacingState, dispatch_monotonic: float) -> None:
    state.prev_wall_monotonic = dispatch_monotonic


def test_pacing_absorbs_small_oversleep_without_growing_delay():
    state = MediaPacingState()
    frame_interval = 1.0 / 30.0
    scheduler_oversleep = 0.001

    # First frame anchors media and wall clocks.
    first = compute_pacing_decision(state, media_ts=0.0, now_monotonic=0.0)
    assert first.sleep_s == 0.0
    _dispatch(state, dispatch_monotonic=0.0)

    # Subsequent frames are available immediately from a backlog.
    now = 0.0
    sleeps: list[float] = []
    for i in range(1, 10):
        decision = compute_pacing_decision(
            state,
            media_ts=i * frame_interval,
            now_monotonic=now,
        )
        sleeps.append(decision.sleep_s)
        now = now + decision.sleep_s + scheduler_oversleep
        _dispatch(state, dispatch_monotonic=now)

    assert max(sleeps) - min(sleeps) < 0.003
    assert abs(sleeps[-1] - (frame_interval - scheduler_oversleep)) < 0.003


def test_wall_clock_stall_with_continuous_pts_catches_up_without_hard_reset():
    state = MediaPacingState()
    frame_interval = 1.0 / 30.0

    first = compute_pacing_decision(state, media_ts=0.0, now_monotonic=0.0)
    assert first.sleep_s == 0.0
    _dispatch(state, dispatch_monotonic=0.0)

    second = compute_pacing_decision(
        state,
        media_ts=frame_interval,
        now_monotonic=0.0,
    )
    _dispatch(state, dispatch_monotonic=second.sleep_s)

    stalled_now = second.sleep_s + 0.25
    third = compute_pacing_decision(
        state,
        media_ts=2.0 * frame_interval,
        now_monotonic=stalled_now,
    )

    assert third.hard_reset is False
    assert third.soft_reanchor is False
    assert third.sleep_s == 0.0
    assert third.drift_s is not None and third.drift_s > 0
    assert third.stall_delta_s is not None and third.stall_delta_s > 0


def test_extreme_wall_clock_debt_triggers_soft_reanchor():
    state = MediaPacingState()
    frame_interval = 1.0 / 30.0

    compute_pacing_decision(state, media_ts=0.0, now_monotonic=0.0)
    _dispatch(state, dispatch_monotonic=0.0)

    second = compute_pacing_decision(
        state,
        media_ts=frame_interval,
        now_monotonic=0.0,
    )
    _dispatch(state, dispatch_monotonic=second.sleep_s)

    delayed_now = second.sleep_s + 0.9
    third = compute_pacing_decision(
        state,
        media_ts=2.0 * frame_interval,
        now_monotonic=delayed_now,
    )

    assert third.soft_reanchor is True
    assert third.hard_reset is False
    assert third.sleep_s == 0.0


def test_large_media_discontinuity_triggers_hard_reset():
    state = MediaPacingState()
    frame_interval = 1.0 / 30.0

    compute_pacing_decision(state, media_ts=0.0, now_monotonic=0.0)
    _dispatch(state, dispatch_monotonic=0.0)

    second = compute_pacing_decision(
        state,
        media_ts=frame_interval,
        now_monotonic=0.0,
    )
    _dispatch(state, dispatch_monotonic=second.sleep_s)

    # Build expected delta history.
    third = compute_pacing_decision(
        state,
        media_ts=2.0 * frame_interval,
        now_monotonic=second.sleep_s,
    )
    _dispatch(state, dispatch_monotonic=second.sleep_s + third.sleep_s)

    jump = 2.0 * frame_interval + 1.0
    reset = compute_pacing_decision(
        state,
        media_ts=jump,
        now_monotonic=second.sleep_s + third.sleep_s + 0.01,
    )

    assert reset.hard_reset is True
    assert reset.soft_reanchor is False
    assert reset.sleep_s == 0.0


def test_missing_timestamp_is_hard_reset():
    state = MediaPacingState()
    compute_pacing_decision(state, media_ts=0.0, now_monotonic=0.0)
    _dispatch(state, dispatch_monotonic=0.0)

    decision = compute_pacing_decision(
        state,
        media_ts=None,
        now_monotonic=0.2,
    )
    assert decision.hard_reset is True
    assert decision.has_valid_ts is False
    assert decision.sleep_s == 0.0
