"""Tests for ParameterScheduler.

These tests are intentionally adversarial: they probe boundary math edge cases,
race conditions, and scenarios where the scheduler might silently do the wrong
thing rather than crash.
"""

import threading
import time
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from scope.server.parameter_scheduler import ParameterScheduler

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FakeBeatState:
    bpm: float
    beat_phase: float
    bar_position: float
    beat_count: int
    is_playing: bool = True
    timestamp: float = 0.0
    source: str = "test"


class FakeTempoSync:
    def __init__(self, beat_state=None, beats_per_bar=4):
        self._beat_state = beat_state
        self.beats_per_bar = beats_per_bar

    def get_beat_state(self):
        return self._beat_state


def make_scheduler(
    beat_state=None,
    beats_per_bar=4,
    quantize_mode="beat",
    lookahead_ms=0.0,
):
    applied = []
    notifications = []
    tempo = FakeTempoSync(beat_state, beats_per_bar)
    scheduler = ParameterScheduler(
        tempo_sync=tempo,
        apply_callback=lambda p: applied.append(p),
        notification_callback=lambda m: notifications.append(m),
    )
    scheduler.quantize_mode = quantize_mode
    scheduler.lookahead_ms = lookahead_ms
    return scheduler, applied, notifications, tempo


# ---------------------------------------------------------------------------
# Boundary math: _compute_apply_delay
# ---------------------------------------------------------------------------


class TestComputeApplyDelay:
    """Direct tests of the delay calculation. No timers involved."""

    def test_beat_mode_halfway(self):
        """Halfway through a beat at 120 BPM: 0.25s until boundary."""
        s, _, _, _ = make_scheduler(
            beat_state=FakeBeatState(
                bpm=120, beat_phase=0.5, bar_position=0.5, beat_count=0
            ),
        )
        delay = s._compute_apply_delay(s._tempo_sync.get_beat_state())
        assert delay == pytest.approx(0.25, abs=1e-6)

    def test_beat_mode_start_of_beat(self):
        """Phase = 0.0 means we're right on a boundary, should target the NEXT beat."""
        s, _, _, _ = make_scheduler(
            beat_state=FakeBeatState(
                bpm=120, beat_phase=0.0, bar_position=0.0, beat_count=0
            ),
        )
        delay = s._compute_apply_delay(s._tempo_sync.get_beat_state())
        # beats_until = 1.0, but < 0.01 triggers bump? No: 1.0 - 0.0 = 1.0, not < 0.01
        # Actually 1.0 is not < 0.01, so it stays at 1.0 beat = 0.5s
        assert delay == pytest.approx(0.5, abs=1e-6)

    def test_beat_mode_nearly_on_boundary(self):
        """Phase = 0.995 means beats_until = 0.005 < 0.01, should wrap to next beat."""
        s, _, _, _ = make_scheduler(
            beat_state=FakeBeatState(
                bpm=120, beat_phase=0.995, bar_position=0.995, beat_count=0
            ),
        )
        delay = s._compute_apply_delay(s._tempo_sync.get_beat_state())
        # beats_until = 0.005 < 0.01 => wraps to 1.005 beats
        expected = 1.005 * (60.0 / 120.0)
        assert delay == pytest.approx(expected, abs=1e-6)

    def test_bar_mode_third_beat(self):
        """In a 4/4 bar, at the start of beat 3: 1 beat until next bar."""
        s, _, _, _ = make_scheduler(
            beat_state=FakeBeatState(
                bpm=120, beat_phase=0.0, bar_position=3.0, beat_count=3
            ),
            quantize_mode="bar",
        )
        delay = s._compute_apply_delay(s._tempo_sync.get_beat_state())
        # position_in_bar = 3 % 4 + 0.0 = 3.0, beats_until = 4 - 3 = 1.0
        assert delay == pytest.approx(0.5, abs=1e-6)

    def test_bar_mode_start_of_bar(self):
        """Beat 0, phase 0: we're exactly on a bar boundary, target the next bar."""
        s, _, _, _ = make_scheduler(
            beat_state=FakeBeatState(
                bpm=120, beat_phase=0.0, bar_position=0.0, beat_count=0
            ),
            quantize_mode="bar",
        )
        delay = s._compute_apply_delay(s._tempo_sync.get_beat_state())
        # position_in_bar = 0, beats_until = 4.0 (not < 0.01, no wrap)
        assert delay == pytest.approx(2.0, abs=1e-6)

    def test_2bar_mode_position_calculation(self):
        """At beat 5 (phase 0) in a 2-bar cycle (8 beats): 3 beats remain."""
        s, _, _, _ = make_scheduler(
            beat_state=FakeBeatState(
                bpm=120, beat_phase=0.0, bar_position=1.0, beat_count=5
            ),
            quantize_mode="2_bar",
        )
        delay = s._compute_apply_delay(s._tempo_sync.get_beat_state())
        # cycle = 8, position = 5 % 8 + 0 = 5, beats_until = 3
        assert delay == pytest.approx(1.5, abs=1e-6)

    def test_4bar_mode(self):
        """At beat 10 (phase 0.5) in a 4-bar cycle (16 beats): 5.5 beats remain."""
        s, _, _, _ = make_scheduler(
            beat_state=FakeBeatState(
                bpm=120, beat_phase=0.5, bar_position=2.5, beat_count=10
            ),
            quantize_mode="4_bar",
        )
        delay = s._compute_apply_delay(s._tempo_sync.get_beat_state())
        # cycle = 16, position = 10 % 16 + 0.5 = 10.5, beats_until = 5.5
        assert delay == pytest.approx(5.5 * 0.5, abs=1e-6)

    def test_3_4_time_signature(self):
        """3/4 time: bar mode with 3 beats per bar."""
        s, _, _, _ = make_scheduler(
            beat_state=FakeBeatState(
                bpm=120, beat_phase=0.0, bar_position=1.0, beat_count=1
            ),
            beats_per_bar=3,
            quantize_mode="bar",
        )
        delay = s._compute_apply_delay(s._tempo_sync.get_beat_state())
        # position_in_bar = 1 % 3 + 0 = 1, beats_until = 3 - 1 = 2
        assert delay == pytest.approx(1.0, abs=1e-6)

    def test_7_8_time_signature(self):
        """Odd meter: 7 beats per bar."""
        s, _, _, _ = make_scheduler(
            beat_state=FakeBeatState(
                bpm=120, beat_phase=0.5, bar_position=5.5, beat_count=5
            ),
            beats_per_bar=7,
            quantize_mode="bar",
        )
        delay = s._compute_apply_delay(s._tempo_sync.get_beat_state())
        # position_in_bar = 5 % 7 + 0.5 = 5.5, beats_until = 1.5
        assert delay == pytest.approx(0.75, abs=1e-6)

    def test_bpm_zero_returns_zero(self):
        """BPM of 0 should return 0 (apply immediately)."""
        s, _, _, _ = make_scheduler(
            beat_state=FakeBeatState(
                bpm=0, beat_phase=0.5, bar_position=0.5, beat_count=0
            ),
        )
        assert s._compute_apply_delay(s._tempo_sync.get_beat_state()) == 0.0

    def test_bpm_negative_returns_zero(self):
        s, _, _, _ = make_scheduler(
            beat_state=FakeBeatState(
                bpm=-10, beat_phase=0.5, bar_position=0.5, beat_count=0
            ),
        )
        assert s._compute_apply_delay(s._tempo_sync.get_beat_state()) == 0.0

    def test_lookahead_subtracts_from_delay(self):
        """Lookahead of 200ms should reduce delay by 0.2s."""
        s, _, _, _ = make_scheduler(
            beat_state=FakeBeatState(
                bpm=120, beat_phase=0.0, bar_position=0.0, beat_count=0
            ),
            lookahead_ms=200,
        )
        delay = s._compute_apply_delay(s._tempo_sync.get_beat_state())
        # 1.0 beat * 0.5s/beat - 0.2s = 0.3s
        assert delay == pytest.approx(0.3, abs=1e-6)

    def test_lookahead_exceeds_boundary_distance(self):
        """When lookahead > time_to_boundary, delay wraps to the next cycle.

        At 120 BPM with beat_phase=0.5 (0.25s to boundary) and 500ms lookahead,
        naive delay = 0.25 - 0.5 = -0.25. Wrapping by one cycle (0.5s) gives
        delay = 0.25s, targeting the beat after next.
        """
        s, _, _, _ = make_scheduler(
            beat_state=FakeBeatState(
                bpm=120, beat_phase=0.5, bar_position=0.5, beat_count=0
            ),
            lookahead_ms=500,
        )
        delay = s._compute_apply_delay(s._tempo_sync.get_beat_state())
        # -0.25 + 0.5 (one beat cycle at 120 BPM) = 0.25s
        assert delay == pytest.approx(0.25, abs=1e-6)

    def test_very_fast_tempo_with_lookahead(self):
        """At 300 BPM (beat_duration=0.2s) with 200ms lookahead, every beat
        mode schedule results in delay <= 0. Beat quantize is effectively useless.
        """
        s, _, _, _ = make_scheduler(
            beat_state=FakeBeatState(
                bpm=300, beat_phase=0.0, bar_position=0.0, beat_count=0
            ),
            lookahead_ms=200,
        )
        delay = s._compute_apply_delay(s._tempo_sync.get_beat_state())
        # 1.0 beat * 0.2s - 0.2s = 0.0
        assert delay == pytest.approx(0.0, abs=1e-6)

    def test_bar_mode_large_beat_count(self):
        """Beat count 1_000_003 should still compute correct position in bar."""
        s, _, _, _ = make_scheduler(
            beat_state=FakeBeatState(
                bpm=120, beat_phase=0.25, bar_position=3.25, beat_count=1_000_003
            ),
            quantize_mode="bar",
        )
        delay = s._compute_apply_delay(s._tempo_sync.get_beat_state())
        # 1_000_003 % 4 = 3, position_in_bar = 3.25, beats_until = 0.75
        assert delay == pytest.approx(0.75 * 0.5, abs=1e-6)


class TestComputeApplyDelayEdgeCases:
    """Edge cases that might reveal bugs or undefined behavior."""

    def test_beat_phase_exactly_one(self):
        """beat_phase = 1.0 is technically invalid (should be 0.0 of next beat).
        beats_until = 0.0 < 0.01 => wraps to next beat.
        """
        s, _, _, _ = make_scheduler(
            beat_state=FakeBeatState(
                bpm=120, beat_phase=1.0, bar_position=1.0, beat_count=0
            ),
        )
        delay = s._compute_apply_delay(s._tempo_sync.get_beat_state())
        # beats_until = 1.0 - 1.0 = 0.0 < 0.01 => wraps to 1.0
        assert delay == pytest.approx(0.5, abs=1e-6)

    def test_beat_phase_slightly_over_one(self):
        """beat_phase > 1.0 is invalid and gets clamped to 1.0.
        Treated as on-boundary, targets the next beat.
        """
        s, _, _, _ = make_scheduler(
            beat_state=FakeBeatState(
                bpm=120, beat_phase=1.1, bar_position=1.1, beat_count=0
            ),
        )
        delay = s._compute_apply_delay(s._tempo_sync.get_beat_state())
        # Clamped to 1.0, beats_until = 0.0 < 0.01, wraps to 1.0 beat = 0.5s
        assert delay == pytest.approx(0.5, abs=1e-6)

    def test_beat_phase_negative(self):
        """Negative beat_phase is invalid and gets clamped to 0.0."""
        s, _, _, _ = make_scheduler(
            beat_state=FakeBeatState(
                bpm=120, beat_phase=-0.5, bar_position=-0.5, beat_count=0
            ),
        )
        delay = s._compute_apply_delay(s._tempo_sync.get_beat_state())
        # Clamped to 0.0, beats_until = 1.0 (not < 0.01), delay = 0.5s
        assert delay == pytest.approx(0.5, abs=1e-6)

    def test_bar_mode_with_beat_count_not_aligned_to_bar(self):
        """beat_count=5, beats_per_bar=4: beat 5 is beat 1 of bar 2."""
        s, _, _, _ = make_scheduler(
            beat_state=FakeBeatState(
                bpm=120, beat_phase=0.0, bar_position=1.0, beat_count=5
            ),
            quantize_mode="bar",
        )
        delay = s._compute_apply_delay(s._tempo_sync.get_beat_state())
        # position_in_bar = 5 % 4 + 0 = 1.0, beats_until = 3.0
        assert delay == pytest.approx(1.5, abs=1e-6)

    def test_2bar_boundary_exact(self):
        """Exactly on a 2-bar boundary: beat_count=8 is cycle start, should
        target the NEXT 2-bar boundary (8 beats away).
        """
        s, _, _, _ = make_scheduler(
            beat_state=FakeBeatState(
                bpm=120, beat_phase=0.0, bar_position=0.0, beat_count=8
            ),
            quantize_mode="2_bar",
        )
        delay = s._compute_apply_delay(s._tempo_sync.get_beat_state())
        # cycle=8, position = 8%8 + 0 = 0, beats_until = 8 (not < 0.01, no wrap)
        assert delay == pytest.approx(4.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Scheduling behavior (with actual timer callbacks)
# ---------------------------------------------------------------------------


class TestScheduleBehavior:
    """Tests that schedule() triggers callbacks correctly."""

    def test_mode_none_applies_immediately(self):
        s, applied, notifications, _ = make_scheduler(
            beat_state=FakeBeatState(
                bpm=120, beat_phase=0.5, bar_position=0.5, beat_count=0
            ),
            quantize_mode="none",
        )
        s.schedule({"prompt": "hello"})
        assert len(applied) == 1
        assert applied[0] == {"prompt": "hello"}
        # No "change_scheduled" or "change_applied" notification for immediate
        assert not any(n.get("type") == "change_scheduled" for n in notifications)

    def test_no_beat_state_applies_immediately(self):
        s, applied, _, tempo = make_scheduler(quantize_mode="beat")
        tempo._beat_state = None
        s.schedule({"prompt": "hello"})
        assert len(applied) == 1

    def test_large_lookahead_schedules_later_boundary(self):
        """When lookahead exceeds time to next boundary, the scheduler
        targets a later boundary instead of applying immediately.
        """
        s, applied, notifications, _ = make_scheduler(
            beat_state=FakeBeatState(
                bpm=120, beat_phase=0.9, bar_position=0.9, beat_count=0
            ),
            lookahead_ms=500,
        )
        s.schedule({"prompt": "hello"})
        # Should NOT apply immediately; should schedule for a later boundary
        assert len(applied) == 0
        assert any(n.get("type") == "change_scheduled" for n in notifications)

    def test_scheduled_change_fires_after_delay(self):
        """With a short boundary distance, verify the timer fires."""
        # 120 BPM, phase 0.5 => 0.25s to next beat, no lookahead
        s, applied, notifications, _ = make_scheduler(
            beat_state=FakeBeatState(
                bpm=120, beat_phase=0.5, bar_position=0.5, beat_count=0
            ),
        )
        s.schedule({"prompt": "hello"})
        # Not applied yet
        assert len(applied) == 0
        assert any(n.get("type") == "change_scheduled" for n in notifications)

        # Wait for timer to fire (0.25s + tolerance)
        time.sleep(0.35)
        assert len(applied) == 1
        assert applied[0] == {"prompt": "hello"}
        assert any(n.get("type") == "change_applied" for n in notifications)

    def test_accumulation_merges_params(self):
        """Two rapid schedules should merge into one application."""
        s, applied, _, _ = make_scheduler(
            beat_state=FakeBeatState(
                bpm=60, beat_phase=0.0, bar_position=0.0, beat_count=0
            ),
            # 60 BPM = 1s per beat, so timer is 1s away. Plenty of time to schedule twice.
        )
        s.schedule({"noise_scale": 0.5})
        s.schedule({"prompt": "dog"})
        # Wait for single timer fire
        time.sleep(1.2)
        # Should have been applied exactly once with merged params
        assert len(applied) == 1
        assert applied[0] == {"noise_scale": 0.5, "prompt": "dog"}

    def test_accumulation_later_value_wins(self):
        """When the same key is scheduled twice, last write wins."""
        s, applied, _, _ = make_scheduler(
            beat_state=FakeBeatState(
                bpm=60, beat_phase=0.0, bar_position=0.0, beat_count=0
            ),
        )
        s.schedule({"prompt": "cat"})
        s.schedule({"prompt": "dog"})
        time.sleep(1.2)
        assert len(applied) == 1
        assert applied[0]["prompt"] == "dog"

    def test_accumulation_preserves_target_boundary(self):
        """When a timer is already pending, subsequent schedule() calls merge
        params without recomputing the delay or recreating the timer. The
        first call's target boundary is preserved.
        """
        applied = []
        notifications = []
        tempo = FakeTempoSync(
            FakeBeatState(bpm=120, beat_phase=0.0, bar_position=0.0, beat_count=0),
        )
        s = ParameterScheduler(
            tempo_sync=tempo,
            apply_callback=lambda p: applied.append(p),
            notification_callback=lambda m: notifications.append(m),
        )
        s.quantize_mode = "beat"

        # First schedule targets beat 1 (0.5s away at 120 BPM)
        s.schedule({"a": 1})
        scheduled_msgs = [n for n in notifications if n["type"] == "change_scheduled"]
        assert len(scheduled_msgs) == 1
        assert scheduled_msgs[0]["delay_ms"] == 500

        # Advance beat state as if time passed
        tempo._beat_state = FakeBeatState(
            bpm=120, beat_phase=0.0, bar_position=1.0, beat_count=1
        )
        # Second schedule merges into existing timer, no new "change_scheduled"
        s.schedule({"b": 2})
        scheduled_msgs = [n for n in notifications if n["type"] == "change_scheduled"]
        assert len(scheduled_msgs) == 1  # Still just the first one

        # Wait for the original timer to fire (0.5s from first schedule)
        time.sleep(0.7)
        assert len(applied) == 1
        assert applied[0] == {"a": 1, "b": 2}

    def test_cancel_pending_prevents_application(self):
        s, applied, _, _ = make_scheduler(
            beat_state=FakeBeatState(
                bpm=60, beat_phase=0.0, bar_position=0.0, beat_count=0
            ),
        )
        s.schedule({"prompt": "hello"})
        assert len(applied) == 0
        s.cancel_pending()
        time.sleep(1.2)
        assert len(applied) == 0

    def test_schedule_after_cancel_works(self):
        s, applied, _, _ = make_scheduler(
            beat_state=FakeBeatState(
                bpm=120, beat_phase=0.5, bar_position=0.5, beat_count=0
            ),
        )
        s.schedule({"prompt": "first"})
        s.cancel_pending()
        s.schedule({"prompt": "second"})
        time.sleep(0.35)
        assert len(applied) == 1
        assert applied[0] == {"prompt": "second"}


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


class TestConcurrency:
    """Probe for race conditions under concurrent schedule() calls."""

    def test_many_concurrent_schedules(self):
        """Fire 50 schedule() calls from different threads. Should not crash,
        and exactly one application should happen at the boundary.
        """
        s, applied, _, _ = make_scheduler(
            beat_state=FakeBeatState(
                bpm=120, beat_phase=0.5, bar_position=0.5, beat_count=0
            ),
        )
        barrier = threading.Barrier(50)

        def do_schedule(i):
            barrier.wait()
            s.schedule({"idx": i})

        threads = [threading.Thread(target=do_schedule, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Wait for the timer to fire
        time.sleep(0.5)
        # Exactly one application with merged params
        assert len(applied) == 1
        # The "idx" key should exist (last writer wins, value is arbitrary)
        assert "idx" in applied[0]

    def test_schedule_during_apply(self):
        """Schedule a new change while _apply_pending is executing the callback.
        The new change should schedule a fresh timer, not be lost.
        """
        apply_event = threading.Event()
        apply_done = threading.Event()
        applied = []

        def slow_apply(params):
            applied.append(dict(params))
            apply_event.set()  # Signal that we're inside apply
            apply_done.wait(timeout=2)  # Block until test releases us

        tempo = FakeTempoSync(
            FakeBeatState(bpm=120, beat_phase=0.8, bar_position=0.8, beat_count=0),
        )
        s = ParameterScheduler(
            tempo_sync=tempo,
            apply_callback=slow_apply,
        )
        s.quantize_mode = "beat"

        # Schedule first change (0.1s delay: (1-0.8)*0.5 = 0.1s)
        s.schedule({"first": True})
        # Wait for apply to start
        assert apply_event.wait(timeout=1)

        # While apply is blocked, schedule another change
        tempo._beat_state = FakeBeatState(
            bpm=120, beat_phase=0.5, bar_position=0.5, beat_count=1
        )
        s.schedule({"second": True})

        # Release the slow apply
        apply_done.set()

        # Wait for the second timer to fire
        time.sleep(0.5)

        # Both should have been applied (in separate apply calls)
        assert len(applied) == 2
        assert applied[0] == {"first": True}
        assert applied[1] == {"second": True}


# ---------------------------------------------------------------------------
# Property validation
# ---------------------------------------------------------------------------


class TestPropertyValidation:
    def test_invalid_quantize_mode_rejected(self):
        s, _, _, _ = make_scheduler(quantize_mode="none")
        s.quantize_mode = "invalid_mode"
        assert s.quantize_mode == "none"

    def test_valid_quantize_modes_accepted(self):
        s, _, _, _ = make_scheduler(quantize_mode="none")
        for mode in ("none", "beat", "bar", "2_bar", "4_bar"):
            s.quantize_mode = mode
            assert s.quantize_mode == mode

    def test_negative_lookahead_clamped(self):
        s, _, _, _ = make_scheduler()
        s.lookahead_ms = -100
        assert s.lookahead_ms == 0.0

    def test_lookahead_accepts_int(self):
        s, _, _, _ = make_scheduler()
        s.lookahead_ms = 200
        assert s.lookahead_ms == 200.0

    def test_notification_callback_exception_does_not_crash(self):
        """If the notification callback raises, schedule should not crash."""
        tempo = FakeTempoSync(
            FakeBeatState(bpm=120, beat_phase=0.5, bar_position=0.5, beat_count=0),
        )
        applied = []
        s = ParameterScheduler(
            tempo_sync=tempo,
            apply_callback=lambda p: applied.append(p),
            notification_callback=MagicMock(side_effect=RuntimeError("boom")),
        )
        s.quantize_mode = "beat"
        # Should not raise despite notification callback exploding
        s.schedule({"prompt": "test"})
        time.sleep(0.35)
        assert len(applied) == 1
