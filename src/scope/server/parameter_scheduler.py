"""Parameter scheduler for beat-synced parameter changes.

Instead of gating/freezing output, this module schedules parameter
application *ahead* of beat boundaries so that the visual change
lands on the beat. The user-adjustable lookahead compensates for
pipeline processing latency.
"""

import logging
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .tempo_sync import TempoSync

logger = logging.getLogger(__name__)


class ParameterScheduler:
    """Schedules discrete parameter changes to land on beat boundaries.

    When a change is scheduled:
    1. Compute the wall-clock time of the next beat boundary
    2. Subtract the lookahead (compensating for pipeline latency)
    3. Apply the parameters at that computed time via a timer

    The pipeline continues running smoothly on old parameters until
    the switch, so there is no frozen output.
    """

    def __init__(
        self,
        tempo_sync: "TempoSync",
        apply_callback: Callable[[dict], None],
        notification_callback: Callable[[dict], None] | None = None,
    ):
        self._tempo_sync = tempo_sync
        self._apply_callback = apply_callback
        self._notification_callback = notification_callback

        self._lock = threading.Lock()
        self._quantize_mode: str = "none"
        self._lookahead_ms: float = 0.0
        self._pending_params: dict | None = None
        self._pending_timer: threading.Timer | None = None

    @property
    def quantize_mode(self) -> str:
        return self._quantize_mode

    @quantize_mode.setter
    def quantize_mode(self, mode: str) -> None:
        valid = ("none", "beat", "bar", "2_bar", "4_bar")
        if mode not in valid:
            logger.warning(f"[SCHEDULER] Invalid quantize mode '{mode}', ignoring")
            return
        self.cancel_pending()
        self._quantize_mode = mode
        logger.info(f"[SCHEDULER] Quantize mode set to '{mode}'")

    @property
    def lookahead_ms(self) -> float:
        return self._lookahead_ms

    @lookahead_ms.setter
    def lookahead_ms(self, ms: float) -> None:
        self.cancel_pending()
        self._lookahead_ms = max(0.0, float(ms))
        logger.info(f"[SCHEDULER] Lookahead set to {self._lookahead_ms:.0f}ms")

    def schedule(self, params: dict) -> None:
        """Schedule a discrete parameter change for the next beat boundary.

        If quantize_mode is "none" or no beat state is available, the
        parameters are applied immediately.

        If a timer is already pending, new params are merged into the
        existing schedule without recomputing the target boundary.
        """
        if self._quantize_mode == "none":
            self._apply_callback(params)
            return

        beat_state = self._tempo_sync.get_beat_state()
        if beat_state is None:
            self._apply_callback(params)
            return

        apply_immediately = False
        scheduled_delay = 0.0

        with self._lock:
            # If a timer is already pending, merge params without
            # recomputing the target boundary
            if self._pending_timer is not None:
                if self._pending_params is not None:
                    self._pending_params.update(params)
                else:
                    self._pending_params = dict(params)
                logger.info("[SCHEDULER] Params merged into existing scheduled change")
                return

            delay = self._compute_apply_delay(beat_state)

            if delay <= 0:
                apply_immediately = True
            else:
                self._pending_params = dict(params)
                self._pending_timer = threading.Timer(delay, self._apply_pending)
                self._pending_timer.daemon = True
                self._pending_timer.start()
                scheduled_delay = delay

        if apply_immediately:
            self._apply_callback(params)
            self._notify({"type": "change_applied"})
            return

        logger.info(
            f"[SCHEDULER] Change scheduled in {scheduled_delay * 1000:.0f}ms "
            f"(mode={self._quantize_mode}, lookahead={self._lookahead_ms:.0f}ms)"
        )
        self._notify(
            {"type": "change_scheduled", "delay_ms": int(scheduled_delay * 1000)}
        )

    def cancel_pending(self) -> None:
        """Cancel any pending scheduled change."""
        with self._lock:
            if self._pending_timer is not None:
                self._pending_timer.cancel()
                self._pending_timer = None
            self._pending_params = None

    def _apply_pending(self) -> None:
        """Timer callback: apply the accumulated pending parameters."""
        with self._lock:
            params = self._pending_params
            self._pending_params = None
            self._pending_timer = None

        if params is not None:
            logger.info(f"[SCHEDULER] Applying scheduled change ({len(params)} params)")
            self._apply_callback(params)
            self._notify({"type": "change_applied"})

    def _compute_apply_delay(self, beat_state: Any) -> float:
        """Compute delay in seconds until parameters should be applied.

        Returns 0 if parameters should be applied immediately (e.g. BPM <= 0).
        Always returns >= 0; if lookahead exceeds the time to the nearest
        boundary, the target advances to the next cycle boundary.
        """
        bpm = beat_state.bpm
        if bpm <= 0:
            return 0.0

        beat_duration = 60.0 / bpm
        beats_per_bar = self._tempo_sync.beats_per_bar

        # Defensively clamp invalid beat_phase to [0.0, 1.0]
        beat_phase = max(0.0, min(beat_state.beat_phase, 1.0))
        beat_count = beat_state.beat_count

        if self._quantize_mode == "beat":
            beats_until = 1.0 - beat_phase
            cycle_beats = 1.0
        elif self._quantize_mode == "bar":
            position_in_bar = (beat_count % beats_per_bar) + beat_phase
            beats_until = beats_per_bar - position_in_bar
            cycle_beats = beats_per_bar
        elif self._quantize_mode == "2_bar":
            cycle_beats = 2 * beats_per_bar
            position = (beat_count % cycle_beats) + beat_phase
            beats_until = cycle_beats - position
        elif self._quantize_mode == "4_bar":
            cycle_beats = 4 * beats_per_bar
            position = (beat_count % cycle_beats) + beat_phase
            beats_until = cycle_beats - position
        else:
            return 0.0

        # If we're essentially on a boundary, target the next one
        if beats_until < 0.01:
            beats_until += cycle_beats

        time_until_boundary = beats_until * beat_duration
        delay = time_until_boundary - (self._lookahead_ms / 1000.0)

        # If lookahead exceeds time to this boundary, advance to a later cycle
        cycle_duration = cycle_beats * beat_duration
        while delay < 0:
            delay += cycle_duration

        return delay

    def _notify(self, message: dict) -> None:
        if self._notification_callback is not None:
            try:
                self._notification_callback(message)
            except Exception as e:
                logger.error(f"[SCHEDULER] Notification error: {e}")
