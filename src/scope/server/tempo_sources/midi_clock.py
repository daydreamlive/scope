"""MIDI clock tempo source adapter.

Receives MIDI clock messages (24 pulses per quarter note) from an external
device or DAW and derives BPM, beat phase, and bar position. Uses mido with
the python-rtmidi backend for low-latency MIDI I/O.
"""

import logging
import threading
import time

try:
    import mido
except ImportError:
    mido = None

from ..tempo_sync import BeatState, TempoSource

logger = logging.getLogger(__name__)

PPQN = 24  # Pulses per quarter note (MIDI standard)
EMA_ALPHA = 0.15  # Exponential moving average smoothing factor for BPM


class MIDIClockTempoSource(TempoSource):
    """MIDI clock tempo source.

    Listens on a MIDI input port for clock ticks (0xF8), Start (0xFA),
    Stop (0xFC), and Continue (0xFB) messages. Derives BPM from the
    average inter-tick interval using an exponential moving average.
    """

    def __init__(
        self,
        device_name: str | None = None,
        beats_per_bar: int = 4,
    ):
        self._device_name = device_name
        self._beats_per_bar = beats_per_bar

        self._port: mido.ports.BaseInput | None = None
        self._thread: threading.Thread | None = None
        self._running = False

        self._tick_count = 0
        self._last_tick_time: float | None = None
        self._ema_interval: float | None = None
        self._is_playing = False

        self._cached_state: BeatState | None = None
        self._state_lock = threading.Lock()

    @property
    def name(self) -> str:
        return "midi_clock"

    def get_beat_state(self) -> BeatState | None:
        with self._state_lock:
            return self._cached_state

    async def start(self) -> None:
        if mido is None:
            raise ImportError(
                "mido is not installed. Install with: uv sync --extra midi"
            )
        if self._device_name:
            self._port = mido.open_input(self._device_name)
        else:
            available = mido.get_input_names()
            if not available:
                raise RuntimeError("No MIDI input devices found")
            self._device_name = available[0]
            self._port = mido.open_input(self._device_name)
            logger.info("Auto-selected MIDI device: %s", self._device_name)

        self._running = True
        self._thread = threading.Thread(
            target=self._listen_loop, daemon=True, name="midi-clock-listener"
        )
        self._thread.start()
        logger.info("MIDI clock started on device: %s", self._device_name)

    async def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._port is not None:
            self._port.close()
            self._port = None
        logger.info("MIDI clock stopped")

    def _listen_loop(self) -> None:
        """Read MIDI messages and update beat state."""
        while self._running and self._port is not None:
            for msg in self._port.iter_pending():
                if msg.type == "clock":
                    self._on_clock_tick()
                elif msg.type == "start":
                    self._on_start()
                elif msg.type == "stop":
                    self._on_stop()
                elif msg.type == "continue":
                    self._on_continue()
            # Brief sleep to avoid busy-waiting while still being responsive
            # iter_pending() is non-blocking so we need our own throttle
            threading.Event().wait(0.001)

    def _on_clock_tick(self) -> None:
        now = time.monotonic()
        self._tick_count += 1

        if self._last_tick_time is not None:
            interval = now - self._last_tick_time
            if interval > 0:
                if self._ema_interval is None:
                    self._ema_interval = interval
                else:
                    self._ema_interval = (
                        EMA_ALPHA * interval + (1 - EMA_ALPHA) * self._ema_interval
                    )
                logger.debug(
                    "MIDI tick #%d: interval=%.6fs ema=%.6fs bpm=%.1f",
                    self._tick_count,
                    interval,
                    self._ema_interval,
                    60.0 / (self._ema_interval * PPQN),
                )

        self._last_tick_time = now

        if self._ema_interval is not None and self._ema_interval > 0:
            bpm = 60.0 / (self._ema_interval * PPQN)
            beat_float = self._tick_count / PPQN
            beat_phase = (self._tick_count % PPQN) / PPQN
            bar_ticks = PPQN * self._beats_per_bar
            bar_position = (self._tick_count % bar_ticks) / PPQN

            state = BeatState(
                bpm=bpm,
                beat_phase=beat_phase,
                bar_position=bar_position,
                beat_count=int(beat_float),
                is_playing=self._is_playing,
                timestamp=time.time(),
                source="midi_clock",
            )
            with self._state_lock:
                self._cached_state = state

    def _on_start(self) -> None:
        self._tick_count = 0
        self._last_tick_time = None
        self._ema_interval = None
        self._is_playing = True
        with self._state_lock:
            self._cached_state = None
        logger.info("MIDI Start received")

    def _on_stop(self) -> None:
        self._is_playing = False
        with self._state_lock:
            self._cached_state = None
        logger.info("MIDI Stop received")

    def _on_continue(self) -> None:
        self._is_playing = True
        self._last_tick_time = None
        with self._state_lock:
            self._cached_state = None
        logger.info("MIDI Continue received")
