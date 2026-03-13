"""Ableton Link tempo source adapter.

Uses the aalink library (async Python wrapper for Ableton Link) to join
a shared tempo session on the local network. Provides beat phase, BPM,
and bar position synchronized with other Link-enabled applications
(Ableton, Resolume, TouchDesigner, etc.).
"""

import asyncio
import logging
import threading
import time

try:
    from aalink import Link
except ImportError:
    Link = None

from ..tempo_sync import BeatState, TempoSource

logger = logging.getLogger(__name__)

POLL_INTERVAL = 0.01  # 100Hz polling


class LinkTempoSource(TempoSource):
    """Ableton Link tempo source.

    Runs an asyncio task that polls Link state at ~100Hz and caches the
    latest BeatState for thread-safe access from pipeline processor threads.
    """

    def __init__(self, bpm: float = 120.0, beats_per_bar: int = 4):
        self._initial_bpm = bpm
        self._beats_per_bar = beats_per_bar
        self._link: Link | None = None
        self._poll_task: asyncio.Task | None = None

        self._cached_state: BeatState | None = None
        self._state_lock = threading.Lock()

    @property
    def name(self) -> str:
        return "link"

    @property
    def num_peers(self) -> int:
        if self._link is not None:
            return self._link.num_peers
        return 0

    def get_beat_state(self) -> BeatState | None:
        with self._state_lock:
            return self._cached_state

    async def start(self) -> None:
        if Link is None:
            raise ImportError(
                "aalink is not installed. Install with: uv sync --group link"
            )
        loop = asyncio.get_running_loop()
        self._link = Link(self._initial_bpm, loop)
        self._link.quantum = self._beats_per_bar
        self._link.enabled = True

        self._poll_task = asyncio.ensure_future(self._poll_loop())
        logger.info(
            f"Ableton Link started: bpm={self._initial_bpm}, "
            f"peers={self._link.num_peers}"
        )

    async def stop(self) -> None:
        if self._poll_task is not None:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            self._poll_task = None

        if self._link is not None:
            self._link.enabled = False
            self._link = None

        logger.info("Ableton Link stopped")

    async def _poll_loop(self) -> None:
        """Poll Link state at high frequency and cache the result."""
        try:
            while True:
                if self._link is not None:
                    beat = self._link.beat
                    tempo = self._link.tempo
                    playing = self._link.playing

                    beat_phase = beat % 1.0
                    bar_position = beat % self._beats_per_bar
                    beat_count = int(beat)

                    state = BeatState(
                        bpm=tempo,
                        beat_phase=beat_phase,
                        bar_position=bar_position,
                        beat_count=beat_count,
                        is_playing=playing,
                        timestamp=time.time(),
                        source="link",
                    )
                    with self._state_lock:
                        self._cached_state = state

                await asyncio.sleep(POLL_INTERVAL)
        except asyncio.CancelledError:
            pass

    def set_tempo(self, bpm: float) -> None:
        """Set the Link session tempo (makes Scope the tempo leader)."""
        if self._link is not None:
            self._link.tempo = bpm
