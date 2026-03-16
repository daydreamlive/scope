"""Tempo/beat synchronization manager.

Provides a unified beat clock that abstracts over multiple tempo sources
(Ableton Link, MIDI clock, client-forwarded beat state). Exposes current
beat state to pipeline processors for injection into pipeline kwargs.
"""

import abc
import asyncio
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Literal

logger = logging.getLogger(__name__)

BEAT_STATE_KEYS = frozenset(
    {"bpm", "beat_phase", "bar_position", "beat_count", "is_playing"}
)


def get_beat_boundary(rate: str, beat_count: int, beats_per_bar: int) -> int:
    """Return an integer boundary index for the current beat position.

    The boundary increments each time we cross the configured rate's period.
    """
    if rate == "beat":
        return beat_count
    elif rate == "bar":
        return beat_count // max(beats_per_bar, 1)
    elif rate == "2_bar":
        return beat_count // max(beats_per_bar * 2, 1)
    elif rate == "4_bar":
        return beat_count // max(beats_per_bar * 4, 1)
    return -1


CLIENT_BEAT_STATE_STALE_SECONDS = 2.0


@dataclass(frozen=True)
class BeatState:
    """Snapshot of the current beat clock state."""

    bpm: float
    beat_phase: float
    bar_position: float
    beat_count: int
    is_playing: bool
    timestamp: float
    source: str


class TempoSource(abc.ABC):
    """Abstract base for tempo sources."""

    @property
    @abc.abstractmethod
    def name(self) -> str: ...

    @abc.abstractmethod
    def get_beat_state(self) -> BeatState | None: ...

    @abc.abstractmethod
    async def start(self) -> None: ...

    @abc.abstractmethod
    async def stop(self) -> None: ...

    def set_tempo(self, bpm: float) -> None:
        """Set the session tempo. Override in sources that support it."""
        raise RuntimeError(f"{self.name} does not support set_tempo")


class TempoSync:
    """Central tempo synchronization manager.

    Supports two input paths that converge at get_beat_state():
      1. Server-side tempo source (Link or MIDI clock) for local mode
      2. Client-forwarded beat state via data channel for cloud mode

    Client-forwarded state takes priority when fresh (received within the
    last CLIENT_BEAT_STATE_STALE_SECONDS). This allows cloud mode to work
    without any server-side tempo source.
    """

    def __init__(self, beats_per_bar: int = 4):
        self._source: TempoSource | None = None
        self._source_lock = threading.Lock()
        self._beats_per_bar = beats_per_bar

        self._client_beat_state: BeatState | None = None
        self._client_state_lock = threading.Lock()

        self._enabled = False
        self._enabled_lock = threading.Lock()
        self._notification_task: asyncio.Task | None = None
        self._notification_sessions: list[Any] = []
        self._notification_lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        with self._enabled_lock:
            return self._enabled

    @property
    def source_name(self) -> str | None:
        with self._source_lock:
            if self._source is not None:
                return self._source.name
        return None

    @property
    def beats_per_bar(self) -> int:
        return self._beats_per_bar

    def get_beat_state(self) -> BeatState | None:
        """Get the current beat state. Thread-safe.

        Returns None when disabled. Otherwise returns client-forwarded
        state if fresh, falling back to the server-side tempo source.
        """
        with self._enabled_lock:
            if not self._enabled:
                return None

        with self._client_state_lock:
            client_state = self._client_beat_state

        if client_state is not None:
            age = time.time() - client_state.timestamp
            if age < CLIENT_BEAT_STATE_STALE_SECONDS:
                return client_state

        with self._source_lock:
            if self._source is not None:
                return self._source.get_beat_state()

        return None

    def update_client_beat_state(self, params: dict) -> dict:
        """Extract and cache beat state from client-forwarded parameters.

        Returns the params dict with beat state keys removed (to avoid
        them being forwarded as regular pipeline parameters, since they
        are injected separately by PipelineProcessor).
        """
        if not BEAT_STATE_KEYS.intersection(params):
            return params

        remaining = {}
        beat_data = {}
        for k, v in params.items():
            if k in BEAT_STATE_KEYS:
                beat_data[k] = v
            else:
                remaining[k] = v

        if "bpm" in beat_data and "beat_phase" in beat_data:
            beat_phase = float(beat_data["beat_phase"])
            state = BeatState(
                bpm=float(beat_data["bpm"]),
                beat_phase=beat_phase,
                bar_position=float(beat_data.get("bar_position", beat_phase)),
                beat_count=int(beat_data.get("beat_count", 0)),
                is_playing=bool(beat_data.get("is_playing", True)),
                timestamp=time.time(),
                source="client",
            )
            with self._client_state_lock:
                self._client_beat_state = state

        return remaining

    async def enable(
        self,
        source_type: Literal["link", "midi_clock"],
        bpm: float = 120.0,
        midi_device: str | None = None,
        beats_per_bar: int = 4,
    ) -> None:
        """Enable tempo sync with the specified source."""
        await self.disable()

        self._beats_per_bar = beats_per_bar

        if source_type == "link":
            source = self._create_link_source(bpm)
        elif source_type == "midi_clock":
            source = self._create_midi_clock_source(midi_device, beats_per_bar)
        else:
            raise ValueError(f"Unknown tempo source type: {source_type}")

        if source is None:
            hints = {
                "link": "Install with: uv sync --extra link",
                "midi_clock": "Install with: uv sync --extra midi",
            }
            raise RuntimeError(
                f"Failed to create {source_type} tempo source. "
                f"{hints.get(source_type, '')}"
            )

        await source.start()

        with self._source_lock:
            self._source = source
        with self._enabled_lock:
            self._enabled = True

        logger.info("Tempo sync enabled: source=%s, bpm=%s", source_type, bpm)

        self._start_notifications()

    def set_tempo(self, bpm: float) -> None:
        """Set the session tempo. Only supported by some sources (e.g. Link)."""
        with self._source_lock:
            if self._source is None:
                raise RuntimeError("No active tempo source")
            self._source.set_tempo(bpm)

    async def disable(self) -> None:
        """Disable tempo sync and stop the current source."""
        with self._enabled_lock:
            self._enabled = False
        self._stop_notifications()

        with self._client_state_lock:
            self._client_beat_state = None

        with self._source_lock:
            source = self._source
            self._source = None

        if source is not None:
            await source.stop()
            logger.info("Tempo sync disabled (was: %s)", source.name)

    async def stop(self) -> None:
        """Shutdown the tempo sync manager."""
        await self.disable()

    def get_status(self) -> dict:
        """Get current tempo sync status for the REST API."""
        beat_state = self.get_beat_state()
        source_info = {}

        with self._source_lock:
            if self._source is not None:
                source_info["type"] = self._source.name
                if hasattr(self._source, "num_peers"):
                    source_info["num_peers"] = self._source.num_peers

        return {
            "enabled": self.enabled,
            "source": source_info if source_info else None,
            "beats_per_bar": self._beats_per_bar,
            "beat_state": {
                "bpm": beat_state.bpm,
                "beat_phase": beat_state.beat_phase,
                "bar_position": beat_state.bar_position,
                "beat_count": beat_state.beat_count,
                "is_playing": beat_state.is_playing,
                "source": beat_state.source,
            }
            if beat_state
            else None,
        }

    def get_available_sources(self) -> dict:
        """Get available tempo sources and their capabilities."""
        sources = {}

        try:
            import aalink  # noqa: F401

            sources["link"] = {"available": True, "name": "Ableton Link"}
        except ImportError:
            sources["link"] = {
                "available": False,
                "name": "Ableton Link",
                "install_hint": "uv sync --extra link",
            }

        try:
            import mido  # noqa: F401

            devices = []
            try:
                devices = list(mido.get_input_names())
            except Exception:
                pass
            sources["midi_clock"] = {
                "available": True,
                "name": "MIDI Clock",
                "devices": devices,
            }
        except ImportError:
            sources["midi_clock"] = {
                "available": False,
                "name": "MIDI Clock",
                "install_hint": "uv sync --extra midi",
            }

        return sources

    def register_notification_session(self, notification_sender: Any) -> None:
        """Register a WebRTC session to receive tempo notifications."""
        with self._notification_lock:
            self._notification_sessions.append(notification_sender)

    def unregister_notification_session(self, notification_sender: Any) -> None:
        """Unregister a WebRTC session from tempo notifications."""
        with self._notification_lock:
            self._notification_sessions = [
                s for s in self._notification_sessions if s is not notification_sender
            ]

    def _start_notifications(self) -> None:
        """Start the background task that pushes beat state to frontend sessions."""
        if self._notification_task is not None:
            return

        try:
            loop = asyncio.get_running_loop()
            self._notification_task = loop.create_task(self._notification_loop())
        except RuntimeError:
            logger.warning("No running event loop, skipping tempo notifications")

    def _stop_notifications(self) -> None:
        """Stop the notification background task."""
        if self._notification_task is not None:
            self._notification_task.cancel()
            self._notification_task = None

    async def _notification_loop(self) -> None:
        """Push tempo updates to registered sessions at ~15Hz."""
        try:
            while True:
                beat_state = self.get_beat_state()
                if beat_state is not None:
                    message = {
                        "type": "tempo_update",
                        "bpm": round(beat_state.bpm, 2),
                        "beat_phase": round(beat_state.beat_phase, 4),
                        "bar_position": round(beat_state.bar_position, 4),
                        "beat_count": beat_state.beat_count,
                        "is_playing": beat_state.is_playing,
                    }
                    dead: list[Any] = []
                    with self._notification_lock:
                        for sender in self._notification_sessions:
                            try:
                                sender.call(message)
                            except Exception:
                                dead.append(sender)
                    if dead:
                        with self._notification_lock:
                            self._notification_sessions = [
                                s for s in self._notification_sessions if s not in dead
                            ]
                        logger.debug("Pruned %d dead notification sender(s)", len(dead))
                await asyncio.sleep(1.0 / 15.0)
        except asyncio.CancelledError:
            pass

    def _create_link_source(self, bpm: float) -> TempoSource | None:
        try:
            from .tempo_sources.link import LinkTempoSource

            return LinkTempoSource(bpm=bpm, beats_per_bar=self._beats_per_bar)
        except ImportError:
            logger.error("aalink is not installed. Install with: uv sync --extra link")
            return None

    def _create_midi_clock_source(
        self, device: str | None, beats_per_bar: int
    ) -> TempoSource | None:
        try:
            from .tempo_sources.midi_clock import MIDIClockTempoSource

            return MIDIClockTempoSource(device_name=device, beats_per_bar=beats_per_bar)
        except ImportError:
            logger.error(
                "mido/python-rtmidi is not installed. "
                "Install with: uv sync --extra midi"
            )
            return None
