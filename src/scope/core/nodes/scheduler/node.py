"""Scheduler node implementation.

A time-based trigger scheduler that fires named trigger outputs at specific
time points.  Supports an internal monotonic clock and can be extended to
use external clock sources (MIDI, Ableton Link) in the future.

Thread management is handled by the node itself: a dedicated timer thread
runs the playback loop while the main thread receives input updates.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Any

from scope.core.nodes.interface import BaseNode, ConnectorDef

if TYPE_CHECKING:
    from collections.abc import Callable

from .schema import SchedulerNodeConfig

logger = logging.getLogger(__name__)

_TICK_INTERVAL = 0.005  # 5 ms → ~200 Hz internal resolution
_ELAPSED_INTERVAL = 0.05  # 50 ms → ~20 Hz for elapsed broadcasts


class TriggerPoint:
    """A single trigger event on the scheduler."""

    __slots__ = ("time", "port_name", "_fired")

    def __init__(self, t: float, port_name: str) -> None:
        self.time = t
        self.port_name = port_name
        self._fired = False

    def reset(self) -> None:
        self._fired = False


class SchedulerNode(BaseNode):
    """Time-based trigger scheduler node.

    Configuration (via ``update_config``):
    - ``triggers``: list of ``{"time": float, "port_name": str}``
    - ``loop``: bool — whether to loop
    - ``duration``: float — total duration in seconds (for looping)
    - ``clock_source``: ``"internal"`` (default; MIDI/Link planned)

    Trigger outputs emit an incrementing integer counter (1, 2, 3, …) each
    time they fire.  This is robust against React state-batching: as long as
    the frontend sees a counter change, it knows a trigger occurred.
    """

    @classmethod
    def get_config_class(cls) -> type[SchedulerNodeConfig]:
        return SchedulerNodeConfig

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup(self, emit_output: Callable[[str, Any], None]) -> None:
        self._emit = emit_output
        self._lock = threading.Lock()

        self._triggers: list[TriggerPoint] = []
        self._dynamic_outputs: list[ConnectorDef] = []
        self._loop = False
        self._duration = 0.0

        self._stream_active = True
        self._playing = False
        self._start_time: float | None = None
        self._elapsed = 0.0

        self._fire_counts: dict[str, int] = {}
        self._tick_count = 0

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def teardown(self) -> None:
        self._stream_active = False
        self._stop_playback()

    def on_stream_stop(self) -> None:
        self._stream_active = False
        self._stop_playback()
        self._reset()

    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------

    def update_input(self, name: str, value: Any) -> None:
        if name == "start":
            if value:
                if self._playing:
                    self._stop_playback()
                else:
                    self._start_playback()
        elif name == "reset":
            if value:
                self._reset()

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def update_config(self, config: dict[str, Any]) -> None:
        with self._lock:
            if "triggers" in config:
                self._triggers = [
                    TriggerPoint(t["time"], t["port_name"]) for t in config["triggers"]
                ]
                self._rebuild_dynamic_outputs()

            if "loop" in config:
                self._loop = bool(config["loop"])

            if "duration" in config:
                self._duration = float(config["duration"])

    def _rebuild_dynamic_outputs(self) -> None:
        """Rebuild the dynamic output port list from current triggers."""
        seen: set[str] = set()
        static_names = {"tick", "elapsed", "is_playing"}
        ports: list[ConnectorDef] = []
        for tp in self._triggers:
            if tp.port_name not in seen and tp.port_name not in static_names:
                ports.append(
                    ConnectorDef(
                        name=tp.port_name,
                        type="trigger",
                        direction="output",
                    )
                )
                seen.add(tp.port_name)
        self._dynamic_outputs = ports

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def get_state(self) -> dict[str, Any]:
        with self._lock:
            return {
                "is_playing": self._playing,
                "elapsed": round(self._elapsed, 3),
                "loop": self._loop,
                "duration": self._duration,
                "triggers": [
                    {"time": tp.time, "port_name": tp.port_name}
                    for tp in self._triggers
                ],
            }

    def get_current_ports(self) -> dict[str, list[ConnectorDef]] | None:
        cfg = self.get_config_class()
        with self._lock:
            return {
                "inputs": list(cfg.inputs),
                "outputs": list(cfg.outputs) + list(self._dynamic_outputs),
            }

    # ------------------------------------------------------------------
    # Playback engine (runs in a dedicated thread)
    # ------------------------------------------------------------------

    def _start_playback(self) -> None:
        with self._lock:
            if self._playing or not self._stream_active:
                return
            self._playing = True
            self._start_time = time.monotonic() - self._elapsed
            for tp in self._triggers:
                if tp.time > self._elapsed:
                    tp.reset()
            self._stop_event.clear()

        self._emit("is_playing", True)
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="scheduler-playback"
        )
        self._thread.start()

    def _stop_playback(self) -> None:
        with self._lock:
            if not self._playing:
                return
            self._playing = False

        self._emit("is_playing", False)

        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _reset(self) -> None:
        was_playing = self._playing
        if was_playing:
            self._stop_playback()

        with self._lock:
            self._elapsed = 0.0
            self._start_time = None
            self._fire_counts.clear()
            self._tick_count = 0
            for tp in self._triggers:
                tp.reset()

        self._emit("elapsed", 0.0)
        port_names = {tp.port_name for tp in self._triggers}
        for port_name in port_names:
            self._emit(port_name, 0)
        self._emit("tick", 0)
        if was_playing:
            self._start_playback()

    def _run_loop(self) -> None:
        """Playback loop — runs on a dedicated thread."""
        last_elapsed_emit = 0.0
        loop_count = 0
        try:
            while not self._stop_event.is_set():
                now = time.monotonic()

                with self._lock:
                    if self._start_time is None:
                        break
                    self._elapsed = now - self._start_time
                    elapsed = self._elapsed

                    fired_ports: list[str] = []
                    for tp in self._triggers:
                        if not tp._fired and tp.time <= elapsed:
                            tp._fired = True
                            fired_ports.append(tp.port_name)

                    should_auto_stop = False
                    if self._duration > 0 and elapsed >= self._duration:
                        if self._loop:
                            loop_count += 1
                            self._elapsed = 0.0
                            self._start_time = now
                            for tp in self._triggers:
                                tp.reset()
                        else:
                            should_auto_stop = all(tp._fired for tp in self._triggers)

                if fired_ports:
                    for port in fired_ports:
                        count = self._fire_counts.get(port, 0) + 1
                        self._fire_counts[port] = count
                        self._emit(port, count)
                    self._tick_count += 1
                    self._emit("tick", self._tick_count)

                if now - last_elapsed_emit >= _ELAPSED_INTERVAL:
                    self._emit("elapsed", round(elapsed, 3))
                    last_elapsed_emit = now

                if should_auto_stop:
                    break

                self._stop_event.wait(_TICK_INTERVAL)
        except Exception:
            logger.exception("Scheduler playback loop error")
        finally:
            with self._lock:
                self._playing = False
            self._emit("is_playing", False)
