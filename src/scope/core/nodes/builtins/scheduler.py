"""Scheduler node — time-based trigger sequencer.

Fires named triggers at specified times. Supports looping, dynamic
output ports derived from the trigger list, and an internal monotonic
clock running at ~200 Hz for sub-frame timing accuracy.

Port semantics
--------------
- Static inputs:  ``start`` (trigger), ``reset`` (trigger)
- Static outputs: ``tick`` (number), ``elapsed`` (number), ``is_playing`` (boolean)
- Dynamic outputs: one per unique trigger ``port_name``, emitting an
  incrementing counter each time the trigger fires.  Counters (rather
  than booleans) allow downstream nodes to detect every firing even
  when multiple events coincide within a single frame.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, ClassVar

from ..base import BaseNode, NodeDefinition, NodePort

logger = logging.getLogger(__name__)

# Internal tick interval — 5 ms gives ~200 Hz resolution while keeping
# CPU usage negligible.
_TICK_INTERVAL = 0.005

# Throttle elapsed-time broadcasts to avoid flooding downstream nodes.
_ELAPSED_BROADCAST_INTERVAL = 0.05  # 50 ms → ~20 Hz


class SchedulerNode(BaseNode):
    """Time-based trigger scheduler.

    Configured via the ``triggers``, ``loop``, and ``duration`` parameters.
    Each trigger has a ``time`` (seconds) and ``port_name`` (output port).
    When elapsed time reaches a trigger's time the corresponding output
    fires with an incrementing counter.
    """

    node_type_id: ClassVar[str] = "scheduler"

    def __init__(self, node_id: str, config: dict[str, Any] | None = None):
        super().__init__(node_id, config)
        self._lock = threading.Lock()
        self._playing = False
        self._start_time: float | None = None
        self._elapsed = 0.0
        self._tick_count = 0
        self._fire_counts: dict[str, int] = {}
        self._fired_keys: set[str] = set()
        self._triggers: list[dict] = []
        self._loop = False
        self._duration = 0.0
        self._last_elapsed_broadcast = 0.0
        self._timer_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        # Accumulated outputs that will be returned on next execute()
        self._pending_outputs: dict[str, Any] = {}
        self._pending_lock = threading.Lock()

    @classmethod
    def get_definition(cls) -> NodeDefinition:
        return NodeDefinition(
            node_type_id=cls.node_type_id,
            display_name="Scheduler",
            category="timing",
            description=(
                "Time-based trigger scheduler. Add trigger points and "
                "connect them to other nodes to drive timed actions."
            ),
            continuous=True,
            inputs=[
                NodePort(
                    name="start",
                    port_type="trigger",
                    required=False,
                    description="Play / pause toggle",
                ),
                NodePort(
                    name="reset",
                    port_type="trigger",
                    required=False,
                    description="Reset elapsed time and trigger state",
                ),
            ],
            outputs=[
                NodePort(name="tick", port_type="number", description="Tick counter"),
                NodePort(
                    name="elapsed",
                    port_type="number",
                    description="Elapsed time (seconds)",
                ),
                NodePort(
                    name="is_playing",
                    port_type="boolean",
                    description="Whether the scheduler is playing",
                ),
            ],
        )

    # ------------------------------------------------------------------
    # Internal timer
    # ------------------------------------------------------------------

    def _start_timer(self):
        if self._timer_thread is not None and self._timer_thread.is_alive():
            return
        self._stop_event.clear()
        self._timer_thread = threading.Thread(target=self._timer_loop, daemon=True)
        self._timer_thread.start()

    def _stop_timer(self):
        self._stop_event.set()
        if self._timer_thread is not None:
            self._timer_thread.join(timeout=1.0)
            self._timer_thread = None

    def _timer_loop(self):
        """Background thread: checks triggers at ~200 Hz."""
        while not self._stop_event.is_set():
            with self._lock:
                if self._playing and self._start_time is not None:
                    self._elapsed = time.monotonic() - self._start_time
                    self._check_triggers()
            self._stop_event.wait(_TICK_INTERVAL)

    def _check_triggers(self):
        """Fire any triggers whose time has been reached (called under lock)."""
        for trig in self._triggers:
            t = trig["time"]
            port = trig["port_name"]
            key = f"{port}@{t}"
            if t <= self._elapsed and key not in self._fired_keys:
                self._fired_keys.add(key)
                self._fire_counts.setdefault(port, 0)
                self._fire_counts[port] += 1
                self._tick_count += 1
                with self._pending_lock:
                    self._pending_outputs[port] = self._fire_counts[port]
                    self._pending_outputs["tick"] = self._tick_count

        # Broadcast elapsed at throttled rate
        now = time.monotonic()
        if now - self._last_elapsed_broadcast >= _ELAPSED_BROADCAST_INTERVAL:
            self._last_elapsed_broadcast = now
            with self._pending_lock:
                self._pending_outputs["elapsed"] = round(self._elapsed, 3)
                self._pending_outputs["is_playing"] = self._playing

        # Handle loop / auto-stop
        if self._duration > 0 and self._elapsed >= self._duration:
            if self._loop:
                self._reset_state()
                self._start_time = time.monotonic()
            else:
                # Check if all triggers fired
                all_fired = all(
                    f"{t['port_name']}@{t['time']}" in self._fired_keys
                    for t in self._triggers
                )
                if all_fired:
                    self._playing = False
                    with self._pending_lock:
                        self._pending_outputs["is_playing"] = False

    def _reset_state(self):
        """Reset elapsed time and trigger state (called under lock)."""
        self._elapsed = 0.0
        self._fired_keys.clear()
        self._fire_counts.clear()
        self._tick_count = 0

    # ------------------------------------------------------------------
    # Node interface
    # ------------------------------------------------------------------

    def execute(self, inputs: dict[str, Any], **kwargs) -> dict[str, Any]:
        # Apply configuration from parameters
        triggers = kwargs.get("triggers", self.config.get("triggers", []))
        loop = kwargs.get("loop", self.config.get("loop", False))
        duration = kwargs.get("duration", self.config.get("duration", 0.0))

        with self._lock:
            self._triggers = triggers if isinstance(triggers, list) else []
            self._loop = bool(loop)
            self._duration = float(duration)

        # Handle start/reset inputs
        start_val = inputs.get("start")
        reset_val = inputs.get("reset")

        if reset_val is not None:
            with self._lock:
                self._reset_state()
                self._start_time = time.monotonic() if self._playing else None

        if start_val is not None:
            with self._lock:
                self._playing = not self._playing
                if self._playing:
                    self._start_time = time.monotonic() - self._elapsed
                    self._start_timer()
                else:
                    pass  # Timer keeps running but won't advance elapsed

        # Auto-start on first execute if not yet started
        if not self._playing and self._start_time is None and self._triggers:
            with self._lock:
                self._playing = True
                self._start_time = time.monotonic()
                self._start_timer()

        # Collect pending outputs
        with self._pending_lock:
            outputs = dict(self._pending_outputs)
            self._pending_outputs.clear()

        # Always include current state
        with self._lock:
            outputs.setdefault("elapsed", round(self._elapsed, 3))
            outputs.setdefault("is_playing", self._playing)
            outputs.setdefault("tick", self._tick_count)

        # Yield to avoid busy-loop
        time.sleep(_TICK_INTERVAL * 2)

        return outputs
