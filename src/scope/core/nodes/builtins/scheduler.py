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

import threading
import time
from typing import Any, ClassVar, TypedDict

from ..base import BaseNode, NodeDefinition, NodeParam, NodePort

# Internal tick interval — 5 ms gives ~200 Hz resolution while keeping
# CPU usage negligible.
_TICK_INTERVAL = 0.005

# Throttle elapsed-time broadcasts to avoid flooding downstream nodes.
_ELAPSED_BROADCAST_INTERVAL = 0.05  # 50 ms → ~20 Hz


class TriggerSpec(TypedDict):
    time: float
    port_name: str


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
        # Single lock guards all mutable state (including _pending_outputs)
        # so lock ordering can never deadlock.
        self._lock = threading.Lock()
        self._playing = False
        self._start_time: float | None = None
        self._elapsed = 0.0
        self._tick_count = 0
        self._fire_counts: dict[str, int] = {}
        self._fired_keys: set[tuple[str, float]] = set()
        self._triggers: list[TriggerSpec] = []
        self._loop = False
        self._duration = 0.0
        self._last_elapsed_broadcast = 0.0
        self._timer_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        # Accumulated outputs drained on next execute(). Only populated
        # when something has actually changed so downstream queues don't
        # get flooded with unchanged state.
        self._pending_outputs: dict[str, Any] = {}
        # Previous counter values on edge-triggered inputs. Any strict
        # increment toggles the corresponding action; a stale value
        # sitting on the queue is ignored.
        self._prev_start_counter: int | None = None
        self._prev_reset_counter: int | None = None

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
            params=[
                NodeParam(
                    name="triggers",
                    param_type="string",
                    default=[],
                    description="List of { time, port_name } trigger points",
                    ui={"widget": "trigger_list"},
                    convertible_to_input=False,
                ),
                NodeParam(
                    name="loop",
                    param_type="boolean",
                    default=False,
                    description="Loop when duration is reached",
                ),
                NodeParam(
                    name="duration",
                    param_type="number",
                    default=30.0,
                    description="Total duration (s); 0 disables auto-stop",
                    ui={"min": 0, "max": 3600, "step": 0.1},
                ),
            ],
        )

    @classmethod
    def get_dynamic_output_ports(cls, params: dict[str, Any]) -> set[str]:
        triggers = params.get("triggers") or []
        if not isinstance(triggers, list):
            return set()
        names: set[str] = set()
        for trig in triggers:
            if isinstance(trig, dict):
                name = trig.get("port_name")
                if isinstance(name, str) and name:
                    names.add(name)
        return names

    # ------------------------------------------------------------------
    # Internal timer
    # ------------------------------------------------------------------

    def _start_timer(self) -> None:
        if self._timer_thread is not None and self._timer_thread.is_alive():
            return
        self._stop_event.clear()
        self._timer_thread = threading.Thread(
            target=self._timer_loop,
            daemon=True,
            name=f"SchedulerNode[{self.node_id}]",
        )
        self._timer_thread.start()

    def _stop_timer(self) -> None:
        self._stop_event.set()
        t = self._timer_thread
        self._timer_thread = None
        if t is not None and t is not threading.current_thread():
            t.join(timeout=1.0)

    def _timer_loop(self) -> None:
        """Background thread: checks triggers at ~200 Hz."""
        while not self._stop_event.is_set():
            with self._lock:
                if self._playing and self._start_time is not None:
                    self._elapsed = time.monotonic() - self._start_time
                    self._check_triggers()
            self._stop_event.wait(_TICK_INTERVAL)

    def _check_triggers(self) -> None:
        """Fire any triggers whose time has been reached. Caller holds ``_lock``."""
        for trig in self._triggers:
            t = float(trig["time"])
            port = trig["port_name"]
            key = (port, t)
            if t <= self._elapsed and key not in self._fired_keys:
                self._fired_keys.add(key)
                self._fire_counts[port] = self._fire_counts.get(port, 0) + 1
                self._tick_count += 1
                self._pending_outputs[port] = self._fire_counts[port]
                self._pending_outputs["tick"] = self._tick_count

        # Broadcast elapsed at a throttled rate so downstream nodes see
        # a heartbeat without being flooded.
        now = time.monotonic()
        if now - self._last_elapsed_broadcast >= _ELAPSED_BROADCAST_INTERVAL:
            self._last_elapsed_broadcast = now
            self._pending_outputs["elapsed"] = round(self._elapsed, 3)
            self._pending_outputs["is_playing"] = self._playing

        # Handle loop / auto-stop
        if self._duration > 0 and self._elapsed >= self._duration:
            if self._loop:
                self._reset_state()
                self._start_time = time.monotonic()
            else:
                all_fired = all(
                    (t["port_name"], float(t["time"])) in self._fired_keys
                    for t in self._triggers
                )
                if all_fired:
                    self._playing = False
                    self._pending_outputs["is_playing"] = False

    def _reset_state(self) -> None:
        """Reset elapsed time and trigger state. Caller holds ``_lock``."""
        self._elapsed = 0.0
        self._fired_keys.clear()
        self._fire_counts.clear()
        self._tick_count = 0

    # ------------------------------------------------------------------
    # Node interface
    # ------------------------------------------------------------------

    def execute(self, inputs: dict[str, Any], **kwargs) -> dict[str, Any]:
        triggers = kwargs.get("triggers", self.config.get("triggers", []))
        loop = kwargs.get("loop", self.config.get("loop", False))
        duration = kwargs.get("duration", self.config.get("duration", 0.0))

        # Edge-detect start/reset counters so stale values on the input
        # queue don't retrigger actions.
        start_val = _as_counter(inputs.get("start"))
        reset_val = _as_counter(inputs.get("reset"))
        start_fired = (
            start_val is not None
            and self._prev_start_counter is not None
            and start_val > self._prev_start_counter
        )
        reset_fired = (
            reset_val is not None
            and self._prev_reset_counter is not None
            and reset_val > self._prev_reset_counter
        )
        if start_val is not None:
            self._prev_start_counter = start_val
        if reset_val is not None:
            self._prev_reset_counter = reset_val

        with self._lock:
            normalized_triggers: list[TriggerSpec] = []
            if isinstance(triggers, list):
                for t in triggers:
                    if not isinstance(t, dict):
                        continue
                    port = t.get("port_name")
                    time_val = t.get("time")
                    if isinstance(port, str) and port and time_val is not None:
                        normalized_triggers.append(
                            {"time": float(time_val), "port_name": port}
                        )
            self._triggers = normalized_triggers
            self._loop = bool(loop)
            self._duration = float(duration)

            if reset_fired:
                self._reset_state()
                self._start_time = time.monotonic() if self._playing else None

            if start_fired:
                self._playing = not self._playing
                if self._playing:
                    self._start_time = time.monotonic() - self._elapsed
                    self._start_timer()

            # Auto-start on first execute so wiring the node into a graph
            # drives downstream triggers without requiring an explicit start
            # pulse.
            if (
                not self._playing
                and self._start_time is None
                and self._triggers
                and self._prev_start_counter is None
                and self._prev_reset_counter is None
            ):
                self._playing = True
                self._start_time = time.monotonic()
                self._start_timer()

            if not self._pending_outputs:
                return {}

            outputs = self._pending_outputs
            self._pending_outputs = {}
            return outputs

    def shutdown(self) -> None:
        self._stop_timer()


def _as_counter(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
