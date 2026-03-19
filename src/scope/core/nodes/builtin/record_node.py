"""Record node — recording trigger with timer."""

from __future__ import annotations

import time
from typing import Any, ClassVar

from scope.server.tempo_sync import BeatState

from ..base import BaseNode, NodeDefinition, NodePort


class RecordNode(BaseNode):
    """Controls recording start/stop with an elapsed timer output."""

    node_type_id: ClassVar[str] = "record"

    @classmethod
    def get_definition(cls) -> NodeDefinition:
        return NodeDefinition(
            node_type_id=cls.node_type_id,
            display_name="Record",
            category="output",
            description="Recording trigger with elapsed-time output.",
            inputs=[
                NodePort(name="trigger", port_type="number"),
            ],
            outputs=[
                NodePort(name="recording", port_type="number"),
                NodePort(name="elapsed", port_type="number"),
            ],
            is_animated=True,
        )

    def execute(
        self,
        inputs: dict[str, Any],
        tick_time: float,
        beat_state: BeatState | None,
    ) -> dict[str, Any]:
        # Rising-edge detection on trigger input
        trigger_val = _to_float(inputs.get("trigger"))
        prev_trigger = self._state.get("prev_trigger", 0.0)
        recording = self._state.get("recording", False)

        if trigger_val is not None:
            if trigger_val > 0.5 and prev_trigger <= 0.5:
                recording = not recording
                if recording:
                    self._state["record_start"] = time.monotonic()
            self._state["prev_trigger"] = trigger_val

        self._state["recording"] = recording

        elapsed = 0.0
        if recording:
            start = self._state.get("record_start", time.monotonic())
            elapsed = time.monotonic() - start

        return {
            "recording": 1.0 if recording else 0.0,
            "elapsed": elapsed,
        }

    def on_event(self, event_type: str, payload: dict[str, Any]) -> None:
        if event_type == "toggle_recording":
            recording = self._state.get("recording", False)
            recording = not recording
            self._state["recording"] = recording
            if recording:
                self._state["record_start"] = time.monotonic()


def _to_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
