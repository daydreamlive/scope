"""MIDI node — receives MIDI CC/Note values from the backend.

This replaces the frontend Web MIDI API usage.  The actual ``mido``/``rtmidi``
integration is left as a Phase 5 task; for now the node reads values injected
via ``on_event`` (which the MIDI listener will call).
"""

from __future__ import annotations

from typing import Any, ClassVar

from scope.server.tempo_sync import BeatState

from ..base import BaseNode, NodeDefinition, NodePort


class MidiNode(BaseNode):
    """Outputs MIDI CC channel values. Updated via ``on_event`` from MIDI listener."""

    node_type_id: ClassVar[str] = "midi"

    @classmethod
    def get_definition(cls) -> NodeDefinition:
        return NodeDefinition(
            node_type_id=cls.node_type_id,
            display_name="MIDI",
            category="input",
            description="MIDI CC/Note values from backend MIDI input.",
            inputs=[],
            outputs=[
                NodePort(name="value", port_type="number"),
            ],
        )

    def execute(
        self,
        inputs: dict[str, Any],
        tick_time: float,
        beat_state: BeatState | None,
    ) -> dict[str, Any]:
        channels: dict[str, float] = self._state.get("channels", {})
        result: dict[str, Any] = {}
        for key, value in channels.items():
            result[key] = value
        return result

    def on_event(self, event_type: str, payload: dict[str, Any]) -> None:
        if event_type == "midi_cc":
            channel = payload.get("channel")
            value = payload.get("value", 0)
            if channel is not None:
                channels = self._state.setdefault("channels", {})
                channels[f"midi_{channel}"] = float(value)
        elif event_type == "midi_note":
            note = payload.get("note")
            velocity = payload.get("velocity", 0)
            if note is not None:
                channels = self._state.setdefault("channels", {})
                channels[f"note_{note}"] = float(velocity)
