"""Trigger node — emits a 1-tick pulse on fire event."""

from __future__ import annotations

from typing import Any, ClassVar

from scope.server.tempo_sync import BeatState

from ..base import BaseNode, NodeDefinition, NodePort


class TriggerNode(BaseNode):
    """Emits 1 on fire event tick, 0 otherwise."""

    node_type_id: ClassVar[str] = "trigger"

    @classmethod
    def get_definition(cls) -> NodeDefinition:
        return NodeDefinition(
            node_type_id=cls.node_type_id,
            display_name="Trigger",
            category="input",
            description="Momentary trigger that emits a single-tick pulse.",
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
        fired = self._state.pop("fired", False)
        return {"value": 1.0 if fired else 0.0}

    def on_event(self, event_type: str, payload: dict[str, Any]) -> None:
        if event_type == "fire":
            self._state["fired"] = True
