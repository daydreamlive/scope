"""Knobs node — multi-output knob bank with configurable ranges."""

from __future__ import annotations

from typing import Any, ClassVar

from scope.server.tempo_sync import BeatState

from ..base import BaseNode, NodeDefinition, NodePort


class KnobsNode(BaseNode):
    """Multiple knobs, each producing an independent numeric output."""

    node_type_id: ClassVar[str] = "knobs"

    @classmethod
    def get_definition(cls) -> NodeDefinition:
        return NodeDefinition(
            node_type_id=cls.node_type_id,
            display_name="Knobs",
            category="input",
            description="Bank of rotary knobs, each with independent range.",
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
        knobs: list[dict[str, Any]] = self.config.get("knobs", [])
        result: dict[str, Any] = {}
        for i, knob in enumerate(knobs):
            value = float(knob.get("value", 0))
            min_val = float(knob.get("min", 0))
            max_val = float(knob.get("max", 1))
            value = max(min_val, min(max_val, value))
            result[f"knob_{i}"] = value
        return result

    def on_event(self, event_type: str, payload: dict[str, Any]) -> None:
        if event_type == "knob_change":
            idx = payload.get("index")
            value = payload.get("value")
            knobs: list[dict[str, Any]] = self.config.get("knobs", [])
            if idx is not None and 0 <= idx < len(knobs):
                knobs[idx]["value"] = value
