"""Slider node — holds a numeric value clamped to a configurable range."""

from __future__ import annotations

from typing import Any, ClassVar

from scope.server.tempo_sync import BeatState

from ..base import BaseNode, NodeDefinition, NodePort


class SliderNode(BaseNode):
    """Outputs a clamped numeric value. Updated via ``on_event``."""

    node_type_id: ClassVar[str] = "slider"

    @classmethod
    def get_definition(cls) -> NodeDefinition:
        return NodeDefinition(
            node_type_id=cls.node_type_id,
            display_name="Slider",
            category="input",
            description="Numeric slider with configurable range.",
            inputs=[
                NodePort(name="value", port_type="number"),
            ],
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
        min_val = float(self.config.get("sliderMin", 0))
        max_val = float(self.config.get("sliderMax", 1))
        current = self.config.get("value", (min_val + max_val) / 2.0)

        # Allow input connection to override
        if "value" in inputs and inputs["value"] is not None:
            try:
                current = float(inputs["value"])
            except (TypeError, ValueError):
                pass

        value = max(min_val, min(max_val, float(current)))
        return {"value": value}

    def on_event(self, event_type: str, payload: dict[str, Any]) -> None:
        if event_type == "value_change" and "value" in payload:
            self.config["value"] = payload["value"]
