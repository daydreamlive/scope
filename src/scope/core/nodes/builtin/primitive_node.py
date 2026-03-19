"""Primitive node — holds a typed constant value (string, number, boolean)."""

from __future__ import annotations

from typing import Any, ClassVar

from scope.server.tempo_sync import BeatState

from ..base import BaseNode, NodeDefinition, NodePort


class PrimitiveNode(BaseNode):
    """Pass-through value holder with type awareness."""

    node_type_id: ClassVar[str] = "primitive"

    @classmethod
    def get_definition(cls) -> NodeDefinition:
        return NodeDefinition(
            node_type_id=cls.node_type_id,
            display_name="Primitive",
            category="utility",
            description="Holds a constant value (string, number, or boolean).",
            inputs=[
                NodePort(name="value", port_type="any"),
            ],
            outputs=[
                NodePort(name="value", port_type="any"),
            ],
        )

    def execute(
        self,
        inputs: dict[str, Any],
        tick_time: float,
        beat_state: BeatState | None,
    ) -> dict[str, Any]:
        # Input connection overrides config value
        if "value" in inputs and inputs["value"] is not None:
            return {"value": inputs["value"]}
        return {"value": self.config.get("value")}

    def on_event(self, event_type: str, payload: dict[str, Any]) -> None:
        if event_type == "value_change" and "value" in payload:
            self.config["value"] = payload["value"]
