"""XY Pad node — 2D coordinate output with configurable ranges."""

from __future__ import annotations

from typing import Any, ClassVar

from scope.server.tempo_sync import BeatState

from ..base import BaseNode, NodeDefinition, NodePort


class XYPadNode(BaseNode):
    """Produces X and Y coordinate values from a 2D pad."""

    node_type_id: ClassVar[str] = "xypad"

    @classmethod
    def get_definition(cls) -> NodeDefinition:
        return NodeDefinition(
            node_type_id=cls.node_type_id,
            display_name="XY Pad",
            category="input",
            description="2D coordinate pad with independent X/Y ranges.",
            inputs=[],
            outputs=[
                NodePort(name="x", port_type="number"),
                NodePort(name="y", port_type="number"),
            ],
        )

    def execute(
        self,
        inputs: dict[str, Any],
        tick_time: float,
        beat_state: BeatState | None,
    ) -> dict[str, Any]:
        min_x = float(self.config.get("padMinX", 0))
        max_x = float(self.config.get("padMaxX", 1))
        min_y = float(self.config.get("padMinY", 0))
        max_y = float(self.config.get("padMaxY", 1))

        x = max(min_x, min(max_x, float(self.config.get("padX", 0.5))))
        y = max(min_y, min(max_y, float(self.config.get("padY", 0.5))))
        return {"x": x, "y": y}

    def on_event(self, event_type: str, payload: dict[str, Any]) -> None:
        if event_type == "pad_change":
            if "x" in payload:
                self.config["padX"] = payload["x"]
            if "y" in payload:
                self.config["padY"] = payload["y"]
