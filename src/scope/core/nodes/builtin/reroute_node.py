"""Reroute node — transparent wire pass-through."""

from __future__ import annotations

from typing import Any, ClassVar

from scope.server.tempo_sync import BeatState

from ..base import BaseNode, NodeDefinition, NodePort


class RerouteNode(BaseNode):
    """Pass-through node for clean wire routing."""

    node_type_id: ClassVar[str] = "reroute"

    @classmethod
    def get_definition(cls) -> NodeDefinition:
        return NodeDefinition(
            node_type_id=cls.node_type_id,
            display_name="Reroute",
            category="utility",
            description="Transparent wire pass-through for cleaner layouts.",
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
        return {"value": inputs.get("value", self.config.get("value"))}
