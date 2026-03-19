"""Subgraph boundary nodes — input/output ports for subgraph encapsulation."""

from __future__ import annotations

from typing import Any, ClassVar

from scope.server.tempo_sync import BeatState

from ..base import BaseNode, NodeDefinition, NodePort


class SubgraphInputNode(BaseNode):
    """Boundary input node inside a subgraph. Receives values from parent."""

    node_type_id: ClassVar[str] = "subgraph_input"

    @classmethod
    def get_definition(cls) -> NodeDefinition:
        return NodeDefinition(
            node_type_id=cls.node_type_id,
            display_name="Subgraph Input",
            category="utility",
            description="Boundary input inside a subgraph.",
            inputs=[],
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
        # Value is injected by the engine from the parent graph's edge
        return {"value": self._state.get("value")}

    def set_boundary_value(self, port_name: str, value: Any) -> None:
        """Called by the engine to seed this boundary input."""
        self._state["value"] = value


class SubgraphOutputNode(BaseNode):
    """Boundary output node inside a subgraph. Exports values to parent."""

    node_type_id: ClassVar[str] = "subgraph_output"

    @classmethod
    def get_definition(cls) -> NodeDefinition:
        return NodeDefinition(
            node_type_id=cls.node_type_id,
            display_name="Subgraph Output",
            category="utility",
            description="Boundary output inside a subgraph.",
            inputs=[
                NodePort(name="value", port_type="any"),
            ],
            outputs=[],
        )

    def execute(
        self,
        inputs: dict[str, Any],
        tick_time: float,
        beat_state: BeatState | None,
    ) -> dict[str, Any]:
        return {"value": inputs.get("value")}
