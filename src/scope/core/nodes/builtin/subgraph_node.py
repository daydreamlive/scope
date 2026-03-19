"""Subgraph node — recursively evaluates an inner graph.

The engine handles subgraph evaluation by instantiating a nested set of nodes
from ``inner_nodes`` / ``inner_edges`` and evaluating them in topological order
within a single tick.
"""

from __future__ import annotations

from typing import Any, ClassVar

from scope.server.tempo_sync import BeatState

from ..base import BaseNode, NodeDefinition


class SubgraphNode(BaseNode):
    """Encapsulates an inner graph with boundary input/output ports."""

    node_type_id: ClassVar[str] = "subgraph"

    @classmethod
    def get_definition(cls) -> NodeDefinition:
        return NodeDefinition(
            node_type_id=cls.node_type_id,
            display_name="Subgraph",
            category="utility",
            description="Encapsulates an inner graph with named boundary ports.",
            inputs=[],  # Dynamic — determined by inner subgraph inputs
            outputs=[],  # Dynamic — determined by inner subgraph outputs
        )

    def execute(
        self,
        inputs: dict[str, Any],
        tick_time: float,
        beat_state: BeatState | None,
    ) -> dict[str, Any]:
        # Actual recursive evaluation is performed by the GraphEngine.
        # The engine calls _evaluate_subgraph() which populates the result.
        # This execute() returns whatever the engine stored after inner eval.
        return dict(self._state.get("outputs", {}))

    def set_inner_outputs(self, outputs: dict[str, Any]) -> None:
        """Called by the engine after evaluating the inner graph."""
        self._state["outputs"] = outputs
