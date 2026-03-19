"""Pipeline bridge node — connects data nodes to pipeline parameter updates.

This node sits at the boundary between the data-node graph and the existing
``PipelineProcessor`` system.  When upstream data nodes change values, the
engine calls this node which translates them into ``update_parameters()``
calls on the target pipeline processor.
"""

from __future__ import annotations

from typing import Any, ClassVar

from scope.server.tempo_sync import BeatState

from ..base import BaseNode, NodeDefinition


class PipelineBridgeNode(BaseNode):
    """Bridges data-node outputs → PipelineProcessor.update_parameters()."""

    node_type_id: ClassVar[str] = "pipeline"

    @classmethod
    def get_definition(cls) -> NodeDefinition:
        return NodeDefinition(
            node_type_id=cls.node_type_id,
            display_name="Pipeline",
            category="pipeline",
            description="Bridges data-node values to a pipeline's parameters.",
            inputs=[],  # Dynamic — determined by pipeline config schema
            outputs=[],  # Dynamic — stream outputs
            is_stream_node=True,
        )

    def execute(
        self,
        inputs: dict[str, Any],
        tick_time: float,
        beat_state: BeatState | None,
    ) -> dict[str, Any]:
        # The engine handles parameter forwarding to the pipeline processor.
        # This execute() collects inputs for the engine to forward.
        params: dict[str, Any] = {}
        for port_name, value in inputs.items():
            if value is not None:
                # Handle special prompt port
                if port_name == "__prompt":
                    params["prompts"] = [{"text": str(value), "weight": 100}]
                elif port_name == "__vace":
                    params["vace_input_frames"] = value
                else:
                    params[port_name] = value
        self._state["pending_params"] = params
        return {}

    def get_pending_params(self) -> dict[str, Any]:
        """Retrieve and clear pending parameter updates for the pipeline."""
        return self._state.pop("pending_params", {})
