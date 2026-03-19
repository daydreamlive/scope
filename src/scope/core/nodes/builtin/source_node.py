"""Source node — represents an input source in the graph."""

from __future__ import annotations

from typing import Any, ClassVar

from scope.server.tempo_sync import BeatState

from ..base import BaseNode, NodeDefinition, NodePort


class SourceNode(BaseNode):
    """Input source configuration node (camera, video file, Spout, NDI, etc.)."""

    node_type_id: ClassVar[str] = "source"

    @classmethod
    def get_definition(cls) -> NodeDefinition:
        return NodeDefinition(
            node_type_id=cls.node_type_id,
            display_name="Source",
            category="input",
            description="Video input source (camera, file, Spout, NDI, Syphon).",
            inputs=[],
            outputs=[
                NodePort(name="video", port_type="stream"),
            ],
            is_stream_node=True,
        )

    def execute(
        self,
        inputs: dict[str, Any],
        tick_time: float,
        beat_state: BeatState | None,
    ) -> dict[str, Any]:
        # Stream nodes are handled by the existing GraphRun/FrameProcessor.
        # This node only holds source configuration.
        return {}
