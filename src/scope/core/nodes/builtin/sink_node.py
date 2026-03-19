"""Sink node — represents an output destination in the graph."""

from __future__ import annotations

from typing import Any, ClassVar

from scope.server.tempo_sync import BeatState

from ..base import BaseNode, NodeDefinition, NodePort


class SinkNode(BaseNode):
    """Output destination node (display, recording target)."""

    node_type_id: ClassVar[str] = "sink"

    @classmethod
    def get_definition(cls) -> NodeDefinition:
        return NodeDefinition(
            node_type_id=cls.node_type_id,
            display_name="Sink",
            category="output",
            description="Video output destination.",
            inputs=[
                NodePort(name="video", port_type="stream"),
            ],
            outputs=[],
            is_stream_node=True,
        )

    def execute(
        self,
        inputs: dict[str, Any],
        tick_time: float,
        beat_state: BeatState | None,
    ) -> dict[str, Any]:
        # Stream nodes are handled by the existing GraphRun/FrameProcessor.
        return {}
