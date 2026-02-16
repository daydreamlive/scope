"""DAG (Directed Acyclic Graph) schema for pipeline execution.

Defines a JSON-friendly format to describe:
- Nodes: source (input), pipeline instances, sink (output)
- Edges: connections between nodes via named ports

All frame ports (video, vace_input_frames, vace_input_masks) use stream edges
(frame-by-frame queues). The optional "parameter" kind is for future event-like
data only.

Example (YOLO plugin + Longlive with shared input video):

    {
      "nodes": [
        {"id": "input", "type": "source"},
        {"id": "yolo_plugin", "type": "pipeline", "pipeline_id": "yolo_plugin"},
        {"id": "longlive", "type": "pipeline", "pipeline_id": "longlive"},
        {"id": "output", "type": "sink"}
      ],
      "edges": [
        {"from": "input", "from_port": "video", "to_node": "yolo_plugin", "to_port": "video", "kind": "stream"},
        {"from": "input", "from_port": "video", "to_node": "longlive", "to_port": "video", "kind": "stream"},
        {"from": "yolo_plugin", "from_port": "vace_input_frames", "to_node": "longlive", "to_port": "vace_input_frames", "kind": "stream"},
        {"from": "yolo_plugin", "from_port": "vace_input_masks", "to_node": "longlive", "to_port": "vace_input_masks", "kind": "stream"},
        {"from": "longlive", "from_port": "video", "to_node": "output", "to_port": "video", "kind": "stream"}
      ]
    }
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class DagNode(BaseModel):
    """A node in the pipeline DAG."""

    id: str = Field(
        ...,
        description="Unique node id (e.g. 'input', 'yolo_plugin', 'longlive', 'output')",
    )
    type: Literal["source", "pipeline", "sink"] = Field(
        ...,
        description="source = external input, pipeline = pipeline instance, sink = output",
    )
    pipeline_id: str | None = Field(
        default=None,
        description="Pipeline ID (registry key) when type is 'pipeline'",
    )


class DagEdge(BaseModel):
    """An edge connecting an output port to an input port."""

    from_node: str = Field(..., alias="from", description="Source node id")
    from_port: str = Field(
        ..., description="Source port (e.g. 'video', 'vace_input_frames')"
    )
    to_node: str = Field(..., description="Target node id")
    to_port: str = Field(..., description="Target port name")
    kind: Literal["stream", "parameter"] = Field(
        default="stream",
        description="stream = queue (frame-by-frame), parameter = chunk-level pass-through",
    )

    model_config = {"populate_by_name": True}


class DagConfig(BaseModel):
    """Root DAG configuration (graph definition)."""

    nodes: list[DagNode] = Field(..., description="DAG nodes")
    edges: list[DagEdge] = Field(..., description="Connections between nodes")

    def get_pipeline_node_ids(self) -> list[str]:
        """Return node ids that are pipeline nodes, in definition order."""
        return [n.id for n in self.nodes if n.type == "pipeline"]

    def get_source_node_ids(self) -> list[str]:
        """Return node ids that are source nodes."""
        return [n.id for n in self.nodes if n.type == "source"]

    def get_sink_node_ids(self) -> list[str]:
        """Return node ids that are sink nodes."""
        return [n.id for n in self.nodes if n.type == "sink"]

    def edges_from(self, node_id: str) -> list[DagEdge]:
        """Return edges whose source is the given node."""
        return [e for e in self.edges if e.from_node == node_id]

    def edges_to(self, node_id: str) -> list[DagEdge]:
        """Return edges whose target is the given node."""
        return [e for e in self.edges if e.to_node == node_id]

    def stream_edges_from(self, node_id: str) -> list[DagEdge]:
        """Return stream edges whose source is the given node."""
        return [e for e in self.edges_from(node_id) if e.kind == "stream"]

    def parameter_edges_from(self, node_id: str) -> list[DagEdge]:
        """Return parameter edges whose source is the given node."""
        return [e for e in self.edges_from(node_id) if e.kind == "parameter"]

    def node_by_id(self, node_id: str) -> DagNode | None:
        """Return the node with the given id."""
        for n in self.nodes:
            if n.id == node_id:
                return n
        return None
