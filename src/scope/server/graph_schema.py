"""Graph (Directed Acyclic Graph) schema for pipeline execution.

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

import logging
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)


class GraphNode(BaseModel):
    """A node in the pipeline graph."""

    id: str = Field(
        ...,
        description="Unique node id (e.g. 'input', 'yolo_plugin', 'longlive', 'output')",
    )
    type: Literal["source", "pipeline", "sink", "record"] = Field(
        ...,
        description="source = external input, pipeline = pipeline instance, sink = output, record = file recorder",
    )
    pipeline_id: str | None = Field(
        default=None,
        description="Pipeline ID (registry key) when type is 'pipeline'",
    )
    source_mode: str | None = Field(
        default=None,
        description="Video source mode for source nodes: 'video', 'camera', 'spout', 'ndi', 'syphon'",
    )
    source_name: str | None = Field(
        default=None,
        description="Source name/identifier for Spout/NDI/Syphon sources (sender name for Spout, source identifier for NDI/Syphon)",
    )
    tempo_sync: bool = Field(
        default=False,
        description="When true, this pipeline receives beat state injection, modulation, and beat cache resets.",
    )
    sink_mode: str | None = Field(
        default=None,
        description="Output sink mode for sink nodes: 'spout', 'ndi', 'syphon'. When set, frames are sent to the specified output sink instead of (or in addition to) WebRTC.",
    )
    sink_name: str | None = Field(
        default=None,
        description="Sink name/identifier for Spout/NDI/Syphon output sinks (sender name for Spout, source identifier for NDI/Syphon)",
    )


class GraphEdge(BaseModel):
    """An edge connecting an output port to an input port.

    Accepts both the current schema (``from``, ``from_port``, ``to_node``,
    ``to_port``) and the legacy schema (``source``, ``target``) for backwards
    compatibility with older Scope desktop clients.  When the legacy keys are
    present the port names default to ``"video"``.
    """

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

    @model_validator(mode="before")
    @classmethod
    def _coerce_legacy_edge(cls, data: Any) -> Any:
        """Map legacy ``source``/``target`` keys to the current schema.

        Older clients send edges as::

            {"source": "input", "target": "pipeline"}

        The current schema requires ``from``, ``from_port``, ``to_node``,
        ``to_port``.  This validator accepts any mix of legacy and current keys,
        mapping them where the canonical field is absent.  Port names default to
        ``"video"`` when the legacy payload omits port information.
        """
        if not isinstance(data, dict):
            return data

        has_legacy = "source" in data or "target" in data
        if not has_legacy:
            return data

        logger.warning(
            "GraphEdge: received legacy edge schema (source/target). "
            "Please update the Scope client to send 'from'/'to_node' edges. "
            "Coercing automatically. source=%r target=%r",
            data.get("source"),
            data.get("target"),
        )

        data = dict(data)  # make a mutable copy

        # Map source → from (only when 'from' is absent)
        if "source" in data and "from" not in data:
            data["from"] = data.pop("source")
        else:
            data.pop("source", None)

        # Map target → to_node (only when 'to_node' is absent)
        if "target" in data and "to_node" not in data:
            data["to_node"] = data.pop("target")
        else:
            data.pop("target", None)

        # Apply port defaults when the caller omitted them
        data.setdefault("from_port", data.pop("source_port", "video"))
        data.setdefault("to_port", data.pop("target_port", "video"))

        return data


class GraphConfig(BaseModel):
    """Root graph configuration (graph definition)."""

    nodes: list[GraphNode] = Field(..., description="Graph nodes")
    edges: list[GraphEdge] = Field(..., description="Connections between nodes")

    def get_pipeline_node_ids(self) -> list[str]:
        """Return node ids that are pipeline nodes, in definition order."""
        return [n.id for n in self.nodes if n.type == "pipeline"]

    def get_source_node_ids(self) -> list[str]:
        """Return node ids that are source nodes."""
        return [n.id for n in self.nodes if n.type == "source"]

    def get_sink_node_ids(self) -> list[str]:
        """Return node ids that are sink nodes."""
        return [n.id for n in self.nodes if n.type == "sink"]

    def get_record_node_ids(self) -> list[str]:
        """Return node ids that are record nodes."""
        return [n.id for n in self.nodes if n.type == "record"]

    def edges_from(self, node_id: str) -> list[GraphEdge]:
        """Return edges whose source is the given node."""
        return [e for e in self.edges if e.from_node == node_id]

    def edges_to(self, node_id: str) -> list[GraphEdge]:
        """Return edges whose target is the given node."""
        return [e for e in self.edges if e.to_node == node_id]

    def stream_edges_from(self, node_id: str) -> list[GraphEdge]:
        """Return stream edges whose source is the given node."""
        return [e for e in self.edges_from(node_id) if e.kind == "stream"]

    def validate_structure(self) -> list[str]:
        """Validate the graph structure and return a list of error messages.

        Checks:
        - No duplicate node IDs
        - At least one sink node (source nodes are optional)
        - Pipeline nodes have a pipeline_id
        - All edge references point to existing nodes
        """
        errors: list[str] = []
        node_ids = [n.id for n in self.nodes]

        # Check for duplicate node IDs
        seen: set[str] = set()
        for nid in node_ids:
            if nid in seen:
                errors.append(f"Duplicate node ID: '{nid}'")
            seen.add(nid)

        # At least one sink
        if not self.get_sink_node_ids():
            errors.append("Graph must have at least one sink node")

        # Pipeline nodes must have pipeline_id
        for node in self.nodes:
            if node.type == "pipeline" and not node.pipeline_id:
                errors.append(f"Pipeline node '{node.id}' is missing pipeline_id")

        # Edge references must point to existing nodes
        node_id_set = set(node_ids)
        for edge in self.edges:
            if edge.from_node not in node_id_set:
                errors.append(
                    f"Edge references non-existent source node: '{edge.from_node}'"
                )
            if edge.to_node not in node_id_set:
                errors.append(
                    f"Edge references non-existent target node: '{edge.to_node}'"
                )

        return errors


def build_linear_graph(
    pipeline_ids: list[str],
    vace_input_video_ids: set[str] | None = None,
) -> GraphConfig:
    """Build a linear GraphConfig: source → pipeline_a → pipeline_b → … → sink.

    Args:
        pipeline_ids: Ordered list of pipeline IDs to chain.
        vace_input_video_ids: Pipeline IDs that should receive input video as
            ``vace_input_frames`` instead of ``video``.  When VACE is enabled
            with ``vace_use_input_video``, callers resolve which pipelines
            support VACE and pass their IDs here.
    """
    nodes = [GraphNode(id="input", type="source")]
    edges: list[GraphEdge] = []
    _vace_ids = vace_input_video_ids or set()

    prev_node_id = "input"
    for pid in pipeline_ids:
        nodes.append(GraphNode(id=pid, type="pipeline", pipeline_id=pid))
        to_port = "vace_input_frames" if pid in _vace_ids else "video"
        edges.append(
            GraphEdge(
                **{
                    "from": prev_node_id,
                    "from_port": "video",
                    "to_node": pid,
                    "to_port": to_port,
                    "kind": "stream",
                }
            )
        )
        prev_node_id = pid

    nodes.append(GraphNode(id="output", type="sink"))
    edges.append(
        GraphEdge(
            **{
                "from": prev_node_id,
                "from_port": "video",
                "to_node": "output",
                "to_port": "video",
                "kind": "stream",
            }
        )
    )

    return GraphConfig(nodes=nodes, edges=edges)
