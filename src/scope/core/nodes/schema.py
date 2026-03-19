"""Extended graph schema for the backend node engine.

Defines ``NodeInstance``, ``EdgeInstance``, and ``GraphDefinition`` — the
wire format sent from the frontend when the user builds / updates a graph
in graph mode.  This extends the existing ``graph_schema.py`` (which is
kept for backward compatibility with the pipeline-only DAG).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class NodeInstance(BaseModel):
    """A single node in the graph (any type: math, control, pipeline, etc.)."""

    id: str
    type: str  # node_type_id from NodeRegistry
    config: dict[str, Any] = Field(default_factory=dict)
    position: dict[str, Any] = Field(default_factory=dict)  # x, y for frontend
    pipeline_id: str | None = None  # For pipeline bridge nodes
    inner_nodes: list[NodeInstance] | None = None  # For subgraph nodes
    inner_edges: list[EdgeInstance] | None = None

    model_config = {"populate_by_name": True}


class EdgeInstance(BaseModel):
    """A connection between two node ports."""

    id: str
    source: str  # Source node ID
    source_port: str  # Output port name
    target: str  # Target node ID
    target_port: str  # Input port name
    kind: Literal["param", "stream"] = "param"


class GraphDefinition(BaseModel):
    """Root graph definition submitted by the frontend.

    Contains *all* node types (data nodes + pipeline/source/sink nodes).
    The ``GraphEngine`` extracts the pipeline subset into a ``GraphConfig``
    for the existing ``build_graph()`` pipeline executor.
    """

    nodes: list[NodeInstance] = Field(default_factory=list)
    edges: list[EdgeInstance] = Field(default_factory=list)
