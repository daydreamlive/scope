"""Tests for graph_schema.GraphConfig.validate_structure()."""
import pytest
from scope.server.graph_schema import GraphConfig, GraphNode, GraphEdge


def _make_graph(nodes: list[GraphNode], edges: list[GraphEdge] | None = None) -> GraphConfig:
    return GraphConfig(nodes=nodes, edges=edges or [])


def _pipeline_node(node_id: str = "p1", pipeline_id: str = "pipe-1") -> GraphNode:
    return GraphNode(id=node_id, type="pipeline", pipeline_id=pipeline_id)


def _sink_node(node_id: str = "sink") -> GraphNode:
    return GraphNode(id=node_id, type="sink")


def _source_node(node_id: str = "source") -> GraphNode:
    return GraphNode(id=node_id, type="source")


def _edge(from_node: str, to_node: str) -> GraphEdge:
    return GraphEdge(**{"from": from_node, "from_port": "video", "to_node": to_node, "to_port": "video", "kind": "stream"})


class TestValidateStructure:
    def test_valid_minimal_graph(self):
        graph = _make_graph(
            nodes=[_pipeline_node(), _sink_node()],
            edges=[_edge("p1", "sink")],
        )
        assert graph.validate_structure() == []

    def test_valid_with_source(self):
        graph = _make_graph(
            nodes=[_source_node(), _pipeline_node(), _sink_node()],
            edges=[_edge("source", "p1"), _edge("p1", "sink")],
        )
        assert graph.validate_structure() == []

    def test_no_sink_node_returns_error(self):
        graph = _make_graph(nodes=[_pipeline_node()])
        errors = graph.validate_structure()
        assert len(errors) == 1
        assert "sink node" in errors[0]

    def test_no_sink_error_message_is_user_friendly(self):
        """Error message should hint at how to fix, not just describe the problem."""
        graph = _make_graph(nodes=[_pipeline_node()])
        errors = graph.validate_structure()
        msg = errors[0]
        # Should mention the fix (add a Preview or Output node)
        assert "Preview" in msg or "Output" in msg or "output" in msg

    def test_duplicate_node_ids(self):
        graph = _make_graph(
            nodes=[_pipeline_node("p1"), _pipeline_node("p1"), _sink_node()],
        )
        errors = graph.validate_structure()
        assert any("Duplicate" in e for e in errors)

    def test_pipeline_missing_pipeline_id(self):
        graph = _make_graph(
            nodes=[GraphNode(id="p1", type="pipeline"), _sink_node()],
        )
        errors = graph.validate_structure()
        assert any("missing pipeline_id" in e for e in errors)

    def test_edge_references_nonexistent_source(self):
        graph = _make_graph(
            nodes=[_pipeline_node(), _sink_node()],
            edges=[_edge("nonexistent", "sink")],
        )
        errors = graph.validate_structure()
        assert any("nonexistent" in e for e in errors)

    def test_edge_references_nonexistent_target(self):
        graph = _make_graph(
            nodes=[_pipeline_node(), _sink_node()],
            edges=[_edge("p1", "does-not-exist")],
        )
        errors = graph.validate_structure()
        assert any("does-not-exist" in e for e in errors)

    def test_sink_with_sink_mode_counts_as_sink(self):
        """An output node (Spout/NDI/Syphon) with sink_mode set should count as a valid sink."""
        output_node = GraphNode(id="spout-out", type="sink", sink_mode="spout")
        graph = _make_graph(
            nodes=[_pipeline_node(), output_node],
            edges=[_edge("p1", "spout-out")],
        )
        assert graph.validate_structure() == []

    def test_multiple_errors_returned(self):
        """Multiple structural errors should all be reported."""
        graph = _make_graph(
            nodes=[GraphNode(id="p1", type="pipeline")],  # no pipeline_id, no sink
        )
        errors = graph.validate_structure()
        assert len(errors) >= 2
