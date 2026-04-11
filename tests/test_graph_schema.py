"""Tests for graph_schema backwards-compatibility (issue #895).

Verifies that GraphEdge and GraphConfig accept both the legacy
``source``/``target`` edge format and the current ``from``/``to_node`` format.
"""

from __future__ import annotations

import logging

import pytest

from scope.server.graph_schema import GraphConfig, GraphEdge, GraphNode


# ---------------------------------------------------------------------------
# GraphEdge unit tests
# ---------------------------------------------------------------------------


class TestGraphEdgeLegacyKeys:
    """GraphEdge should accept the old source/target format."""

    def test_legacy_source_target_minimal(self):
        """Basic source/target without ports → defaults applied."""
        edge = GraphEdge.model_validate({"source": "input", "target": "pipeline"})
        assert edge.from_node == "input"
        assert edge.to_node == "pipeline"
        assert edge.from_port == "video"  # default
        assert edge.to_port == "video"  # default
        assert edge.kind == "stream"  # default

    def test_legacy_source_target_with_ports(self):
        """Legacy source/target alongside explicit port names."""
        edge = GraphEdge.model_validate(
            {
                "source": "input",
                "target": "pipeline",
                "from_port": "video",
                "to_port": "video",
            }
        )
        assert edge.from_node == "input"
        assert edge.to_node == "pipeline"
        assert edge.from_port == "video"
        assert edge.to_port == "video"

    def test_legacy_emits_deprecation_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger="scope.server.graph_schema"):
            GraphEdge.model_validate({"source": "a", "target": "b"})
        assert any("legacy edge schema" in r.message for r in caplog.records)

    def test_legacy_only_source(self):
        """Only 'source' provided (no 'target') — should still parse."""
        edge = GraphEdge.model_validate(
            {"source": "a", "to_node": "b", "from_port": "video", "to_port": "video"}
        )
        assert edge.from_node == "a"
        assert edge.to_node == "b"

    def test_legacy_only_target(self):
        """Only 'target' provided (no 'source') — should still parse."""
        edge = GraphEdge.model_validate(
            {"from": "a", "target": "b", "from_port": "video", "to_port": "video"}
        )
        assert edge.from_node == "a"
        assert edge.to_node == "b"


class TestGraphEdgeCurrentKeys:
    """Existing schema (from/from_port/to_node/to_port) must still work."""

    def test_current_format(self):
        edge = GraphEdge.model_validate(
            {
                "from": "input",
                "from_port": "video",
                "to_node": "pipeline",
                "to_port": "video",
                "kind": "stream",
            }
        )
        assert edge.from_node == "input"
        assert edge.from_port == "video"
        assert edge.to_node == "pipeline"
        assert edge.to_port == "video"
        assert edge.kind == "stream"

    def test_current_format_no_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger="scope.server.graph_schema"):
            GraphEdge.model_validate(
                {
                    "from": "input",
                    "from_port": "video",
                    "to_node": "pipeline",
                    "to_port": "video",
                }
            )
        assert not any("Deprecated" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# GraphConfig integration test
# ---------------------------------------------------------------------------


class TestGraphConfigLegacyEdges:
    """GraphConfig should parse correctly even when edges use legacy keys."""

    def _make_config(self, edges):
        return GraphConfig.model_validate(
            {
                "nodes": [
                    {"id": "input", "type": "source"},
                    {"id": "pipeline", "type": "pipeline", "pipeline_id": "my_pipe"},
                    {"id": "output", "type": "sink"},
                ],
                "edges": edges,
            }
        )

    def test_legacy_edges_in_graph_config(self):
        cfg = self._make_config(
            [
                {"source": "input", "target": "pipeline"},
                {"source": "pipeline", "target": "output"},
            ]
        )
        assert len(cfg.edges) == 2
        assert cfg.edges[0].from_node == "input"
        assert cfg.edges[0].to_node == "pipeline"
        assert cfg.edges[1].from_node == "pipeline"
        assert cfg.edges[1].to_node == "output"

    def test_mixed_edges_in_graph_config(self):
        """Mix of legacy and current edge formats in the same config."""
        cfg = self._make_config(
            [
                {"source": "input", "target": "pipeline"},
                {
                    "from": "pipeline",
                    "from_port": "video",
                    "to_node": "output",
                    "to_port": "video",
                },
            ]
        )
        assert cfg.edges[0].from_node == "input"
        assert cfg.edges[1].from_node == "pipeline"

    def test_validate_structure_passes(self):
        cfg = self._make_config(
            [
                {"source": "input", "target": "pipeline"},
                {"source": "pipeline", "target": "output"},
            ]
        )
        errors = cfg.validate_structure()
        assert errors == []
