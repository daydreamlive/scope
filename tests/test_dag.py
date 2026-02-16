"""Tests for DAG pipeline execution (dag_schema, dag_executor, linear DAG)."""

import queue
from unittest.mock import MagicMock

import pytest

from scope.server.dag_executor import build_dag, linear_dag_from_pipeline_ids
from scope.server.dag_schema import DagConfig


class TestLinearDagFromPipelineIds:
    """Tests for linear_dag_from_pipeline_ids."""

    def test_single_pipeline(self):
        dag = linear_dag_from_pipeline_ids(["longlive"])
        assert [n.id for n in dag.nodes] == ["input", "longlive", "output"]
        edges = [(e.from_node, e.to_node, e.from_port, e.kind) for e in dag.edges]
        assert ("input", "longlive", "video", "stream") in edges
        assert ("longlive", "output", "video", "stream") in edges
        assert all(e.kind == "stream" for e in dag.edges)

    def test_two_pipelines_includes_vace_stream_edges(self):
        dag = linear_dag_from_pipeline_ids(["passthrough", "longlive"])
        assert len(dag.get_pipeline_node_ids()) == 2
        vace_edges = [
            e
            for e in dag.edges
            if e.from_port in ("vace_input_frames", "vace_input_masks")
        ]
        assert len(vace_edges) == 2
        assert all(e.kind == "stream" for e in vace_edges)

    def test_explicit_dag_config_roundtrip(self):
        raw = {
            "nodes": [
                {"id": "input", "type": "source"},
                {"id": "p1", "type": "pipeline", "pipeline_id": "passthrough"},
                {"id": "output", "type": "sink"},
            ],
            "edges": [
                {
                    "from": "input",
                    "from_port": "video",
                    "to_node": "p1",
                    "to_port": "video",
                    "kind": "stream",
                },
                {
                    "from": "p1",
                    "from_port": "video",
                    "to_node": "output",
                    "to_port": "video",
                    "kind": "stream",
                },
            ],
        }
        config = DagConfig.model_validate(raw)
        assert config.node_by_id("p1").pipeline_id == "passthrough"
        assert len(config.edges) == 2


class TestBuildDag:
    """Tests for build_dag with a mock pipeline manager."""

    @pytest.fixture
    def mock_pipeline(self):
        """Minimal pipeline mock: prepare() and __call__ return video."""
        p = MagicMock()
        p.prepare.return_value = MagicMock(input_size=4)
        return p

    @pytest.fixture
    def mock_pipeline_manager(self, mock_pipeline):
        """Pipeline manager that returns the mock pipeline for any known id."""
        mgr = MagicMock()
        mgr.get_pipeline_by_id.side_effect = lambda pid: mock_pipeline
        return mgr

    def test_build_linear_dag_returns_dag_run(self, mock_pipeline_manager):
        dag = linear_dag_from_pipeline_ids(["passthrough"])
        run = build_dag(
            dag=dag,
            pipeline_manager=mock_pipeline_manager,
            initial_parameters={"pipeline_ids": ["passthrough"], "input_mode": "video"},
        )
        assert run.sink_processor is not None
        assert run.pipeline_ids == ["passthrough"]
        assert len(run.processors) == 1
        assert len(run.source_queues) == 1

    def test_build_dag_wires_source_queues(self, mock_pipeline_manager):
        dag = linear_dag_from_pipeline_ids(["passthrough"])
        run = build_dag(
            dag=dag,
            pipeline_manager=mock_pipeline_manager,
            initial_parameters={},
        )
        # One source node (input) -> one queue to first pipeline
        assert len(run.source_queues) == 1
        q = run.source_queues[0]
        assert isinstance(q, queue.Queue)

    def test_build_dag_sink_has_video_output_queue(self, mock_pipeline_manager):
        dag = linear_dag_from_pipeline_ids(["passthrough"])
        run = build_dag(
            dag=dag,
            pipeline_manager=mock_pipeline_manager,
            initial_parameters={},
        )
        assert run.sink_processor is not None
        assert run.sink_processor.output_queue is not None
        assert "video" in run.sink_processor.output_queues
