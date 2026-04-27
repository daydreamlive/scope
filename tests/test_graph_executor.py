import queue

from scope.server.graph_executor import build_graph
from scope.server.graph_schema import GraphConfig


class _StubRequirements:
    input_size = 1


class _StubPipelineConfig:
    inputs = ["video"]
    outputs = ["video"]


class _StubPipeline:
    def get_config_class(self):
        return _StubPipelineConfig

    def prepare(self, **kwargs):
        return _StubRequirements()


class _StubPipelineManager:
    def __init__(self):
        self._pipeline = _StubPipeline()

    def get_pipeline_by_id(self, node_id: str):
        return self._pipeline


def test_build_graph_routes_direct_source_sink_edges():
    graph = GraphConfig.model_validate(
        {
            "nodes": [
                {"id": "input", "type": "source", "source_mode": "syphon"},
                {
                    "id": "passthrough",
                    "type": "pipeline",
                    "pipeline_id": "passthrough",
                },
                {"id": "preview", "type": "sink"},
                {
                    "id": "syphon_out",
                    "type": "sink",
                    "sink_mode": "syphon",
                    "sink_name": "Scope",
                },
            ],
            "edges": [
                {
                    "from": "input",
                    "from_port": "video",
                    "to_node": "passthrough",
                    "to_port": "video",
                    "kind": "stream",
                },
                {
                    "from": "input",
                    "from_port": "video",
                    "to_node": "preview",
                    "to_port": "video",
                    "kind": "stream",
                },
                {
                    "from": "input",
                    "from_port": "video",
                    "to_node": "syphon_out",
                    "to_port": "video",
                    "kind": "stream",
                },
            ],
        }
    )

    graph_run = build_graph(
        graph=graph,
        pipeline_manager=_StubPipelineManager(),
        initial_parameters={},
    )

    input_queues = graph_run.source_queues_by_node["input"]
    assert len(input_queues) == 4

    preview_queue = graph_run.sink_queues_by_node["preview"]
    syphon_preview_queue = graph_run.sink_queues_by_node["syphon_out"]
    syphon_hardware_queue = graph_run.sink_hardware_queues_by_node["syphon_out"]

    assert preview_queue in input_queues
    assert syphon_preview_queue in input_queues
    assert syphon_hardware_queue in input_queues
    assert "preview" not in graph_run.sink_processors_by_node
    assert "syphon_out" not in graph_run.sink_processors_by_node


def test_build_graph_routes_direct_source_record_edges():
    graph = GraphConfig.model_validate(
        {
            "nodes": [
                {"id": "input", "type": "source", "source_mode": "syphon"},
                {"id": "preview", "type": "sink"},
                {"id": "record", "type": "record"},
            ],
            "edges": [
                {
                    "from": "input",
                    "from_port": "video",
                    "to_node": "preview",
                    "to_port": "video",
                    "kind": "stream",
                },
                {
                    "from": "input",
                    "from_port": "video",
                    "to_node": "record",
                    "to_port": "video",
                    "kind": "stream",
                },
            ],
        }
    )

    graph_run = build_graph(
        graph=graph,
        pipeline_manager=_StubPipelineManager(),
        initial_parameters={},
    )

    record_queue = graph_run.record_queues_by_node["record"]
    input_queues = graph_run.source_queues_by_node["input"]

    assert isinstance(record_queue, queue.Queue)
    assert record_queue in input_queues
    assert graph_run.sink_queues_by_node["preview"] in input_queues


def test_build_graph_preserves_pipeline_to_sink_routing():
    graph = GraphConfig.model_validate(
        {
            "nodes": [
                {"id": "input", "type": "source"},
                {"id": "pipe", "type": "pipeline", "pipeline_id": "passthrough"},
                {"id": "output", "type": "sink"},
            ],
            "edges": [
                {
                    "from": "input",
                    "from_port": "video",
                    "to_node": "pipe",
                    "to_port": "video",
                    "kind": "stream",
                },
                {
                    "from": "pipe",
                    "from_port": "video",
                    "to_node": "output",
                    "to_port": "video",
                    "kind": "stream",
                },
            ],
        }
    )

    graph_run = build_graph(
        graph=graph,
        pipeline_manager=_StubPipelineManager(),
        initial_parameters={},
    )

    assert graph_run.sink_processors_by_node["output"].node_id == "pipe"
    assert (
        graph_run.sink_queues_by_node["output"]
        not in graph_run.source_queues_by_node["input"]
    )


def test_build_graph_preserves_pipeline_to_record_routing():
    graph = GraphConfig.model_validate(
        {
            "nodes": [
                {"id": "input", "type": "source"},
                {"id": "pipe", "type": "pipeline", "pipeline_id": "passthrough"},
                {"id": "output", "type": "sink"},
                {"id": "record", "type": "record"},
            ],
            "edges": [
                {
                    "from": "input",
                    "from_port": "video",
                    "to_node": "pipe",
                    "to_port": "video",
                    "kind": "stream",
                },
                {
                    "from": "pipe",
                    "from_port": "video",
                    "to_node": "output",
                    "to_port": "video",
                    "kind": "stream",
                },
                {
                    "from": "pipe",
                    "from_port": "video",
                    "to_node": "record",
                    "to_port": "video",
                    "kind": "stream",
                },
            ],
        }
    )

    graph_run = build_graph(
        graph=graph,
        pipeline_manager=_StubPipelineManager(),
        initial_parameters={},
    )

    record_queue = graph_run.record_queues_by_node["record"]
    assert isinstance(record_queue, queue.Queue)
    assert record_queue not in graph_run.source_queues_by_node["input"]


def test_build_graph_routes_source_sink_record_path():
    graph = GraphConfig.model_validate(
        {
            "nodes": [
                {"id": "input", "type": "source", "source_mode": "syphon"},
                {"id": "preview", "type": "sink"},
                {"id": "record", "type": "record"},
            ],
            "edges": [
                {
                    "from": "input",
                    "from_port": "video",
                    "to_node": "preview",
                    "to_port": "video",
                    "kind": "stream",
                },
                {
                    "from": "preview",
                    "from_port": "video",
                    "to_node": "record",
                    "to_port": "video",
                    "kind": "stream",
                },
            ],
        }
    )

    graph_run = build_graph(
        graph=graph,
        pipeline_manager=_StubPipelineManager(),
        initial_parameters={},
    )

    record_queue = graph_run.record_queues_by_node["record"]
    input_queues = graph_run.source_queues_by_node["input"]

    assert isinstance(record_queue, queue.Queue)
    assert record_queue in input_queues
    assert graph_run.sink_queues_by_node["preview"] in input_queues
