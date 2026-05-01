"""Tests for LivepeerClient graph parsing and sink-attached record collapsing."""

from __future__ import annotations

from scope.server.livepeer_client import LivepeerClient

# Graph exported from the UI (sink-recording workflow).
# - input (source) -> passthrough (pipeline), gray_1 (pipeline)
# - gray_1 -> output_1 (sink) -> record (record)   <-- sink-teed
# - passthrough -> record_1 (record)                <-- pipeline-attached
SINK_RECORDING_GRAPH: dict = {
    "nodes": [
        {"id": "input", "type": "source", "source_mode": "video"},
        {"id": "passthrough", "type": "pipeline", "pipeline_id": "passthrough"},
        {"id": "gray_1", "type": "pipeline", "pipeline_id": "gray"},
        {"id": "output_1", "type": "sink"},
        {"id": "record", "type": "record"},
        {"id": "record_1", "type": "record"},
    ],
    "edges": [
        {
            "from": "output_1",
            "from_port": "out",
            "to_node": "record",
            "to_port": "video",
            "kind": "stream",
        },
        {
            "from": "gray_1",
            "from_port": "video",
            "to_node": "output_1",
            "to_port": "video",
            "kind": "stream",
        },
        {
            "from": "passthrough",
            "from_port": "video",
            "to_node": "record_1",
            "to_port": "video",
            "kind": "stream",
        },
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
            "to_node": "gray_1",
            "to_port": "video",
            "kind": "stream",
        },
    ],
}


def _params(graph: dict) -> dict:
    return {"graph": graph}


def test_parse_classifies_sink_attached_record():
    parsed = LivepeerClient._parse_browser_graph(_params(SINK_RECORDING_GRAPH))

    assert parsed.sink_node_ids == ["output_1"]
    assert parsed.record_node_ids == ["record", "record_1"]
    assert parsed.remote_record_node_ids == ["record_1"]
    assert parsed.sink_teed_records == [("record", 0)]


def test_filter_removes_sink_attached_record_from_runner_params():
    params = _params(SINK_RECORDING_GRAPH)
    parsed = LivepeerClient._parse_browser_graph(params)
    filtered = LivepeerClient._filter_runner_params(params, parsed)

    node_ids = {n["id"] for n in filtered["graph"]["nodes"]}
    assert "record" not in node_ids, "sink-teed record should be removed"
    assert "record_1" in node_ids, "pipeline-attached record should remain"
    assert "output_1" in node_ids

    to_nodes = {e["to_node"] for e in filtered["graph"]["edges"]}
    from_nodes = {e["from"] for e in filtered["graph"]["edges"]}
    assert "record" not in to_nodes
    assert "record" not in from_nodes


def test_filter_does_not_mutate_original_params():
    params = _params(SINK_RECORDING_GRAPH)
    parsed = LivepeerClient._parse_browser_graph(params)
    LivepeerClient._filter_runner_params(params, parsed)

    node_ids = {n["id"] for n in params["graph"]["nodes"]}
    assert "record" in node_ids, "original params should be unchanged"


def test_no_records_returns_identity():
    graph = {
        "nodes": [
            {"id": "input", "type": "source", "source_mode": "video"},
            {"id": "p", "type": "pipeline", "pipeline_id": "passthrough"},
            {"id": "output", "type": "sink"},
        ],
        "edges": [
            {
                "from": "input",
                "from_port": "video",
                "to_node": "p",
                "to_port": "video",
                "kind": "stream",
            },
            {
                "from": "p",
                "from_port": "video",
                "to_node": "output",
                "to_port": "video",
                "kind": "stream",
            },
        ],
    }
    params = _params(graph)
    parsed = LivepeerClient._parse_browser_graph(params)

    assert parsed.record_node_ids == []
    assert parsed.remote_record_node_ids == []
    assert parsed.sink_teed_records == []

    filtered = LivepeerClient._filter_runner_params(params, parsed)
    assert len(filtered["graph"]["nodes"]) == len(graph["nodes"])


def test_all_records_pipeline_attached():
    graph = {
        "nodes": [
            {"id": "input", "type": "source", "source_mode": "video"},
            {"id": "p", "type": "pipeline", "pipeline_id": "passthrough"},
            {"id": "output", "type": "sink"},
            {"id": "rec", "type": "record"},
        ],
        "edges": [
            {
                "from": "input",
                "from_port": "video",
                "to_node": "p",
                "to_port": "video",
                "kind": "stream",
            },
            {
                "from": "p",
                "from_port": "video",
                "to_node": "output",
                "to_port": "video",
                "kind": "stream",
            },
            {
                "from": "p",
                "from_port": "video",
                "to_node": "rec",
                "to_port": "video",
                "kind": "stream",
            },
        ],
    }
    parsed = LivepeerClient._parse_browser_graph(_params(graph))

    assert parsed.record_node_ids == ["rec"]
    assert parsed.remote_record_node_ids == ["rec"]
    assert parsed.sink_teed_records == []

    filtered = LivepeerClient._filter_runner_params(_params(graph), parsed)
    assert {n["id"] for n in filtered["graph"]["nodes"]} == {
        "input",
        "p",
        "output",
        "rec",
    }


def test_output_mapping_sink_recording_workflow():
    parsed = LivepeerClient._parse_browser_graph(_params(SINK_RECORDING_GRAPH))
    mapping = LivepeerClient._build_output_mapping(parsed)

    # 1 sink + 1 pipeline-attached record = 2 runner outputs
    assert mapping.num_output_tracks == 2
    # 1 sink + 2 total records = 3 local handlers
    assert mapping.num_local_handlers == 3
    # runner[0] -> local[0] (sink), runner[1] -> local[2] (record_1)
    assert mapping.remote_to_local == [0, 2]
    # sink handler 0 mirrors into record handler 1
    assert mapping.sink_tee_pairs == [(0, 1)]


def test_output_mapping_no_records():
    graph = {
        "nodes": [
            {"id": "input", "type": "source", "source_mode": "video"},
            {"id": "p", "type": "pipeline", "pipeline_id": "passthrough"},
            {"id": "output", "type": "sink"},
        ],
        "edges": [
            {
                "from": "input",
                "from_port": "video",
                "to_node": "p",
                "to_port": "video",
                "kind": "stream",
            },
            {
                "from": "p",
                "from_port": "video",
                "to_node": "output",
                "to_port": "video",
                "kind": "stream",
            },
        ],
    }
    parsed = LivepeerClient._parse_browser_graph(_params(graph))
    mapping = LivepeerClient._build_output_mapping(parsed)

    assert mapping.num_output_tracks == 1
    assert mapping.num_local_handlers == 1
    assert mapping.remote_to_local == [0]
    assert mapping.sink_tee_pairs == []


def test_output_mapping_all_pipeline_attached():
    graph = {
        "nodes": [
            {"id": "input", "type": "source", "source_mode": "video"},
            {"id": "p", "type": "pipeline", "pipeline_id": "passthrough"},
            {"id": "output", "type": "sink"},
            {"id": "rec", "type": "record"},
        ],
        "edges": [
            {
                "from": "input",
                "from_port": "video",
                "to_node": "p",
                "to_port": "video",
                "kind": "stream",
            },
            {
                "from": "p",
                "from_port": "video",
                "to_node": "output",
                "to_port": "video",
                "kind": "stream",
            },
            {
                "from": "p",
                "from_port": "video",
                "to_node": "rec",
                "to_port": "video",
                "kind": "stream",
            },
        ],
    }
    parsed = LivepeerClient._parse_browser_graph(_params(graph))
    mapping = LivepeerClient._build_output_mapping(parsed)

    # 1 sink + 1 remote record = 2 outputs, 1:1 mapping
    assert mapping.num_output_tracks == 2
    assert mapping.num_local_handlers == 2
    assert mapping.remote_to_local == [0, 1]
    assert mapping.sink_tee_pairs == []


def test_empty_params():
    parsed = LivepeerClient._parse_browser_graph(None)
    assert parsed.record_node_ids == []
    assert parsed.sink_node_ids == []

    filtered = LivepeerClient._filter_runner_params(None, parsed)
    assert filtered == {}
