"""Tests for hardware sink routing in cloud-relay mode.

Hardware sinks (Syphon/NDI/Spout) always run on the local machine; the cloud
runner only renders the pipeline. The webrtc relay code is responsible for
mapping each hardware sink to the cloud output handler whose frames should
feed it — typically the handler of a webrtc sink that shares the same
upstream pipeline.
"""

from unittest.mock import MagicMock

from scope.server.livepeer_client import LivepeerClient, _BrowserGraphInfo
from scope.server.webrtc import _build_cloud_output_taps


def _hw_routes(taps):
    """Filter tap list down to the (handler_index, node_id) pairs we can
    inspect for hardware sinks. We recover node_id by inspecting the
    closure of each callback (set up via _make_put_to_sink_cb)."""
    routes = []
    for handler_index, callback in taps:
        # Hardware-sink callbacks call sink_manager.put_to_sink(node_id, frame);
        # record callbacks call put_to_record. We can't easily distinguish
        # them from the outside, but in tests we register a MagicMock as the
        # frame_processor so any captured node_id is in cell_contents.
        cells = callback.__closure__ or ()
        node_id = next(
            (c.cell_contents for c in cells if isinstance(c.cell_contents, str)),
            None,
        )
        routes.append((handler_index, node_id))
    return routes


def _fake_fp():
    """A frame_processor stand-in whose sink_manager just records calls."""
    return MagicMock()


def test_single_pipeline_webrtc_plus_syphon():
    """User's case: one pipeline feeds both a webrtc sink and a Syphon sink."""
    params = {
        "graph": {
            "nodes": [
                {"id": "input", "type": "source", "source_mode": "camera"},
                {"id": "output", "type": "sink"},
                {"id": "longlive", "type": "pipeline", "pipeline_id": "longlive"},
                {
                    "id": "output_sink",
                    "type": "sink",
                    "sink_mode": "syphon",
                    "sink_name": "Scope",
                },
            ],
            "edges": [
                _edge("input", "longlive"),
                _edge("longlive", "output"),
                _edge("longlive", "output_sink"),
            ],
        }
    }
    taps = _build_cloud_output_taps(
        initial_parameters=params,
        all_sink_node_ids=["output", "output_sink"],
        record_node_ids=[],
        frame_processor=_fake_fp(),
    )
    assert _hw_routes(taps) == [(0, "output_sink")]


def test_no_hardware_sinks_returns_empty():
    params = {
        "graph": {
            "nodes": [
                {"id": "input", "type": "source", "source_mode": "camera"},
                {"id": "output", "type": "sink"},
            ],
            "edges": [_edge("input", "output")],
        }
    }
    taps = _build_cloud_output_taps(
        initial_parameters=params,
        all_sink_node_ids=["output"],
        record_node_ids=[],
        frame_processor=_fake_fp(),
    )
    assert taps == []


def test_multi_pipeline_hw_sink_routes_to_buddy_webrtc_sink():
    """Two pipelines, each with a webrtc sink and a hardware sink.

    Each hardware sink must wire to the cloud output handler of the webrtc
    sink that shares its upstream pipeline, NOT just handler 0.
    """
    params = {
        "graph": {
            "nodes": [
                {"id": "input_a", "type": "source", "source_mode": "camera"},
                {"id": "input_b", "type": "source", "source_mode": "camera"},
                {"id": "pipeline_a", "type": "pipeline", "pipeline_id": "passthrough"},
                {"id": "pipeline_b", "type": "pipeline", "pipeline_id": "passthrough"},
                {"id": "out_a", "type": "sink"},
                {"id": "out_b", "type": "sink"},
                {
                    "id": "syphon_a",
                    "type": "sink",
                    "sink_mode": "syphon",
                    "sink_name": "A",
                },
                {
                    "id": "syphon_b",
                    "type": "sink",
                    "sink_mode": "syphon",
                    "sink_name": "B",
                },
            ],
            "edges": [
                _edge("input_a", "pipeline_a"),
                _edge("pipeline_a", "out_a"),
                _edge("pipeline_a", "syphon_a"),
                _edge("input_b", "pipeline_b"),
                _edge("pipeline_b", "out_b"),
                _edge("pipeline_b", "syphon_b"),
            ],
        }
    }
    taps = _build_cloud_output_taps(
        initial_parameters=params,
        all_sink_node_ids=["out_a", "out_b", "syphon_a", "syphon_b"],
        record_node_ids=[],
        frame_processor=_fake_fp(),
    )
    # syphon_a shares pipeline_a with out_a (handler 0).
    # syphon_b shares pipeline_b with out_b (handler 1).
    assert sorted(_hw_routes(taps)) == [(0, "syphon_a"), (1, "syphon_b")]


def test_spout_and_ndi_treated_as_hardware():
    params = {
        "graph": {
            "nodes": [
                {"id": "src", "type": "source", "source_mode": "camera"},
                {"id": "pipe", "type": "pipeline", "pipeline_id": "passthrough"},
                {"id": "browser", "type": "sink"},
                {"id": "spout", "type": "sink", "sink_mode": "spout"},
                {"id": "ndi", "type": "sink", "sink_mode": "ndi"},
            ],
            "edges": [
                _edge("src", "pipe"),
                _edge("pipe", "browser"),
                _edge("pipe", "spout"),
                _edge("pipe", "ndi"),
            ],
        }
    }
    taps = _build_cloud_output_taps(
        initial_parameters=params,
        all_sink_node_ids=["browser", "spout", "ndi"],
        record_node_ids=[],
        frame_processor=_fake_fp(),
    )
    assert sorted(_hw_routes(taps)) == [(0, "ndi"), (0, "spout")]


def test_orphan_hardware_sink_is_dropped_not_silently_mis_routed(caplog):
    """A hardware sink whose upstream pipeline has no webrtc-sink buddy is
    dropped with a warning, not silently wired to handler 0 (which would
    deliver some other pipeline's frames)."""
    params = {
        "graph": {
            "nodes": [
                {"id": "input_a", "type": "source", "source_mode": "camera"},
                {"id": "input_b", "type": "source", "source_mode": "camera"},
                {"id": "pipeline_a", "type": "pipeline", "pipeline_id": "passthrough"},
                {"id": "pipeline_b", "type": "pipeline", "pipeline_id": "passthrough"},
                {"id": "out_a", "type": "sink"},
                {
                    "id": "syphon_orphan",
                    "type": "sink",
                    "sink_mode": "syphon",
                    "sink_name": "B",
                },
            ],
            "edges": [
                _edge("input_a", "pipeline_a"),
                _edge("pipeline_a", "out_a"),
                _edge("input_b", "pipeline_b"),
                _edge("pipeline_b", "syphon_orphan"),
            ],
        }
    }
    with caplog.at_level("WARNING"):
        taps = _build_cloud_output_taps(
            initial_parameters=params,
            all_sink_node_ids=["out_a", "syphon_orphan"],
            record_node_ids=[],
            frame_processor=_fake_fp(),
        )
    assert taps == []
    assert "syphon_orphan" in caplog.text
    assert "no webrtc sink shares" in caplog.text


def test_record_nodes_get_taps_after_webrtc_sinks():
    """Record-node taps come after the webrtc sinks in the output handler
    index space."""
    params = {
        "graph": {
            "nodes": [
                {"id": "input", "type": "source", "source_mode": "camera"},
                {"id": "pipe", "type": "pipeline", "pipeline_id": "passthrough"},
                {"id": "out_a", "type": "sink"},
                {"id": "out_b", "type": "sink"},
                {"id": "rec", "type": "record"},
            ],
            "edges": [
                _edge("input", "pipe"),
                _edge("pipe", "out_a"),
                _edge("pipe", "out_b"),
            ],
        }
    }
    taps = _build_cloud_output_taps(
        initial_parameters=params,
        all_sink_node_ids=["out_a", "out_b"],
        record_node_ids=["rec"],
        frame_processor=_fake_fp(),
    )
    # 2 webrtc sinks → record tap should land at handler 2.
    handler_indices = [t[0] for t in taps]
    assert handler_indices == [2]


def test_filter_runner_params_strips_hardware_sinks():
    """The graph sent to the cloud runner must not contain hardware sinks."""
    initial = {
        "graph": {
            "nodes": [
                {"id": "input", "type": "source", "source_mode": "camera"},
                {"id": "output", "type": "sink"},
                {"id": "longlive", "type": "pipeline", "pipeline_id": "longlive"},
                {
                    "id": "output_sink",
                    "type": "sink",
                    "sink_mode": "syphon",
                    "sink_name": "Scope",
                },
            ],
            "edges": [
                _edge("input", "longlive"),
                _edge("longlive", "output"),
                _edge("longlive", "output_sink"),
            ],
        }
    }
    parsed = LivepeerClient._parse_browser_graph(initial)
    filtered = LivepeerClient._filter_runner_params(initial, parsed)
    node_ids = [n["id"] for n in filtered["graph"]["nodes"]]
    assert "output_sink" not in node_ids
    assert "output" in node_ids
    # The edge feeding the hardware sink should be gone.
    edges = filtered["graph"]["edges"]
    assert all(e.get("to_node") != "output_sink" for e in edges)


def test_filter_runner_params_preserves_graph_when_no_hardware():
    initial = {
        "graph": {
            "nodes": [
                {"id": "input", "type": "source", "source_mode": "camera"},
                {"id": "output", "type": "sink"},
            ],
            "edges": [_edge("input", "output")],
        }
    }
    parsed = LivepeerClient._parse_browser_graph(initial)
    filtered = LivepeerClient._filter_runner_params(initial, parsed)
    assert filtered["graph"] == initial["graph"]


def test_filter_runner_params_strips_both_hardware_and_sink_teed_records():
    """Both transformations should compose.

    Note: sink-teed-record detection in _parse_browser_graph uses
    ``edge.get("from")`` (alias) rather than ``from_node``, so the
    sink-teed edge here uses the alias form.
    """
    initial = {
        "graph": {
            "nodes": [
                {"id": "input", "type": "source", "source_mode": "camera"},
                {"id": "pipe", "type": "pipeline", "pipeline_id": "passthrough"},
                {"id": "output", "type": "sink"},
                {"id": "rec", "type": "record"},
                {"id": "syphon", "type": "sink", "sink_mode": "syphon"},
            ],
            "edges": [
                _edge("input", "pipe"),
                _edge("pipe", "output"),
                {
                    "from": "output",
                    "to_node": "rec",
                    "from_port": "video",
                    "to_port": "video",
                    "kind": "stream",
                },
                _edge("pipe", "syphon"),
            ],
        }
    }
    parsed = LivepeerClient._parse_browser_graph(initial)
    filtered = LivepeerClient._filter_runner_params(initial, parsed)
    node_ids = {n["id"] for n in filtered["graph"]["nodes"]}
    assert "syphon" not in node_ids
    assert "rec" not in node_ids
    assert {"input", "pipe", "output"} <= node_ids


def test_browser_graph_info_keeps_hardware_sinks_out_of_sink_node_ids():
    """Hardware sinks shouldn't appear in sink_node_ids; cloud doesn't know
    about them after _filter_runner_params strips them."""
    initial = {
        "graph": {
            "nodes": [
                {"id": "input", "type": "source", "source_mode": "camera"},
                {"id": "output", "type": "sink"},
                {"id": "syphon", "type": "sink", "sink_mode": "syphon"},
            ],
            "edges": [],
        }
    }
    parsed: _BrowserGraphInfo = LivepeerClient._parse_browser_graph(initial)
    assert parsed.sink_node_ids == ["output"]


def _edge(from_node: str, to_node: str) -> dict:
    return {
        "from_node": from_node,
        "to_node": to_node,
        "from_port": "video",
        "to_port": "video",
        "kind": "stream",
    }
