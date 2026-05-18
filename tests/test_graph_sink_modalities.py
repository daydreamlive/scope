"""Tests for ``GraphConfig.get_sink_modalities`` — the consolidated check
used by webrtc/mcp_router to decide what tracks a graph emits.
"""

from __future__ import annotations

from scope.server.graph_schema import GraphConfig


def _build(nodes: list[dict], edges: list[dict]) -> GraphConfig:
    return GraphConfig.model_validate({"nodes": nodes, "edges": edges})


def test_video_only_graph() -> None:
    g = _build(
        nodes=[
            {"id": "in", "type": "source"},
            {"id": "p", "type": "pipeline", "pipeline_id": "passthrough"},
            {"id": "out", "type": "sink"},
        ],
        edges=[
            {
                "from": "in",
                "from_port": "video",
                "to_node": "p",
                "to_port": "video",
                "kind": "stream",
            },
            {
                "from": "p",
                "from_port": "video",
                "to_node": "out",
                "to_port": "video",
                "kind": "stream",
            },
        ],
    )
    assert g.get_sink_modalities() == (True, False)


def test_audio_only_graph() -> None:
    g = _build(
        nodes=[
            {"id": "src", "type": "source"},
            {"id": "out", "type": "sink"},
        ],
        edges=[
            {
                "from": "src",
                "from_port": "audio",
                "to_node": "out",
                "to_port": "audio",
                "kind": "stream",
            }
        ],
    )
    assert g.get_sink_modalities() == (False, True)


def test_mixed_video_and_audio_sinks() -> None:
    g = _build(
        nodes=[
            {"id": "src", "type": "source"},
            {"id": "video_out", "type": "sink"},
            {"id": "audio_out", "type": "sink"},
        ],
        edges=[
            {
                "from": "src",
                "from_port": "video",
                "to_node": "video_out",
                "to_port": "video",
                "kind": "stream",
            },
            {
                "from": "src",
                "from_port": "audio",
                "to_node": "audio_out",
                "to_port": "audio",
                "kind": "stream",
            },
        ],
    )
    assert g.get_sink_modalities() == (True, True)


def test_no_sink_returns_false_false() -> None:
    g = _build(
        nodes=[{"id": "src", "type": "source"}],
        edges=[],
    )
    assert g.get_sink_modalities() == (False, False)


def test_non_stream_edges_ignored() -> None:
    # Only "stream" edges count toward sink modalities; "parameter" edges
    # carry control data and shouldn't promote a sink to video/audio.
    g = _build(
        nodes=[
            {"id": "src", "type": "source"},
            {"id": "out", "type": "sink"},
        ],
        edges=[
            {
                "from": "src",
                "from_port": "video",
                "to_node": "out",
                "to_port": "video",
                "kind": "parameter",
            }
        ],
    )
    assert g.get_sink_modalities() == (False, False)


def test_edges_not_targeting_sink_ignored() -> None:
    # An audio edge between two non-sink nodes must not mark the graph as
    # audio-producing — only edges into a sink count.
    g = _build(
        nodes=[
            {"id": "src", "type": "source"},
            {"id": "encoder", "type": "node", "node_type_id": "audio.Encode"},
            {"id": "out", "type": "sink"},
        ],
        edges=[
            {
                "from": "src",
                "from_port": "audio",
                "to_node": "encoder",
                "to_port": "audio",
                "kind": "stream",
            },
            {
                "from": "encoder",
                "from_port": "video",
                "to_node": "out",
                "to_port": "video",
                "kind": "stream",
            },
        ],
    )
    assert g.get_sink_modalities() == (True, False)
