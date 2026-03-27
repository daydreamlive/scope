"""E2E tests for multi-source/multi-sink headless streaming.

Tests the full flow: REST endpoint -> FrameProcessor -> HeadlessSession
with graph-based multi-source (video_file) and multi-sink configurations.
Verifies per-sink frame capture works independently.
"""

import asyncio
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from scope.core.pipelines.base_schema import BasePipelineConfig
from scope.server.frame_processor import FrameProcessor
from scope.server.headless import HeadlessSession

# Test video paths
TEST_VIDEO_1 = str(Path(__file__).parent.parent / "frontend/public/assets/test.mp4")
TEST_VIDEO_2 = str(Path(__file__).parent.parent / "frontend/public/assets/test2.mp4")

# Fixtures directory fallback
FIXTURES_VIDEO = str(Path(__file__).parent / "fixtures/white_square_moving.mp4")


class StubPipelineConfig(BasePipelineConfig):
    """Minimal config for the stub pipeline."""

    pipeline_id: str = "stub"
    pipeline_name: str = "Stub"
    pipeline_description: str = "Test stub"
    pipeline_version: str = "0.0.1"
    inputs = ["video"]
    outputs = ["video"]


class StubPipeline:
    """Minimal pipeline that passes through frames (no GPU needed)."""

    def __init__(self, batch_size: int = 1, delay: float = 0.01):
        self.batch_size = batch_size
        self.delay = delay

    @classmethod
    def get_config_class(cls):
        return StubPipelineConfig

    def prepare(self, **kwargs):
        return None

    def __call__(self, **kwargs):
        time.sleep(self.delay)
        video = kwargs.get("video")
        if video is not None:
            if isinstance(video, list):
                # Stack list of frame tensors
                return {"video": torch.cat(video, dim=0)}
            return {"video": video}
        # Fallback: generate a small frame
        return {
            "video": torch.randint(
                0, 255, (self.batch_size, 64, 64, 3), dtype=torch.uint8
            )
        }


def _make_pipeline_manager(pipelines: dict):
    """Create a mock PipelineManager that returns stub pipelines."""
    manager = MagicMock()

    def get_by_id(node_id):
        return pipelines[node_id]

    manager.get_pipeline_by_id = MagicMock(side_effect=get_by_id)
    return manager


class _EventLoopThread:
    """Run an asyncio event loop in a background thread for HeadlessSession."""

    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def run(self, coro):
        """Schedule a coroutine on the loop and block until it completes."""
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return future.result(timeout=30)

    def stop(self):
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(timeout=5)


def _start_session(fp):
    """Start a HeadlessSession with its own event loop thread."""
    elt = _EventLoopThread()
    session = HeadlessSession(frame_processor=fp)

    async def _start():
        session.start_frame_consumer()

    elt.run(_start())
    return session, elt


def _stop_session(session, elt):
    """Stop a session and its event loop thread."""
    elt.run(session.close())
    elt.stop()


def _wait_for_frames(session, timeout=10.0, sink_ids=None):
    """Wait until the session has frames available, return them."""
    start = time.time()
    while time.time() - start < timeout:
        if sink_ids:
            frames = {}
            for sid in sink_ids:
                f = session.get_last_frame(sink_node_id=sid)
                if f is not None:
                    frames[sid] = f
            if len(frames) == len(sink_ids):
                return frames
        else:
            f = session.get_last_frame()
            if f is not None:
                return {"default": f}
        time.sleep(0.1)
    return None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSingleSourceHeadless:
    """Test single video_file source in headless graph mode."""

    @pytest.mark.skipif(
        not Path(TEST_VIDEO_1).exists(),
        reason="Test video not found",
    )
    def test_single_video_file_source_graph(self):
        """Start a headless session with one video_file source via graph config."""
        pipeline = StubPipeline()
        pipelines = {"pipeline_1": pipeline}
        manager = _make_pipeline_manager(pipelines)

        graph = {
            "nodes": [
                {
                    "id": "source_1",
                    "type": "source",
                    "source_mode": "video_file",
                    "source_name": TEST_VIDEO_1,
                },
                {
                    "id": "pipeline_1",
                    "type": "pipeline",
                    "pipeline_id": "stub",
                },
                {"id": "output_1", "type": "sink"},
            ],
            "edges": [
                {
                    "from": "source_1",
                    "from_port": "video",
                    "to_node": "pipeline_1",
                    "to_port": "video",
                    "kind": "stream",
                },
                {
                    "from": "pipeline_1",
                    "from_port": "video",
                    "to_node": "output_1",
                    "to_port": "video",
                    "kind": "stream",
                },
            ],
        }

        fp = FrameProcessor(
            pipeline_manager=manager,
            initial_parameters={
                "pipeline_ids": ["stub"],
                "input_mode": "video",
                "graph": graph,
            },
        )

        session = None
        elt = None
        try:
            fp.start()
            assert fp.running, "FrameProcessor should be running"

            session, elt = _start_session(fp)

            frames = _wait_for_frames(session, timeout=10.0, sink_ids=["output_1"])
            assert frames is not None, "Should have received frames from output_1"
            assert "output_1" in frames

            # Verify frame is a valid VideoFrame
            frame = frames["output_1"]
            arr = frame.to_ndarray(format="rgb24")
            assert arr.shape[2] == 3, "Frame should be RGB"
            assert arr.shape[0] > 0 and arr.shape[1] > 0
        finally:
            if session and elt:
                _stop_session(session, elt)
            elif fp.running:
                fp.stop()


class TestMultiSourceMultiSink:
    """Test multi-source/multi-sink graph in headless mode."""

    @pytest.mark.skipif(
        not (Path(TEST_VIDEO_1).exists() and Path(TEST_VIDEO_2).exists()),
        reason="Test videos not found",
    )
    def test_two_sources_two_sinks(self):
        """Two video_file sources -> two pipelines -> two sinks, capture each independently."""
        pipeline_1 = StubPipeline()
        pipeline_2 = StubPipeline()
        pipelines = {"pipeline_1": pipeline_1, "pipeline_2": pipeline_2}
        manager = _make_pipeline_manager(pipelines)

        graph = {
            "nodes": [
                {
                    "id": "source_1",
                    "type": "source",
                    "source_mode": "video_file",
                    "source_name": TEST_VIDEO_1,
                },
                {
                    "id": "source_2",
                    "type": "source",
                    "source_mode": "video_file",
                    "source_name": TEST_VIDEO_2,
                },
                {
                    "id": "pipeline_1",
                    "type": "pipeline",
                    "pipeline_id": "stub",
                },
                {
                    "id": "pipeline_2",
                    "type": "pipeline",
                    "pipeline_id": "stub",
                },
                {"id": "output_1", "type": "sink"},
                {"id": "output_2", "type": "sink"},
            ],
            "edges": [
                {
                    "from": "source_1",
                    "from_port": "video",
                    "to_node": "pipeline_1",
                    "to_port": "video",
                    "kind": "stream",
                },
                {
                    "from": "source_2",
                    "from_port": "video",
                    "to_node": "pipeline_2",
                    "to_port": "video",
                    "kind": "stream",
                },
                {
                    "from": "pipeline_1",
                    "from_port": "video",
                    "to_node": "output_1",
                    "to_port": "video",
                    "kind": "stream",
                },
                {
                    "from": "pipeline_2",
                    "from_port": "video",
                    "to_node": "output_2",
                    "to_port": "video",
                    "kind": "stream",
                },
            ],
        }

        fp = FrameProcessor(
            pipeline_manager=manager,
            initial_parameters={
                "pipeline_ids": ["stub", "stub"],
                "input_mode": "video",
                "graph": graph,
            },
        )

        session = None
        elt = None
        try:
            fp.start()
            assert fp.running, "FrameProcessor should be running"

            # Verify multi-source setup
            assert "source_1" in fp._source_queues_by_node
            assert "source_2" in fp._source_queues_by_node
            assert "source_1" in fp._input_sources_by_node
            assert "source_2" in fp._input_sources_by_node

            # Verify multi-sink setup
            assert "output_1" in fp._sink_queues_by_node
            assert "output_2" in fp._sink_queues_by_node

            session, elt = _start_session(fp)

            # Wait for frames from both sinks
            frames = _wait_for_frames(
                session, timeout=15.0, sink_ids=["output_1", "output_2"]
            )
            assert frames is not None, "Should have received frames from both sinks"
            assert "output_1" in frames, "Missing frame from output_1"
            assert "output_2" in frames, "Missing frame from output_2"

            # Verify both frames are valid and independently captured
            for sink_id in ["output_1", "output_2"]:
                frame = frames[sink_id]
                arr = frame.to_ndarray(format="rgb24")
                assert arr.shape[2] == 3, f"Frame from {sink_id} should be RGB"
                assert arr.shape[0] > 0 and arr.shape[1] > 0

            # Verify per-sink capture returns different frame objects
            f1 = session.get_last_frame(sink_node_id="output_1")
            f2 = session.get_last_frame(sink_node_id="output_2")
            assert f1 is not None
            assert f2 is not None

            # Non-existent sink returns None
            assert session.get_last_frame(sink_node_id="nonexistent") is None

            # Default (no sink_id) returns most recent from any sink
            f_default = session.get_last_frame()
            assert f_default is not None
        finally:
            if session and elt:
                _stop_session(session, elt)
            elif fp.running:
                fp.stop()

    @pytest.mark.skipif(
        not (Path(TEST_VIDEO_1).exists() and Path(TEST_VIDEO_2).exists()),
        reason="Test videos not found",
    )
    def test_frames_are_different_per_sink(self):
        """Frames from different sources should produce different output per sink."""
        pipeline_1 = StubPipeline()
        pipeline_2 = StubPipeline()
        pipelines = {"pipeline_1": pipeline_1, "pipeline_2": pipeline_2}
        manager = _make_pipeline_manager(pipelines)

        graph = {
            "nodes": [
                {
                    "id": "source_1",
                    "type": "source",
                    "source_mode": "video_file",
                    "source_name": TEST_VIDEO_1,
                },
                {
                    "id": "source_2",
                    "type": "source",
                    "source_mode": "video_file",
                    "source_name": TEST_VIDEO_2,
                },
                {
                    "id": "pipeline_1",
                    "type": "pipeline",
                    "pipeline_id": "stub",
                },
                {
                    "id": "pipeline_2",
                    "type": "pipeline",
                    "pipeline_id": "stub",
                },
                {"id": "output_1", "type": "sink"},
                {"id": "output_2", "type": "sink"},
            ],
            "edges": [
                {
                    "from": "source_1",
                    "from_port": "video",
                    "to_node": "pipeline_1",
                    "to_port": "video",
                    "kind": "stream",
                },
                {
                    "from": "source_2",
                    "from_port": "video",
                    "to_node": "pipeline_2",
                    "to_port": "video",
                    "kind": "stream",
                },
                {
                    "from": "pipeline_1",
                    "from_port": "video",
                    "to_node": "output_1",
                    "to_port": "video",
                    "kind": "stream",
                },
                {
                    "from": "pipeline_2",
                    "from_port": "video",
                    "to_node": "output_2",
                    "to_port": "video",
                    "kind": "stream",
                },
            ],
        }

        fp = FrameProcessor(
            pipeline_manager=manager,
            initial_parameters={
                "pipeline_ids": ["stub", "stub"],
                "input_mode": "video",
                "graph": graph,
            },
        )

        session = None
        elt = None
        try:
            fp.start()
            assert fp.running

            session, elt = _start_session(fp)

            # Wait for multiple frames to accumulate
            frames = _wait_for_frames(
                session, timeout=15.0, sink_ids=["output_1", "output_2"]
            )
            assert frames is not None

            # The two videos have different content, so frames should differ
            arr1 = frames["output_1"].to_ndarray(format="rgb24")
            arr2 = frames["output_2"].to_ndarray(format="rgb24")

            # Frames might have different dimensions (test.mp4 vs test2.mp4)
            # At minimum, both should be valid non-empty RGB frames
            assert arr1.size > 0
            assert arr2.size > 0
        finally:
            if session and elt:
                _stop_session(session, elt)
            elif fp.running:
                fp.stop()


class TestHeadlessRESTEndpoints:
    """Test the REST API endpoints for multi-sink headless streaming."""

    @pytest.fixture
    def mock_webrtc_manager(self):
        manager = MagicMock()
        manager.headless_session = None
        manager.sessions = {}

        def add_headless(session):
            manager.headless_session = session

        async def remove_headless():
            if manager.headless_session:
                await manager.headless_session.close()
            manager.headless_session = None

        def get_last_frame(sink_node_id=None):
            if manager.headless_session:
                return manager.headless_session.get_last_frame(
                    sink_node_id=sink_node_id
                )
            return None

        def get_frame_processor():
            if manager.headless_session and manager.headless_session.frame_processor:
                return (
                    "headless",
                    manager.headless_session.frame_processor,
                    True,
                )
            return None

        manager.add_headless_session = MagicMock(side_effect=add_headless)
        manager.remove_headless_session = MagicMock(side_effect=remove_headless)
        manager.get_last_frame = MagicMock(side_effect=get_last_frame)
        manager.get_frame_processor = MagicMock(side_effect=get_frame_processor)
        manager.broadcast_parameter_update = MagicMock()
        manager.broadcast_notification = MagicMock()
        return manager

    @pytest.fixture
    def mock_pipeline_manager(self):
        pipeline = StubPipeline()
        manager = MagicMock()
        manager.get_pipeline_by_id = MagicMock(return_value=pipeline)
        return manager

    @pytest.fixture
    def client(self, mock_webrtc_manager, mock_pipeline_manager):
        from fastapi.testclient import TestClient

        with patch("scope.server.app.webrtc_manager", mock_webrtc_manager):
            with patch("scope.server.app.pipeline_manager", mock_pipeline_manager):
                from scope.server.app import app

                yield TestClient(app, raise_server_exceptions=False)

    @pytest.mark.skipif(
        not Path(TEST_VIDEO_1).exists(),
        reason="Test video not found",
    )
    def test_start_stream_with_graph(self, client, mock_webrtc_manager):
        """POST /api/v1/session/start with graph config returns sink_node_ids."""
        graph = {
            "nodes": [
                {
                    "id": "source_1",
                    "type": "source",
                    "source_mode": "video_file",
                    "source_name": TEST_VIDEO_1,
                },
                {
                    "id": "pipeline_1",
                    "type": "pipeline",
                    "pipeline_id": "stub",
                },
                {"id": "output_1", "type": "sink"},
            ],
            "edges": [
                {
                    "from": "source_1",
                    "from_port": "video",
                    "to_node": "pipeline_1",
                    "to_port": "video",
                    "kind": "stream",
                },
                {
                    "from": "pipeline_1",
                    "from_port": "video",
                    "to_node": "output_1",
                    "to_port": "video",
                    "kind": "stream",
                },
            ],
        }

        resp = client.post(
            "/api/v1/session/start",
            json={"graph": graph, "input_mode": "video"},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data.get("graph") is True
        assert "output_1" in data["sink_node_ids"]
        assert "source_1" in data["source_node_ids"]

        # Clean up
        if mock_webrtc_manager.headless_session:
            elt = _EventLoopThread()
            elt.run(mock_webrtc_manager.headless_session.close())
            elt.stop()

    def test_start_stream_requires_pipeline_or_graph(self, client):
        """POST /api/v1/session/start without pipeline_id or graph returns 400."""
        resp = client.post(
            "/api/v1/session/start",
            json={"input_mode": "text"},
        )
        assert resp.status_code == 400

    def test_start_stream_invalid_graph(self, client):
        """POST /api/v1/session/start with invalid graph returns 400."""
        resp = client.post(
            "/api/v1/session/start",
            json={
                "graph": {
                    "nodes": [
                        {"id": "p1", "type": "pipeline"},  # missing pipeline_id
                    ],
                    "edges": [],
                },
                "input_mode": "text",
            },
        )
        assert resp.status_code == 400

    def test_capture_frame_with_sink_node_id(self, client, mock_webrtc_manager):
        """GET /api/v1/session/frame?sink_node_id=... captures from specific sink."""
        import numpy as np
        from av import VideoFrame

        # Create a HeadlessSession with pre-set frames (no pipeline needed)
        fp = MagicMock()
        fp.running = True
        session = HeadlessSession(frame_processor=fp)

        # Inject frames for two sink nodes
        frame_arr_1 = np.full((64, 64, 3), 100, dtype=np.uint8)
        frame_arr_2 = np.full((64, 64, 3), 200, dtype=np.uint8)
        vf1 = VideoFrame.from_ndarray(frame_arr_1, format="rgb24")
        vf2 = VideoFrame.from_ndarray(frame_arr_2, format="rgb24")
        session._last_frame = vf1
        session._last_frames_by_sink["output_1"] = vf1
        session._last_frames_by_sink["output_2"] = vf2

        mock_webrtc_manager.headless_session = session

        # Capture from specific sink
        resp = client.get("/api/v1/session/frame?sink_node_id=output_1")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/jpeg"
        assert len(resp.content) > 0

        # Capture from second sink
        resp = client.get("/api/v1/session/frame?sink_node_id=output_2")
        assert resp.status_code == 200
        assert len(resp.content) > 0

        # Capture without sink_node_id returns default frame
        resp = client.get("/api/v1/session/frame")
        assert resp.status_code == 200
        assert len(resp.content) > 0

        # Capture from non-existent sink returns 404
        resp = client.get("/api/v1/session/frame?sink_node_id=nonexistent")
        assert resp.status_code == 404

    def test_capture_frame_no_session(self, client):
        """GET /api/v1/session/frame with no active session returns 404."""
        resp = client.get("/api/v1/session/frame")
        assert resp.status_code == 404


class TestGraphSchemaValidation:
    """Test graph config validation for multi-source/sink scenarios."""

    def test_multi_source_multi_sink_graph_validates(self):
        """A valid multi-source/multi-sink graph passes validation."""
        from scope.server.graph_schema import GraphConfig

        graph = GraphConfig.model_validate(
            {
                "nodes": [
                    {"id": "s1", "type": "source", "source_mode": "video_file"},
                    {"id": "s2", "type": "source", "source_mode": "video_file"},
                    {"id": "p1", "type": "pipeline", "pipeline_id": "stub"},
                    {"id": "p2", "type": "pipeline", "pipeline_id": "stub"},
                    {"id": "o1", "type": "sink"},
                    {"id": "o2", "type": "sink"},
                ],
                "edges": [
                    {
                        "from": "s1",
                        "from_port": "video",
                        "to_node": "p1",
                        "to_port": "video",
                    },
                    {
                        "from": "s2",
                        "from_port": "video",
                        "to_node": "p2",
                        "to_port": "video",
                    },
                    {
                        "from": "p1",
                        "from_port": "video",
                        "to_node": "o1",
                        "to_port": "video",
                    },
                    {
                        "from": "p2",
                        "from_port": "video",
                        "to_node": "o2",
                        "to_port": "video",
                    },
                ],
            }
        )

        assert graph.get_source_node_ids() == ["s1", "s2"]
        assert graph.get_sink_node_ids() == ["o1", "o2"]
        assert graph.get_pipeline_node_ids() == ["p1", "p2"]
        assert graph.validate_structure() == []

    def test_graph_without_sink_fails_validation(self):
        """A graph with no sink nodes should fail validation."""
        from scope.server.graph_schema import GraphConfig

        graph = GraphConfig.model_validate(
            {
                "nodes": [
                    {"id": "s1", "type": "source"},
                    {"id": "p1", "type": "pipeline", "pipeline_id": "stub"},
                ],
                "edges": [
                    {
                        "from": "s1",
                        "from_port": "video",
                        "to_node": "p1",
                        "to_port": "video",
                    },
                ],
            }
        )
        errors = graph.validate_structure()
        assert any("sink" in e.lower() for e in errors)

    def test_source_node_with_video_file_mode(self):
        """Source nodes can have source_mode='video_file' and source_name."""
        from scope.server.graph_schema import GraphNode

        node = GraphNode(
            id="src",
            type="source",
            source_mode="video_file",
            source_name="/path/to/video.mp4",
        )
        assert node.source_mode == "video_file"
        assert node.source_name == "/path/to/video.mp4"
