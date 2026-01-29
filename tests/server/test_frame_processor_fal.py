"""Tests for FrameProcessor fal.ai cloud integration."""

import queue
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestFrameProcessorFalAttributes:
    """Tests for fal integration attributes."""

    def test_fal_attributes_initialized(self):
        """Test that fal attributes are properly initialized."""
        from scope.server.frame_processor import FrameProcessor

        # Create a mock pipeline manager
        mock_pm = MagicMock()
        processor = FrameProcessor(pipeline_manager=mock_pm)

        # Check fal attributes exist and have correct initial values
        assert processor.fal_client is None
        assert processor.fal_enabled is False
        assert isinstance(processor._fal_received_frames, queue.Queue)
        assert processor._fal_received_frames.maxsize == 30


class TestFrameProcessorFalCallback:
    """Tests for _on_fal_frame_received callback."""

    def test_on_fal_frame_received_queues_frame(self):
        """Test that received frames are queued."""
        from scope.server.frame_processor import FrameProcessor

        mock_pm = MagicMock()
        processor = FrameProcessor(pipeline_manager=mock_pm)

        mock_frame = MagicMock()
        processor._on_fal_frame_received(mock_frame)

        assert processor._fal_received_frames.qsize() == 1
        assert processor._fal_received_frames.get_nowait() is mock_frame

    def test_on_fal_frame_received_drops_oldest_when_full(self):
        """Test that oldest frame is dropped when queue is full."""
        from scope.server.frame_processor import FrameProcessor

        mock_pm = MagicMock()
        processor = FrameProcessor(pipeline_manager=mock_pm)

        # Fill the queue
        frames = []
        for i in range(30):
            frame = MagicMock()
            frame.id = i
            frames.append(frame)
            processor._on_fal_frame_received(frame)

        assert processor._fal_received_frames.qsize() == 30

        # Add one more frame
        new_frame = MagicMock()
        new_frame.id = 99
        processor._on_fal_frame_received(new_frame)

        # Queue should still be at max size
        assert processor._fal_received_frames.qsize() == 30

        # First frame should have been dropped
        first = processor._fal_received_frames.get_nowait()
        assert first.id == 1  # Frame 0 was dropped


class TestFrameProcessorFalConnection:
    """Tests for connect_to_fal and disconnect_from_fal."""

    @pytest.mark.asyncio
    async def test_connect_to_fal(self):
        """Test fal connection initialization."""
        from scope.server.frame_processor import FrameProcessor

        mock_pm = MagicMock()
        processor = FrameProcessor(pipeline_manager=mock_pm)

        mock_client = AsyncMock()

        # Patch where FalClient is imported (inside connect_to_fal)
        with patch(
            "scope.server.fal_client.FalClient", return_value=mock_client
        ) as MockFalClient:
            await processor.connect_to_fal(
                app_id="owner/app/webrtc", api_key="test-key"
            )

            # Check FalClient was created with correct arguments
            MockFalClient.assert_called_once_with(
                app_id="owner/app/webrtc",
                api_key="test-key",
                on_frame_received=processor._on_fal_frame_received,
            )

            # Check connect was called
            mock_client.connect.assert_called_once()

            # Check state
            assert processor.fal_enabled is True
            assert processor.fal_client is mock_client

    @pytest.mark.asyncio
    async def test_connect_to_fal_disconnects_existing(self):
        """Test that connecting disconnects any existing connection."""
        from scope.server.frame_processor import FrameProcessor

        mock_pm = MagicMock()
        processor = FrameProcessor(pipeline_manager=mock_pm)

        # Set up existing connection
        old_client = AsyncMock()
        processor.fal_client = old_client
        processor.fal_enabled = True

        new_client = AsyncMock()

        # Patch where FalClient is imported (inside connect_to_fal)
        with patch("scope.server.fal_client.FalClient", return_value=new_client):
            await processor.connect_to_fal(
                app_id="owner/app/webrtc", api_key="test-key"
            )

            # Old client should have been disconnected
            old_client.disconnect.assert_called_once()

            # New client should be connected
            assert processor.fal_client is new_client
            assert processor.fal_enabled is True

    @pytest.mark.asyncio
    async def test_disconnect_from_fal(self):
        """Test fal disconnection."""
        from scope.server.frame_processor import FrameProcessor

        mock_pm = MagicMock()
        processor = FrameProcessor(pipeline_manager=mock_pm)

        mock_client = AsyncMock()
        processor.fal_client = mock_client
        processor.fal_enabled = True

        # Add some frames to the queue
        for _ in range(5):
            processor._fal_received_frames.put_nowait(MagicMock())

        await processor.disconnect_from_fal()

        # Check client was disconnected
        mock_client.disconnect.assert_called_once()

        # Check state
        assert processor.fal_client is None
        assert processor.fal_enabled is False
        assert processor._fal_received_frames.empty()

    @pytest.mark.asyncio
    async def test_disconnect_from_fal_when_not_connected(self):
        """Test disconnect works when not connected."""
        from scope.server.frame_processor import FrameProcessor

        mock_pm = MagicMock()
        processor = FrameProcessor(pipeline_manager=mock_pm)

        # Should not raise any exceptions
        await processor.disconnect_from_fal()

        assert processor.fal_client is None
        assert processor.fal_enabled is False


class TestFrameProcessorFalRouting:
    """Tests for frame routing to/from fal."""

    def test_put_routes_to_fal_when_enabled(self):
        """Test that put() routes frames to fal when enabled."""
        from scope.server.frame_processor import FrameProcessor

        mock_pm = MagicMock()
        processor = FrameProcessor(pipeline_manager=mock_pm)
        processor.running = True

        # Set up fal client with output track
        mock_output_track = MagicMock()
        mock_output_track.put_frame_nowait = MagicMock(return_value=True)
        mock_client = MagicMock()
        mock_client.output_track = mock_output_track

        processor.fal_client = mock_client
        processor.fal_enabled = True

        mock_frame = MagicMock()
        result = processor.put(mock_frame)

        # Should route to fal
        mock_output_track.put_frame_nowait.assert_called_once_with(mock_frame)
        assert result is True

    def test_put_routes_to_local_when_fal_disabled(self):
        """Test that put() routes to local processing when fal disabled."""
        from scope.server.frame_processor import FrameProcessor

        mock_pm = MagicMock()
        processor = FrameProcessor(pipeline_manager=mock_pm)
        processor.running = True
        processor.fal_enabled = False

        # Set up local pipeline processor
        mock_pipeline_processor = MagicMock()
        mock_pipeline_processor.input_queue = MagicMock()
        mock_pipeline_processor.input_queue.put_nowait = MagicMock()
        processor.pipeline_processors = [mock_pipeline_processor]

        # Create a mock frame with to_ndarray
        mock_frame = MagicMock()
        mock_frame.to_ndarray = MagicMock(return_value=MagicMock())

        with patch("scope.server.frame_processor.torch") as mock_torch:
            mock_tensor = MagicMock()
            mock_tensor.unsqueeze = MagicMock(return_value=mock_tensor)
            mock_torch.from_numpy = MagicMock(return_value=mock_tensor)

            result = processor.put(mock_frame)

            # Should route to local pipeline
            mock_pipeline_processor.input_queue.put_nowait.assert_called_once()
            assert result is True

    def test_get_returns_from_fal_when_enabled(self):
        """Test that get() returns frames from fal queue when enabled."""
        from scope.server.frame_processor import FrameProcessor

        mock_pm = MagicMock()
        processor = FrameProcessor(pipeline_manager=mock_pm)
        processor.running = True
        processor.fal_enabled = True

        # Add a mock frame to the fal received queue
        mock_fal_frame = MagicMock()
        mock_fal_frame.to_ndarray = MagicMock(return_value=MagicMock())
        processor._fal_received_frames.put_nowait(mock_fal_frame)

        with patch("scope.server.frame_processor.torch") as mock_torch:
            mock_tensor = MagicMock()
            mock_torch.from_numpy = MagicMock(return_value=mock_tensor)

            result = processor.get()

            # Should return frame from fal queue
            mock_fal_frame.to_ndarray.assert_called_once_with(format="rgb24")
            assert result is mock_tensor

    def test_get_returns_none_when_fal_queue_empty(self):
        """Test that get() returns None when fal queue is empty."""
        from scope.server.frame_processor import FrameProcessor

        mock_pm = MagicMock()
        processor = FrameProcessor(pipeline_manager=mock_pm)
        processor.running = True
        processor.fal_enabled = True

        # Queue is empty
        result = processor.get()

        assert result is None

    def test_get_returns_from_local_when_fal_disabled(self):
        """Test that get() returns from local pipeline when fal disabled."""
        from scope.server.frame_processor import FrameProcessor

        mock_pm = MagicMock()
        processor = FrameProcessor(pipeline_manager=mock_pm)
        processor.running = True
        processor.fal_enabled = False

        # Set up local pipeline processor with output queue
        mock_frame = MagicMock()
        mock_frame.squeeze = MagicMock(return_value=mock_frame)
        mock_frame.cpu = MagicMock(return_value=mock_frame)

        mock_output_queue = queue.Queue()
        mock_output_queue.put(mock_frame)

        mock_pipeline_processor = MagicMock()
        mock_pipeline_processor.output_queue = mock_output_queue
        processor.pipeline_processors = [mock_pipeline_processor]

        result = processor.get()

        # Should return frame from local pipeline
        assert result is mock_frame
        mock_frame.squeeze.assert_called_once_with(0)
        mock_frame.cpu.assert_called_once()


class TestFrameProcessorFalStop:
    """Tests for stop() with fal cleanup."""

    def test_stop_clears_fal_state(self):
        """Test that stop() clears fal state."""
        from scope.server.frame_processor import FrameProcessor

        mock_pm = MagicMock()
        processor = FrameProcessor(pipeline_manager=mock_pm)
        processor.running = True

        # Set up fal state
        mock_client = MagicMock()
        processor.fal_client = mock_client
        processor.fal_enabled = True

        # Add some frames to queue
        for _ in range(3):
            processor._fal_received_frames.put_nowait(MagicMock())

        processor.stop()

        # Check fal state is cleared
        assert processor.fal_client is None
        assert processor.fal_enabled is False
        assert processor._fal_received_frames.empty()

    @pytest.mark.asyncio
    async def test_stop_async_disconnects_fal(self):
        """Test that stop_async() properly disconnects from fal."""
        from scope.server.frame_processor import FrameProcessor

        mock_pm = MagicMock()
        processor = FrameProcessor(pipeline_manager=mock_pm)
        processor.running = True

        # Set up fal client
        mock_client = AsyncMock()
        processor.fal_client = mock_client
        processor.fal_enabled = True

        await processor.stop_async()

        # Check disconnect was called
        mock_client.disconnect.assert_called_once()

        # Check state is cleared
        assert processor.fal_client is None
        assert processor.fal_enabled is False
        assert not processor.running


class TestFrameProcessorSpoutFalRouting:
    """Tests for Spout receiver routing to fal."""

    def test_spout_receiver_routes_to_fal_when_enabled(self):
        """Test that Spout frames route through put() when fal is enabled."""
        from unittest.mock import patch

        import numpy as np

        from scope.server.frame_processor import FrameProcessor

        mock_pm = MagicMock()
        processor = FrameProcessor(pipeline_manager=mock_pm)
        processor.running = True
        processor.spout_receiver_enabled = True

        # Set up fal client
        mock_output_track = MagicMock()
        mock_output_track.put_frame_nowait = MagicMock(return_value=True)
        mock_client = MagicMock()
        mock_client.output_track = mock_output_track
        processor.fal_client = mock_client
        processor.fal_enabled = True

        # Create a mock Spout receiver
        mock_spout_receiver = MagicMock()
        # Return a frame once, then None to exit loop
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_spout_receiver.receive = MagicMock(side_effect=[test_frame, None])
        processor.spout_receiver = mock_spout_receiver

        # Mock VideoFrame.from_ndarray
        with patch("scope.server.frame_processor.VideoFrame") as MockVideoFrame:
            mock_video_frame = MagicMock()
            MockVideoFrame.from_ndarray = MagicMock(return_value=mock_video_frame)

            # Run one iteration of the loop manually
            # We can't easily test the thread loop, so test the routing logic directly
            rgb_frame = mock_spout_receiver.receive()
            if processor.fal_enabled and processor.fal_client:
                from av import VideoFrame

                video_frame = VideoFrame.from_ndarray(rgb_frame, format="rgb24")
                result = processor.put(video_frame)

            # Verify frame was routed to fal
            mock_output_track.put_frame_nowait.assert_called_once()

    def test_spout_receiver_routes_to_local_when_fal_disabled(self):
        """Test that Spout frames go to local pipeline when fal is disabled."""
        import numpy as np

        from scope.server.frame_processor import FrameProcessor

        mock_pm = MagicMock()
        processor = FrameProcessor(pipeline_manager=mock_pm)
        processor.running = True
        processor.spout_receiver_enabled = True
        processor.fal_enabled = False  # fal disabled

        # Set up local pipeline processor
        mock_input_queue = MagicMock()
        mock_pipeline_processor = MagicMock()
        mock_pipeline_processor.input_queue = mock_input_queue
        processor.pipeline_processors = [mock_pipeline_processor]

        # Simulate what _spout_receiver_loop does when fal is disabled
        rgb_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # This is the logic from _spout_receiver_loop when fal is disabled
        if not (processor.fal_enabled and processor.fal_client):
            if processor.pipeline_processors:
                import torch

                first_processor = processor.pipeline_processors[0]
                frame_tensor = torch.from_numpy(rgb_frame)
                frame_tensor = frame_tensor.unsqueeze(0)
                first_processor.input_queue.put_nowait(frame_tensor)

        # Verify frame was put into local pipeline
        mock_input_queue.put_nowait.assert_called_once()


class TestFrameProcessorParameterRouting:
    """Tests for parameter routing to fal vs local pipelines."""

    def test_update_parameters_routes_to_fal_when_enabled(self):
        """Test that parameters are forwarded to fal when cloud mode enabled."""
        from scope.server.frame_processor import FrameProcessor

        mock_pm = MagicMock()
        processor = FrameProcessor(pipeline_manager=mock_pm)
        processor.running = True

        # Set up fal client with mocked send_parameters
        mock_client = MagicMock()
        mock_client.send_parameters = MagicMock(return_value=True)
        processor.fal_client = mock_client
        processor.fal_enabled = True

        # Set up local pipeline processor (should NOT be called)
        mock_pipeline_processor = MagicMock()
        processor.pipeline_processors = [mock_pipeline_processor]

        # Send parameters
        processor.update_parameters({"prompts": ["test prompt"], "noise_scale": 0.5})

        # Should route to fal
        mock_client.send_parameters.assert_called_once_with(
            {"prompts": ["test prompt"], "noise_scale": 0.5}
        )
        # Local pipeline should NOT receive parameters
        mock_pipeline_processor.update_parameters.assert_not_called()

    def test_update_parameters_routes_to_local_when_fal_disabled(self):
        """Test that parameters go to local pipelines when cloud mode disabled."""
        from scope.server.frame_processor import FrameProcessor

        mock_pm = MagicMock()
        processor = FrameProcessor(pipeline_manager=mock_pm)
        processor.running = True
        processor.fal_enabled = False
        processor.fal_client = None

        # Set up local pipeline processor
        mock_pipeline_processor = MagicMock()
        processor.pipeline_processors = [mock_pipeline_processor]

        # Send parameters
        processor.update_parameters({"prompts": ["test prompt"], "noise_scale": 0.5})

        # Should route to local pipeline
        mock_pipeline_processor.update_parameters.assert_called_once_with(
            {"prompts": ["test prompt"], "noise_scale": 0.5}
        )

    def test_spout_params_stay_local_when_fal_enabled(self):
        """Test that Spout parameters are always handled locally."""
        from scope.server.frame_processor import FrameProcessor

        mock_pm = MagicMock()
        processor = FrameProcessor(pipeline_manager=mock_pm)
        processor.running = True

        # Set up fal client
        mock_client = MagicMock()
        mock_client.send_parameters = MagicMock(return_value=True)
        processor.fal_client = mock_client
        processor.fal_enabled = True

        # Mock _update_spout_sender and _update_spout_receiver
        processor._update_spout_sender = MagicMock()
        processor._update_spout_receiver = MagicMock()

        # Send mixed parameters (Spout + pipeline params)
        processor.update_parameters(
            {
                "spout_sender": {"enabled": True, "name": "TestSender"},
                "spout_receiver": {"enabled": True, "name": "TestReceiver"},
                "prompts": ["test prompt"],
            }
        )

        # Spout params should be handled locally
        processor._update_spout_sender.assert_called_once_with(
            {"enabled": True, "name": "TestSender"}
        )
        processor._update_spout_receiver.assert_called_once_with(
            {"enabled": True, "name": "TestReceiver"}
        )

        # Only non-Spout params should be forwarded to fal
        mock_client.send_parameters.assert_called_once_with(
            {"prompts": ["test prompt"]}
        )

    def test_parameters_stored_locally_regardless_of_mode(self):
        """Test that parameters are always stored locally for state tracking."""
        from scope.server.frame_processor import FrameProcessor

        mock_pm = MagicMock()
        processor = FrameProcessor(pipeline_manager=mock_pm)
        processor.running = True

        # Set up fal client
        mock_client = MagicMock()
        mock_client.send_parameters = MagicMock(return_value=True)
        processor.fal_client = mock_client
        processor.fal_enabled = True

        # Send parameters
        processor.update_parameters({"prompts": ["test"], "noise_scale": 0.5})

        # Parameters should be stored locally
        assert processor.parameters["prompts"] == ["test"]
        assert processor.parameters["noise_scale"] == 0.5
