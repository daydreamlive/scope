"""Tests for fal tracks module."""

import asyncio
import fractions
from unittest.mock import MagicMock

import pytest

from scope.server.fal_tracks import FalInputTrack, FalOutputTrack


class TestFalOutputTrack:
    """Tests for FalOutputTrack class."""

    def test_initialization(self):
        """Test FalOutputTrack initializes with correct defaults."""
        track = FalOutputTrack()

        assert track.kind == "video"
        assert track.target_fps == 30
        assert track._frame_count == 0
        assert track.frame_queue.maxsize == 30

    def test_initialization_custom_fps(self):
        """Test FalOutputTrack with custom FPS."""
        track = FalOutputTrack(target_fps=60)

        assert track.target_fps == 60
        assert track.frame_queue.maxsize == 30

    @pytest.mark.asyncio
    async def test_recv_returns_frame_with_pts(self):
        """Test recv() returns frame with correct pts and time_base."""
        track = FalOutputTrack(target_fps=30)

        # Create mock frame
        mock_frame = MagicMock()
        mock_frame.pts = None
        mock_frame.time_base = None

        # Put frame in queue
        await track.frame_queue.put(mock_frame)

        # Receive frame
        result = await track.recv()

        assert result is mock_frame
        assert result.pts == 1
        assert result.time_base == fractions.Fraction(1, 30)

    @pytest.mark.asyncio
    async def test_recv_increments_frame_count(self):
        """Test recv() increments frame count with each call."""
        track = FalOutputTrack()

        for i in range(3):
            mock_frame = MagicMock()
            await track.frame_queue.put(mock_frame)
            result = await track.recv()
            assert result.pts == i + 1

        assert track._frame_count == 3

    @pytest.mark.asyncio
    async def test_put_frame_success(self):
        """Test put_frame() successfully queues frame."""
        track = FalOutputTrack()
        mock_frame = MagicMock()

        result = await track.put_frame(mock_frame)

        assert result is True
        assert track.frame_queue.qsize() == 1

    @pytest.mark.asyncio
    async def test_put_frame_drops_oldest_when_full(self):
        """Test put_frame() drops oldest frame when queue is full."""
        track = FalOutputTrack()
        track.frame_queue = asyncio.Queue(maxsize=2)

        frame1 = MagicMock(name="frame1")
        frame2 = MagicMock(name="frame2")
        frame3 = MagicMock(name="frame3")

        await track.put_frame(frame1)
        await track.put_frame(frame2)
        # Queue is now full, frame3 should replace frame1
        result = await track.put_frame(frame3)

        assert result is True
        assert track.frame_queue.qsize() == 2

        # First frame out should be frame2 (frame1 was dropped)
        out1 = await track.frame_queue.get()
        out2 = await track.frame_queue.get()
        assert out1 is frame2
        assert out2 is frame3

    def test_put_frame_nowait_success(self):
        """Test put_frame_nowait() successfully queues frame."""
        track = FalOutputTrack()
        mock_frame = MagicMock()

        result = track.put_frame_nowait(mock_frame)

        assert result is True
        assert track.frame_queue.qsize() == 1

    def test_put_frame_nowait_returns_false_when_full(self):
        """Test put_frame_nowait() returns False when queue is full."""
        track = FalOutputTrack()
        track.frame_queue = asyncio.Queue(maxsize=1)

        frame1 = MagicMock()
        frame2 = MagicMock()

        track.put_frame_nowait(frame1)
        result = track.put_frame_nowait(frame2)

        assert result is False
        assert track.frame_queue.qsize() == 1

    def test_put_frame_sync_calls_nowait(self):
        """Test put_frame_sync() uses put_frame_nowait()."""
        track = FalOutputTrack()
        mock_frame = MagicMock()

        result = track.put_frame_sync(mock_frame)

        assert result is True
        assert track.frame_queue.qsize() == 1


class TestFalInputTrack:
    """Tests for FalInputTrack class."""

    def test_initialization(self):
        """Test FalInputTrack initializes correctly."""
        mock_source = MagicMock()
        track = FalInputTrack(mock_source)

        assert track.kind == "video"
        assert track.source_track is mock_source
        assert track.frame_queue.maxsize == 30
        assert track._consume_task is None

    def test_start_consuming_creates_task(self):
        """Test start_consuming() creates asyncio task."""
        mock_source = MagicMock()
        track = FalInputTrack(mock_source)

        # Mock asyncio.create_task
        with pytest.MonkeyPatch.context() as mp:
            mock_task = MagicMock()
            mp.setattr(asyncio, "create_task", lambda coro: mock_task)
            track.start_consuming()

        assert track._consume_task is mock_task

    @pytest.mark.asyncio
    async def test_recv_returns_frame_from_queue(self):
        """Test recv() returns frame from queue."""
        mock_source = MagicMock()
        track = FalInputTrack(mock_source)

        mock_frame = MagicMock()
        await track.frame_queue.put(mock_frame)

        result = await track.recv()

        assert result is mock_frame

    def test_get_frame_nowait_returns_frame(self):
        """Test get_frame_nowait() returns frame when available."""
        mock_source = MagicMock()
        track = FalInputTrack(mock_source)

        mock_frame = MagicMock()
        track.frame_queue.put_nowait(mock_frame)

        result = track.get_frame_nowait()

        assert result is mock_frame

    def test_get_frame_nowait_returns_none_when_empty(self):
        """Test get_frame_nowait() returns None when queue is empty."""
        mock_source = MagicMock()
        track = FalInputTrack(mock_source)

        result = track.get_frame_nowait()

        assert result is None

    @pytest.mark.asyncio
    async def test_stop_cancels_consume_task(self):
        """Test stop() cancels the consume task."""
        mock_source = MagicMock()
        track = FalInputTrack(mock_source)

        # Create a real task that can be cancelled
        async def dummy_loop():
            while True:
                await asyncio.sleep(1)

        task = asyncio.create_task(dummy_loop())
        track._consume_task = task

        await track.stop()

        assert task.cancelled()

    @pytest.mark.asyncio
    async def test_stop_handles_no_task(self):
        """Test stop() handles case when no task exists."""
        mock_source = MagicMock()
        track = FalInputTrack(mock_source)

        # Should not raise
        await track.stop()

        assert track._consume_task is None

    @pytest.mark.asyncio
    async def test_consume_loop_queues_frames(self):
        """Test _consume_loop() receives and queues frames."""
        mock_source = MagicMock()
        track = FalInputTrack(mock_source)

        frames = [MagicMock(name=f"frame{i}") for i in range(3)]
        frame_iter = iter(frames)

        async def mock_recv():
            try:
                return next(frame_iter)
            except StopIteration:
                raise Exception("End of frames")

        mock_source.recv = mock_recv

        # Start consuming
        track.start_consuming()

        # Wait for frames to be consumed
        await asyncio.sleep(0.1)

        # Stop consuming
        await track.stop()

        # Check frames were queued (may not get all due to timing)
        assert track.frame_queue.qsize() > 0

    @pytest.mark.asyncio
    async def test_consume_loop_drops_oldest_when_full(self):
        """Test _consume_loop() drops oldest frame when queue is full."""
        mock_source = MagicMock()
        track = FalInputTrack(mock_source)
        track.frame_queue = asyncio.Queue(maxsize=2)

        frames = [MagicMock(name=f"frame{i}") for i in range(5)]
        frame_index = 0

        async def mock_recv():
            nonlocal frame_index
            if frame_index < len(frames):
                frame = frames[frame_index]
                frame_index += 1
                return frame
            raise Exception("End of frames")

        mock_source.recv = mock_recv

        track.start_consuming()
        await asyncio.sleep(0.1)
        await track.stop()

        # Queue should have at most 2 frames (maxsize)
        assert track.frame_queue.qsize() <= 2
