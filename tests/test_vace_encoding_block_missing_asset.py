"""
Tests for VaceEncodingBlock stale/missing asset path handling.

Regression test for #689: VaceEncodingBlock crashes on fal.ai when a saved
workflow references an asset path from a previous ephemeral worker session.
The block must log an ERROR exactly once per missing path and then skip
VACE conditioning gracefully instead of crashing on every chunk.
"""

import logging
from unittest.mock import MagicMock, patch

import pytest


class TestValidateImagePaths:
    """Unit tests for VaceEncodingBlock._validate_image_paths."""

    def _make_block(self):
        from scope.core.pipelines.wan2_1.vace.blocks.vace_encoding import (
            VaceEncodingBlock,
        )

        return VaceEncodingBlock()

    def test_returns_true_for_existing_paths(self, tmp_path):
        """Returns True when all paths exist."""
        img = tmp_path / "img.jpg"
        img.write_bytes(b"fake")

        block = self._make_block()
        assert block._validate_image_paths([str(img)]) is True

    def test_returns_true_for_none_paths(self):
        """None entries are skipped — returns True."""
        block = self._make_block()
        assert block._validate_image_paths([None, None]) is True

    def test_returns_false_for_missing_path(self, tmp_path):
        """Returns False when a path does not exist."""
        block = self._make_block()
        missing = str(tmp_path / "nonexistent.jpg")
        assert block._validate_image_paths([missing]) is False

    def test_logs_error_on_first_missing_occurrence(self, tmp_path, caplog):
        """Logs ERROR exactly once when a path is first detected as missing."""
        block = self._make_block()
        missing = str(tmp_path / "stale_screenshot.jpg")

        with caplog.at_level(logging.ERROR, logger="scope"):
            block._validate_image_paths([missing])

        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(errors) == 1
        assert missing in errors[0].message

    def test_suppresses_error_on_subsequent_missing_occurrences(self, tmp_path, caplog):
        """Logs ERROR only once for the same missing path (no per-chunk log storm)."""
        block = self._make_block()
        missing = str(tmp_path / "stale_screenshot.jpg")

        with caplog.at_level(logging.ERROR, logger="scope"):
            for _ in range(150):  # simulate 150 chunks
                block._validate_image_paths([missing])

        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(errors) == 1, (
            f"Expected 1 ERROR log, got {len(errors)} — log storm not suppressed"
        )

    def test_mixed_valid_and_missing_paths(self, tmp_path):
        """Returns False when at least one path is missing."""
        existing = tmp_path / "good.jpg"
        existing.write_bytes(b"fake")
        missing = str(tmp_path / "stale.jpg")

        block = self._make_block()
        assert block._validate_image_paths([str(existing), missing]) is False

    def test_multiple_distinct_missing_paths_each_logged_once(self, tmp_path, caplog):
        """Each unique missing path is logged once."""
        block = self._make_block()
        missing_a = str(tmp_path / "a.jpg")
        missing_b = str(tmp_path / "b.jpg")

        with caplog.at_level(logging.ERROR, logger="scope"):
            for _ in range(10):
                block._validate_image_paths([missing_a, missing_b])

        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(errors) == 2
        messages = " ".join(e.message for e in errors)
        assert missing_a in messages
        assert missing_b in messages


class TestCallWithMissingAssets:
    """Integration-style tests for VaceEncodingBlock.__call__ with missing assets."""

    def _make_block(self):
        from scope.core.pipelines.wan2_1.vace.blocks.vace_encoding import (
            VaceEncodingBlock,
        )

        return VaceEncodingBlock()

    def _make_state(self, **kwargs):
        """Build a minimal PipelineState-like mock."""
        block_state = MagicMock()
        block_state.vace_ref_images = kwargs.get("vace_ref_images", None)
        block_state.vace_input_frames = kwargs.get("vace_input_frames", None)
        block_state.first_frame_image = kwargs.get("first_frame_image", None)
        block_state.last_frame_image = kwargs.get("last_frame_image", None)
        block_state.current_start_frame = kwargs.get("current_start_frame", 0)

        state = MagicMock()
        return block_state, state

    def test_skips_encoding_when_ref_image_missing(self, tmp_path, caplog):
        """
        When vace_ref_images contains a stale path, __call__ should skip encoding
        and set vace_context=None / vace_ref_images=None instead of crashing.
        """
        from scope.core.pipelines.wan2_1.vace.blocks.vace_encoding import (
            VaceEncodingBlock,
        )

        block = VaceEncodingBlock()
        missing = str(tmp_path / "stale.jpg")
        block_state, state = self._make_state(vace_ref_images=[missing])

        components = MagicMock()

        with patch.object(block, "get_block_state", return_value=block_state):
            with patch.object(block, "set_block_state") as mock_set:
                with caplog.at_level(logging.ERROR, logger="scope"):
                    result_components, result_state = block(components, state)

        # Should have returned early without crashing
        assert block_state.vace_context is None
        assert block_state.vace_ref_images is None
        mock_set.assert_called_once()

        # Exactly one ERROR for this path
        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(errors) == 1
        assert missing in errors[0].message

    def test_no_log_storm_across_chunks_with_missing_ref(self, tmp_path, caplog):
        """
        Simulates 50 chunks with a stale ref image. Should produce exactly 1 ERROR.
        """
        from scope.core.pipelines.wan2_1.vace.blocks.vace_encoding import (
            VaceEncodingBlock,
        )

        block = VaceEncodingBlock()
        missing = str(tmp_path / "stale.jpg")
        components = MagicMock()

        with caplog.at_level(logging.ERROR, logger="scope"):
            for _ in range(50):
                block_state, state = self._make_state(vace_ref_images=[missing])
                with patch.object(block, "get_block_state", return_value=block_state):
                    with patch.object(block, "set_block_state"):
                        block(components, state)

        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(errors) == 1, (
            f"Expected 1 ERROR across 50 chunks, got {len(errors)}"
        )
