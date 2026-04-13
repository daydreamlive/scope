"""Tests for validate_resolution / snap_to_multiple in pipelines/utils.py."""
import pytest

from scope.core.pipelines.utils import snap_to_multiple, validate_resolution


class TestSnapToMultiple:
    def test_already_aligned(self):
        assert snap_to_multiple(672, 16) == 672

    def test_rounds_down(self):
        assert snap_to_multiple(674, 16) == 672
        assert snap_to_multiple(389, 16) == 384

    def test_smaller_than_multiple(self):
        assert snap_to_multiple(10, 16) == 0


class TestValidateResolution:
    # --- default snap=False behaviour (raises on invalid) ---

    def test_valid_resolution_returns_unchanged(self):
        h, w = validate_resolution(height=384, width=672, scale_factor=16)
        assert (h, w) == (384, 672)

    def test_invalid_raises_value_error(self):
        with pytest.raises(ValueError, match="Invalid resolution"):
            validate_resolution(height=389, width=674, scale_factor=16)

    def test_error_message_contains_suggestion(self):
        with pytest.raises(ValueError, match="672×384"):
            validate_resolution(height=389, width=674, scale_factor=16)

    # --- snap=True behaviour (rounds down, no exception) ---

    def test_snap_invalid_resolution(self):
        h, w = validate_resolution(height=389, width=674, scale_factor=16, snap=True)
        assert (h, w) == (384, 672)

    def test_snap_valid_resolution_unchanged(self):
        h, w = validate_resolution(height=320, width=576, scale_factor=16, snap=True)
        assert (h, w) == (320, 576)

    def test_snap_only_height_unaligned(self):
        h, w = validate_resolution(height=385, width=576, scale_factor=16, snap=True)
        assert (h, w) == (384, 576)

    def test_snap_only_width_unaligned(self):
        h, w = validate_resolution(height=384, width=577, scale_factor=16, snap=True)
        assert (h, w) == (384, 576)

    def test_snap_logs_warning(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING, logger="scope.core.pipelines.utils"):
            validate_resolution(height=389, width=674, scale_factor=16, snap=True)
        assert any("Snapping resolution" in r.message for r in caplog.records)
