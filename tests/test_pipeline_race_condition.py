"""Tests for issue #936 — transient race condition when a plugin isn't yet installed.

Verifies that:
1. ``_load_pipeline_implementation`` raises ``PipelineNotYetRegisteredException``
   (not a plain ``ValueError``) when the pipeline ID is unknown and not a builtin.
2. ``_load_pipeline_by_id_sync`` returns ``False`` **without** setting the
   pipeline status to ``ERROR`` when it catches that exception — it must leave
   the status as ``NOT_LOADED`` so the load can be retried once the plugin
   finishes installing.
"""

from unittest.mock import MagicMock, patch

import pytest

from scope.server.pipeline_manager import (
    PipelineManager,
    PipelineNotYetRegisteredException,
    PipelineStatus,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_manager() -> PipelineManager:
    return PipelineManager()


# ---------------------------------------------------------------------------
# Unit tests for _load_pipeline_implementation
# ---------------------------------------------------------------------------


class TestLoadPipelineImplementationUnknownId:
    """_load_pipeline_implementation must raise PipelineNotYetRegisteredException
    for a pipeline ID that is neither a builtin nor in the registry."""

    def test_raises_for_unregistered_non_builtin(self):
        manager = _make_manager()

        # Patch PipelineRegistry.get to return None (plugin not installed yet)
        with patch(
            "scope.core.pipelines.registry.PipelineRegistry.get", return_value=None
        ):
            with pytest.raises(PipelineNotYetRegisteredException) as exc_info:
                manager._load_pipeline_implementation("yolo_mask")

        assert "yolo_mask" in str(exc_info.value)

    def test_exception_is_subclass_of_value_error(self):
        """PipelineNotYetRegisteredException must be a ValueError subclass."""
        assert issubclass(PipelineNotYetRegisteredException, ValueError)

    def test_does_not_raise_for_builtin(self):
        """Built-in pipeline IDs should NOT raise PipelineNotYetRegisteredException
        — they fall through to their own initialisation logic (which may succeed
        or fail for unrelated reasons, but must never raise the transient
        plugin-not-yet-registered exception)."""
        manager = _make_manager()

        # "passthrough" is a builtin; a missing registry entry is fine for it.
        # The implementation either returns successfully or raises something that
        # is NOT PipelineNotYetRegisteredException.
        with patch(
            "scope.core.pipelines.registry.PipelineRegistry.get", return_value=None
        ):
            try:
                manager._load_pipeline_implementation("passthrough")
                # Completed without exception — that's also fine.
            except PipelineNotYetRegisteredException:
                pytest.fail(
                    "Built-in pipelines should never raise PipelineNotYetRegisteredException"
                )
            except Exception:
                # Any other exception (ImportError, etc.) is acceptable.
                pass


# ---------------------------------------------------------------------------
# Integration tests for _load_pipeline_by_id_sync
# ---------------------------------------------------------------------------


class TestLoadPipelineByIdSyncRaceCondition:
    """_load_pipeline_by_id_sync must handle PipelineNotYetRegisteredException
    gracefully — returning False and leaving status as NOT_LOADED (not ERROR)."""

    def _sync_load_with_not_yet_registered(
        self, pipeline_id: str = "yolo_mask"
    ) -> tuple[PipelineManager, bool]:
        """Run _load_pipeline_by_id_sync where _load_pipeline_implementation
        raises PipelineNotYetRegisteredException, and return (manager, result)."""
        manager = _make_manager()

        def fake_impl(pid, load_params=None, stage_callback=None):
            raise PipelineNotYetRegisteredException(
                f"Invalid pipeline ID: {pid}. Plugin may not be installed yet."
            )

        with patch.object(manager, "_load_pipeline_implementation", side_effect=fake_impl):
            result = manager._load_pipeline_by_id_sync(pipeline_id)

        return manager, result

    def test_returns_false(self):
        _, result = self._sync_load_with_not_yet_registered()
        assert result is False

    def test_status_is_not_loaded_not_error(self):
        """Status must be NOT_LOADED so the frontend never sees ERROR."""
        manager, _ = self._sync_load_with_not_yet_registered()
        status = manager._pipeline_statuses.get("yolo_mask")
        assert status == PipelineStatus.NOT_LOADED, (
            f"Expected NOT_LOADED, got {status!r}"
        )

    def test_pipeline_not_stored(self):
        """No pipeline instance should be stored after a transient failure."""
        manager, _ = self._sync_load_with_not_yet_registered()
        assert "yolo_mask" not in manager._pipelines

    def test_load_event_is_set(self):
        """The load event must be signalled so any waiting threads are unblocked."""
        manager = _make_manager()

        def fake_impl(pid, load_params=None, stage_callback=None):
            raise PipelineNotYetRegisteredException(
                f"Invalid pipeline ID: {pid}. Plugin may not be installed yet."
            )

        with patch.object(manager, "_load_pipeline_implementation", side_effect=fake_impl):
            manager._load_pipeline_by_id_sync("yolo_mask")

        event = manager._load_events.get("yolo_mask")
        # The event should have been set (or cleaned up — either is acceptable,
        # but it must not be left unset and blocking).
        if event is not None:
            assert event.is_set(), "Load event must be set after transient failure"

    def test_no_error_log_emitted(self, caplog):
        """No ERROR-level log message should be emitted for a transient failure."""
        import logging

        manager = _make_manager()

        def fake_impl(pid, load_params=None, stage_callback=None):
            raise PipelineNotYetRegisteredException(
                f"Invalid pipeline ID: {pid}. Plugin may not be installed yet."
            )

        with patch.object(manager, "_load_pipeline_implementation", side_effect=fake_impl):
            with caplog.at_level(logging.WARNING, logger="scope.server.pipeline_manager"):
                manager._load_pipeline_by_id_sync("yolo_mask")

        error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert not error_records, (
            f"Unexpected ERROR log(s): {[r.message for r in error_records]}"
        )

    def test_warning_log_emitted(self, caplog):
        """A WARNING-level log should be emitted to explain the transient state."""
        import logging

        manager = _make_manager()

        def fake_impl(pid, load_params=None, stage_callback=None):
            raise PipelineNotYetRegisteredException(
                f"Invalid pipeline ID: yolo_mask. Plugin may not be installed yet."
            )

        with patch.object(manager, "_load_pipeline_implementation", side_effect=fake_impl):
            with caplog.at_level(logging.WARNING, logger="scope.server.pipeline_manager"):
                manager._load_pipeline_by_id_sync("yolo_mask")

        warn_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warn_records, "Expected at least one WARNING log for transient failure"
        combined = " ".join(r.message for r in warn_records)
        assert "plugin" in combined.lower() or "installing" in combined.lower(), (
            "Warning should mention plugin/installing"
        )
