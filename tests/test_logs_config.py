"""Unit tests for lib.logs_config module."""

import logging
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
from freezegun import freeze_time

from scope.server.logs_config import (
    LOGS_DIR_ENV_VAR,
    ResilientRotatingFileHandler,
    cleanup_old_logs,
    get_most_recent_log_file,
)


@pytest.fixture
def temp_logs_dir(tmp_path, monkeypatch):
    """Create a temporary logs directory and configure it via environment variable."""
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()

    # Set environment variable to point to our temp directory
    monkeypatch.setenv(LOGS_DIR_ENV_VAR, str(logs_dir))

    yield logs_dir


class TestGetMostRecentLogFile:
    """Tests for get_most_recent_log_file function."""

    def test_returns_none_when_directory_does_not_exist(self, tmp_path, monkeypatch):
        """Should return None when logs directory doesn't exist."""
        nonexistent_dir = tmp_path / "nonexistent"

        # Set environment variable to point to non-existent directory
        monkeypatch.setenv(LOGS_DIR_ENV_VAR, str(nonexistent_dir))

        result = get_most_recent_log_file()

        assert result is None

    def test_returns_none_when_directory_is_empty(self, temp_logs_dir):
        """Should return None when logs directory exists but has no log files."""
        result = get_most_recent_log_file()

        assert result is None

    def test_returns_none_when_only_non_log_files_exist(self, temp_logs_dir):
        """Should return None when directory has files but no .log files."""
        # Create non-log files
        (temp_logs_dir / "readme.txt").write_text("Not a log")
        (temp_logs_dir / "data.json").write_text("{}")

        result = get_most_recent_log_file()

        assert result is None

    def test_returns_only_log_file_when_single_file_exists(self, temp_logs_dir):
        """Should return the only log file when only one exists."""
        log_file = temp_logs_dir / "scope-logs-2025-11-04-10-00-00.log"
        log_file.write_text("Log content")

        result = get_most_recent_log_file()

        assert result == log_file

    def test_returns_most_recent_log_file_by_filename(self, temp_logs_dir):
        """Should return the most recent log file based on filename timestamp."""
        # Create log files with different timestamps (order matters for sorting)
        old_log = temp_logs_dir / "scope-logs-2025-11-04-10-00-00.log"
        mid_log = temp_logs_dir / "scope-logs-2025-11-04-14-30-00.log"
        new_log = temp_logs_dir / "scope-logs-2025-11-04-16-45-30.log"

        old_log.write_text("Old log")
        mid_log.write_text("Mid log")
        new_log.write_text("New log")

        result = get_most_recent_log_file()

        assert result == new_log

    def test_ignores_rotated_log_files(self, temp_logs_dir):
        """Should ignore rotated log files (.log.1, .log.2, etc) and only consider base .log files."""
        # Create base log file
        base_log = temp_logs_dir / "scope-logs-2025-11-04-10-00-00.log"
        base_log.write_text("Base log")

        # Create rotated log files (should be ignored)
        rotated1 = temp_logs_dir / "scope-logs-2025-11-04-10-00-00.log.1"
        rotated2 = temp_logs_dir / "scope-logs-2025-11-04-10-00-00.log.2"
        rotated1.write_text("Rotated 1")
        rotated2.write_text("Rotated 2")

        result = get_most_recent_log_file()

        assert result == base_log

    def test_handles_different_date_formats_correctly(self, temp_logs_dir):
        """Should correctly sort files with different date/time components."""
        # Create files that test year, month, day, hour, minute, second ordering
        files = [
            "scope-logs-2024-12-31-23-59-59.log",  # Old year
            "scope-logs-2025-01-01-00-00-00.log",  # New year, start
            "scope-logs-2025-11-03-23-59-59.log",  # Day before
            "scope-logs-2025-11-04-00-00-00.log",  # Target day start
            "scope-logs-2025-11-04-16-45-30.log",  # Target day afternoon
        ]

        for filename in files:
            (temp_logs_dir / filename).write_text("Log")

        result = get_most_recent_log_file()

        assert result == temp_logs_dir / "scope-logs-2025-11-04-16-45-30.log"


class TestCleanupOldLogs:
    """Tests for cleanup_old_logs function."""

    def test_does_nothing_when_directory_does_not_exist(
        self, tmp_path, monkeypatch, caplog
    ):
        """Should not raise error when logs directory doesn't exist."""
        nonexistent_dir = tmp_path / "nonexistent"

        # Set environment variable to point to non-existent directory
        monkeypatch.setenv(LOGS_DIR_ENV_VAR, str(nonexistent_dir))

        cleanup_old_logs(max_age_days=1)

        # Should complete without errors
        assert "error" not in caplog.text.lower()

    @freeze_time("2025-11-04 16:00:00")
    def test_keeps_files_newer_than_cutoff(self, temp_logs_dir):
        """Should keep files that are newer than the cutoff date."""
        # Create a file from 12 hours ago (should be kept with default 1 day cutoff)
        recent_file = temp_logs_dir / "scope-logs-2025-11-04-04-00-00.log"
        recent_file.write_text("Recent log")

        # Set file modification time to 12 hours ago
        twelve_hours_ago = (datetime.now() - timedelta(hours=12)).timestamp()
        os.utime(recent_file, (twelve_hours_ago, twelve_hours_ago))

        cleanup_old_logs(max_age_days=1)

        assert recent_file.exists()

    @freeze_time("2025-11-04 16:00:00")
    def test_deletes_files_older_than_cutoff(self, temp_logs_dir, caplog):
        """Should delete files older than the cutoff date."""
        # Configure caplog to capture INFO level logs
        caplog.set_level(logging.INFO)

        # Create a file from 2 days ago (should be deleted with 1 day cutoff)
        old_file = temp_logs_dir / "scope-logs-2025-11-02-16-00-00.log"
        old_file.write_text("Old log")

        # Set file modification time to 2 days ago
        two_days_ago = (datetime.now() - timedelta(days=2)).timestamp()
        os.utime(old_file, (two_days_ago, two_days_ago))

        cleanup_old_logs(max_age_days=1)

        assert not old_file.exists()
        assert "Cleaned up 1 old log file(s)" in caplog.text

    @freeze_time("2025-11-04 16:00:00")
    def test_deletes_both_base_and_rotated_files(self, temp_logs_dir):
        """Should delete both base .log files and rotated .log.1, .log.2 files."""
        # Create old base log file
        old_base = temp_logs_dir / "scope-logs-2025-11-02-16-00-00.log"
        old_base.write_text("Old base log")

        # Create old rotated files
        old_rotated1 = temp_logs_dir / "scope-logs-2025-11-02-16-00-00.log.1"
        old_rotated2 = temp_logs_dir / "scope-logs-2025-11-02-16-00-00.log.2"
        old_rotated1.write_text("Old rotated 1")
        old_rotated2.write_text("Old rotated 2")

        # Set all to 2 days ago
        two_days_ago = (datetime.now() - timedelta(days=2)).timestamp()
        for file in [old_base, old_rotated1, old_rotated2]:
            os.utime(file, (two_days_ago, two_days_ago))

        cleanup_old_logs(max_age_days=1)

        assert not old_base.exists()
        assert not old_rotated1.exists()
        assert not old_rotated2.exists()

    @freeze_time("2025-11-04 16:00:00")
    def test_respects_custom_max_age_days(self, temp_logs_dir):
        """Should respect custom max_age_days parameter."""
        # Create files at different ages
        file_1_day_old = temp_logs_dir / "scope-logs-2025-11-03-16-00-00.log"
        file_5_days_old = temp_logs_dir / "scope-logs-2025-10-30-16-00-00.log"

        file_1_day_old.write_text("1 day old")
        file_5_days_old.write_text("5 days old")

        one_day_ago = (datetime.now() - timedelta(days=1)).timestamp()
        five_days_ago = (datetime.now() - timedelta(days=5)).timestamp()

        os.utime(file_1_day_old, (one_day_ago, one_day_ago))
        os.utime(file_5_days_old, (five_days_ago, five_days_ago))

        # With max_age_days=3, only 5-day-old file should be deleted
        cleanup_old_logs(max_age_days=3)

        assert file_1_day_old.exists()
        assert not file_5_days_old.exists()

    @freeze_time("2025-11-04 16:00:00")
    def test_uses_default_max_age_of_1_day(self, temp_logs_dir):
        """Should use default max_age_days of 1 when not specified."""
        # Create a file from 2 days ago
        old_file = temp_logs_dir / "scope-logs-2025-11-02-16-00-00.log"
        old_file.write_text("Old log")

        two_days_ago = (datetime.now() - timedelta(days=2)).timestamp()
        os.utime(old_file, (two_days_ago, two_days_ago))

        # Call without specifying max_age_days
        cleanup_old_logs()

        assert not old_file.exists()

    @freeze_time("2025-11-04 16:00:00")
    def test_handles_deletion_errors_gracefully(self, temp_logs_dir, caplog):
        """Should log warning and continue when file deletion fails."""
        # Create an old file
        old_file = temp_logs_dir / "scope-logs-2025-11-02-16-00-00.log"
        old_file.write_text("Old log")

        two_days_ago = (datetime.now() - timedelta(days=2)).timestamp()
        os.utime(old_file, (two_days_ago, two_days_ago))

        # Mock unlink to raise PermissionError
        with patch.object(Path, "unlink", side_effect=PermissionError("Access denied")):
            cleanup_old_logs(max_age_days=1)

        # Should log warning but not crash
        assert "Failed to delete old log file" in caplog.text
        assert "Access denied" in caplog.text

    @freeze_time("2025-11-04 16:00:00")
    def test_logs_info_message_with_deletion_count(self, temp_logs_dir, caplog):
        """Should log info message with count of deleted files."""
        # Configure caplog to capture INFO level logs
        caplog.set_level(logging.INFO)

        # Create 3 old files
        for i in range(3):
            old_file = temp_logs_dir / f"scope-logs-2025-11-02-16-00-0{i}.log"
            old_file.write_text(f"Old log {i}")

            two_days_ago = (datetime.now() - timedelta(days=2)).timestamp()
            os.utime(old_file, (two_days_ago, two_days_ago))

        cleanup_old_logs(max_age_days=1)

        assert "Cleaned up 3 old log file(s) older than 1 day(s)" in caplog.text

    @freeze_time("2025-11-04 16:00:00")
    def test_does_not_log_when_no_files_deleted(self, temp_logs_dir, caplog):
        """Should not log info message when no files are deleted."""
        # Create a recent file
        recent_file = temp_logs_dir / "scope-logs-2025-11-04-15-00-00.log"
        recent_file.write_text("Recent log")

        one_hour_ago = (datetime.now() - timedelta(hours=1)).timestamp()
        os.utime(recent_file, (one_hour_ago, one_hour_ago))

        cleanup_old_logs(max_age_days=1)

        assert "Cleaned up" not in caplog.text

    @freeze_time("2025-11-04 16:00:00")
    def test_mixed_old_and_new_files(self, temp_logs_dir):
        """Should only delete old files and keep new ones in mixed scenario."""
        # Create old files (2 days ago)
        old1 = temp_logs_dir / "scope-logs-2025-11-02-10-00-00.log"
        old2 = temp_logs_dir / "scope-logs-2025-11-02-14-00-00.log.1"

        # Create new files (12 hours ago)
        new1 = temp_logs_dir / "scope-logs-2025-11-04-04-00-00.log"
        new2 = temp_logs_dir / "scope-logs-2025-11-04-04-00-00.log.1"

        for file in [old1, old2, new1, new2]:
            file.write_text("Log content")

        two_days_ago = (datetime.now() - timedelta(days=2)).timestamp()
        twelve_hours_ago = (datetime.now() - timedelta(hours=12)).timestamp()

        os.utime(old1, (two_days_ago, two_days_ago))
        os.utime(old2, (two_days_ago, two_days_ago))
        os.utime(new1, (twelve_hours_ago, twelve_hours_ago))
        os.utime(new2, (twelve_hours_ago, twelve_hours_ago))

        cleanup_old_logs(max_age_days=1)

        # Old files should be deleted
        assert not old1.exists()
        assert not old2.exists()

        # New files should be kept
        assert new1.exists()
        assert new2.exists()


class TestResilientRotatingFileHandler:
    """Tests for ResilientRotatingFileHandler — the mid-session recovery handler."""

    @pytest.fixture
    def log_file(self, tmp_path):
        """Return a path for a log file inside a temp directory."""
        return tmp_path / "logs" / "test.log"

    @pytest.fixture
    def handler(self, log_file):
        """Create a ResilientRotatingFileHandler and ensure it is closed after the test."""
        log_file.parent.mkdir(parents=True, exist_ok=True)
        h = ResilientRotatingFileHandler(str(log_file), maxBytes=1024, backupCount=2)
        h.setFormatter(logging.Formatter("%(message)s"))
        yield h
        h.close()

    # ------------------------------------------------------------------
    # Normal operation
    # ------------------------------------------------------------------

    def test_emits_record_normally(self, handler, log_file):
        """Normal emit should write to the log file."""
        record = logging.LogRecord("test", logging.INFO, "", 0, "hello world", (), None)
        handler.emit(record)
        handler.flush()

        assert "hello world" in log_file.read_text()

    # ------------------------------------------------------------------
    # Recovery after directory deletion (simulates fal.ai /tmp cleanup)
    # ------------------------------------------------------------------

    def test_emit_recovers_after_directory_deleted(self, handler, log_file):
        """emit() should recover if the log directory is deleted after stream closes.

        On Linux, unlinking a file while it is open keeps the FD valid — the
        FileNotFoundError only appears when the stream is closed (e.g. after a
        rollover) and Python then tries to reopen it.  Simulate that by closing
        the stream manually before deleting the directory.
        """
        # Write one record normally
        record = logging.LogRecord("test", logging.INFO, "", 0, "first", (), None)
        handler.emit(record)
        handler.flush()

        # Close the stream (as doRollover() would) then delete the directory
        handler.stream.close()
        handler.stream = None  # type: ignore[assignment]
        shutil.rmtree(log_file.parent)
        assert not log_file.parent.exists()

        # emit() must not raise and must recreate the directory + file
        record2 = logging.LogRecord("test", logging.INFO, "", 0, "second", (), None)
        handler.emit(record2)
        handler.flush()

        assert log_file.exists()
        assert "second" in log_file.read_text()

    def test_emit_recovers_after_stream_closed_and_file_deleted(self, handler, log_file):
        """emit() should recover if the stream is closed and the file is gone."""
        record = logging.LogRecord("test", logging.INFO, "", 0, "first", (), None)
        handler.emit(record)
        handler.flush()

        # Close the stream then delete the file (directory still present)
        handler.stream.close()
        handler.stream = None  # type: ignore[assignment]
        log_file.unlink()
        assert not log_file.exists()

        record2 = logging.LogRecord("test", logging.INFO, "", 0, "second", (), None)
        handler.emit(record2)
        handler.flush()

        assert log_file.exists()
        assert "second" in log_file.read_text()

    # ------------------------------------------------------------------
    # shouldRollover recovery
    # ------------------------------------------------------------------

    def test_should_rollover_recovers_after_stream_closed_and_dir_deleted(
        self, handler, log_file
    ):
        """shouldRollover() should recover without raising if the log dir is gone.

        Mirrors the real failure path: stream closed by doRollover(), then
        the directory vanishes before the next open attempt.
        """
        record = logging.LogRecord("test", logging.INFO, "", 0, "x" * 512, (), None)
        handler.emit(record)
        handler.flush()

        # Close the stream then delete the directory
        handler.stream.close()
        handler.stream = None  # type: ignore[assignment]
        shutil.rmtree(log_file.parent)

        # shouldRollover must not raise
        record2 = logging.LogRecord("test", logging.INFO, "", 0, "y", (), None)
        result = handler.shouldRollover(record2)

        # Result is an int (0 or 1 — either is acceptable after recovery)
        assert isinstance(result, int)
        # Directory should have been recreated by the recovery path
        assert log_file.parent.exists()

    # ------------------------------------------------------------------
    # Irreversible errors still call handleError
    # ------------------------------------------------------------------

    def test_emit_calls_handle_error_on_unrecoverable_failure(self, handler, log_file):
        """emit() should fall back to handleError if recovery itself fails."""
        handle_error_called = []

        def fake_handle_error(record):
            handle_error_called.append(record)

        handler.handleError = fake_handle_error

        # Force emit to always raise FileNotFoundError (even after reopen)
        original_open = handler._open

        def always_fail():
            raise FileNotFoundError("Cannot create file")

        handler._open = always_fail

        # Close the existing stream to force a reopen attempt
        handler.stream.close()
        handler.stream = None  # type: ignore[assignment]

        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        handler.emit(record)  # Must not propagate an exception

        assert len(handle_error_called) == 1
        handler._open = original_open  # restore
