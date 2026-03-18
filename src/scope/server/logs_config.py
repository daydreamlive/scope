"""
Logs configuration module for daydream-scope.

Provides centralized configuration for log storage location with support for:
- Default location: ~/.daydream-scope/logs
- Environment variable override: DAYDREAM_SCOPE_LOGS_DIR
- Fal connection ID injection into log lines (when running on fal.ai)
"""

import logging
import os
import threading
from datetime import datetime, timedelta
from logging.handlers import RotatingFileHandler
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fal connection ID tracking — thread-safe global for log correlation
# ---------------------------------------------------------------------------

_fal_connection_id: str | None = None
_fal_connection_id_lock = threading.Lock()


def set_fal_connection_id(connection_id: str | None) -> None:
    """Set (or clear) the fal connection ID that is injected into every log line."""
    global _fal_connection_id
    with _fal_connection_id_lock:
        _fal_connection_id = connection_id


def get_fal_connection_id() -> str | None:
    """Return the current fal connection ID, or None if not set."""
    return _fal_connection_id


class FalConnectionFilter(logging.Filter):
    """Logging filter that adds a ``fal_conn`` attribute to every record.

    When a fal connection ID is set, ``record.fal_conn`` is ``"[<id>] "``;
    otherwise it is the empty string.  Use ``%(fal_conn)s`` in the format
    string so the prefix appears only when running on fal.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        cid = _fal_connection_id
        record.fal_conn = f"[{cid}] " if cid else ""  # type: ignore[attr-defined]
        return True


LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(fal_conn)s%(message)s"

# Default logs directory
DEFAULT_LOGS_DIR = "~/.daydream-scope/logs"

# Environment variable for overriding logs directory
LOGS_DIR_ENV_VAR = "DAYDREAM_SCOPE_LOGS_DIR"


def get_logs_dir() -> Path:
    """
    Get the logs directory path.

    Priority order:
    1. DAYDREAM_SCOPE_LOGS_DIR environment variable
    2. Default: ~/.daydream-scope/logs

    Returns:
        Path: Absolute path to the logs directory
    """
    # Check environment variable first
    env_dir = os.environ.get(LOGS_DIR_ENV_VAR)
    if env_dir:
        logs_dir = Path(env_dir).expanduser().resolve()
        return logs_dir

    # Use default directory
    logs_dir = Path(DEFAULT_LOGS_DIR).expanduser().resolve()
    return logs_dir


def ensure_logs_dir() -> Path:
    """
    Get the logs directory path and ensure it exists.

    Returns:
        Path: Absolute path to the logs directory
    """
    logs_dir = get_logs_dir()
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir


def get_current_log_file() -> Path:
    """
    Get the path to the current log file with timestamp.

    Creates a new timestamped log file for each app session/startup.
    The RotatingFileHandler will handle rotation within a session if needed.

    Returns:
        Path: Absolute path to scope-logs-YYYY-MM-DD-HH-MM-SS.log
    """
    logs_dir = get_logs_dir()
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    return logs_dir / f"scope-logs-{timestamp}.log"


def get_most_recent_log_file() -> Path | None:
    """
    Get the most recent log file from the logs directory by sorting filenames.

    Returns:
        Path: Absolute path to the most recent scope-logs-*.log file, or None if no logs exist
    """
    logs_dir = get_logs_dir()

    if not logs_dir.exists():
        return None

    # Find all log files matching the pattern (base files only, not rotated .1, .2, etc.)
    log_files = list(logs_dir.glob("scope-logs-*.log"))

    if not log_files:
        return None

    # Sort by filename (timestamp is in the name) and return the most recent
    return sorted(log_files)[-1]


def cleanup_old_logs(max_age_days: int = 1) -> None:
    """
    Delete log files older than max_age_days.

    Args:
        max_age_days: Maximum age of log files to keep (default: 1 day)
    """
    logs_dir = get_logs_dir()

    if not logs_dir.exists():
        return

    cutoff_time = datetime.now() - timedelta(days=max_age_days)
    cutoff_timestamp = cutoff_time.timestamp()

    # Find all log files (including rotated ones like .log.1, .log.2, etc.)
    log_files = list(logs_dir.glob("scope-logs-*.log*"))

    deleted_count = 0
    for log_file in log_files:
        try:
            if log_file.stat().st_mtime < cutoff_timestamp:
                log_file.unlink()
                deleted_count += 1
        except Exception as e:
            logger.warning(f"Failed to delete old log file {log_file}: {e}")

    if deleted_count > 0:
        logger.info(
            f"Cleaned up {deleted_count} old log file(s) older than {max_age_days} day(s)"
        )


class ResilientRotatingFileHandler(RotatingFileHandler):
    """A RotatingFileHandler that recovers gracefully when the log file or its
    parent directory is deleted mid-session (e.g. by OS /tmp cleanup on fal.ai
    workers).

    Standard behaviour: if ``shouldRollover()`` or ``emit()`` encounters a
    ``FileNotFoundError`` the Python logging framework catches it and writes a
    noisy ``--- Logging error ---`` traceback to *stderr* for every subsequent
    log call.

    This subclass intercepts ``FileNotFoundError`` at both sites, recreates the
    log directory and reopens the stream, then retries the operation once.  All
    other exceptions are left to the default ``handleError`` path.
    """

    def _reopen_stream(self) -> None:
        """Close the current stream (if any) and reopen the log file.

        Recreates the parent directory first so the open cannot fail with
        ``FileNotFoundError`` again.
        """
        if self.stream:
            try:
                self.stream.close()
            except Exception:
                pass
            self.stream = None  # type: ignore[assignment]

        Path(self.baseFilename).parent.mkdir(parents=True, exist_ok=True)
        self.stream = self._open()

    def shouldRollover(self, record: logging.LogRecord) -> int:
        """Override to recover if the log file has been deleted."""
        try:
            return super().shouldRollover(record)
        except FileNotFoundError:
            # Log file (or its directory) was deleted; reopen before deciding.
            try:
                self._reopen_stream()
            except Exception:
                return 0  # Can't recover — skip rollover check.
            try:
                return super().shouldRollover(record)
            except Exception:
                return 0

    def emit(self, record: logging.LogRecord) -> None:
        """Override to recover if the log file has been deleted mid-session."""
        try:
            super().emit(record)
        except FileNotFoundError:
            # Directory or file was deleted; recreate and retry once.
            try:
                self._reopen_stream()
                super().emit(record)
            except Exception:
                self.handleError(record)
