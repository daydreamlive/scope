"""
Logs configuration module for daydream-scope.

Provides centralized configuration for log storage location with support for:
- Default location: ~/.daydream-scope/logs
- Environment variable override: DAYDREAM_SCOPE_LOGS_DIR
"""

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

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
