"""Virtual environment snapshot for rollback support.

Provides freeze-based venv state capture and restoration to enable
rollback when plugin installation fails after dependency validation passes.
"""

import logging
import os
import shutil
import subprocess

from .plugins_config import (
    get_freeze_backup_file,
    get_resolved_backup_file,
    get_resolved_file,
)

logger = logging.getLogger(__name__)


class VenvSnapshot:
    """Captures and restores venv state for rollback.

    Uses `uv pip freeze` to capture the current installed state and
    `uv pip sync` to restore it on failure. This provides accurate
    rollback with minimal disk usage (~100KB for backup files vs 7GB+
    for a full venv copy).
    """

    def __init__(self):
        self._freeze_file = get_freeze_backup_file()
        self._resolved_backup = get_resolved_backup_file()
        self._resolved_file = get_resolved_file()

    def capture(self) -> bool:
        """Capture current venv state by running uv pip freeze.

        Creates:
        - freeze.txt: Output of `uv pip freeze`
        - resolved.txt.bak: Backup of current resolved.txt (if exists)

        Returns:
            True if capture succeeded, False otherwise
        """
        try:
            # Run uv pip freeze to capture installed packages
            env = {**os.environ, "PYTHONUTF8": "1"}
            result = subprocess.run(
                ["uv", "pip", "freeze"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
            )

            if result.returncode != 0:
                logger.error(f"Failed to capture venv state: {result.stderr}")
                return False

            # Ensure parent directory exists
            self._freeze_file.parent.mkdir(parents=True, exist_ok=True)

            # Write freeze output
            self._freeze_file.write_text(result.stdout)
            logger.info(f"Captured venv state to {self._freeze_file}")

            # Backup resolved.txt if it exists
            if self._resolved_file.exists():
                shutil.copy2(self._resolved_file, self._resolved_backup)
                logger.info(
                    f"Backed up {self._resolved_file} to {self._resolved_backup}"
                )

            return True

        except Exception as e:
            logger.error(f"Failed to capture venv snapshot: {e}")
            return False

    def restore(self) -> tuple[bool, str | None]:
        """Restore venv to captured state using uv pip sync.

        Restores:
        - Runs `uv pip sync freeze.txt` to restore package state
        - Copies resolved.txt.bak back to resolved.txt (if backup exists)

        Returns:
            Tuple of (success, error_message)
        """
        try:
            # Check if we have a freeze file to restore from
            if not self._freeze_file.exists():
                return False, "No freeze file found for rollback"

            # Restore resolved.txt from backup first
            if self._resolved_backup.exists():
                shutil.copy2(self._resolved_backup, self._resolved_file)
                logger.info(f"Restored {self._resolved_file} from backup")

            # Run uv pip sync to restore packages
            env = {**os.environ, "PYTHONUTF8": "1"}
            result = subprocess.run(
                [
                    "uv",
                    "pip",
                    "sync",
                    "--torch-backend",
                    "cu128",
                    str(self._freeze_file),
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
            )

            if result.returncode != 0:
                error_msg = f"Failed to restore venv state: {result.stderr}"
                logger.error(error_msg)
                return False, error_msg

            logger.info("Successfully restored venv state from freeze")
            return True, None

        except Exception as e:
            error_msg = f"Exception during venv restore: {e}"
            logger.error(error_msg)
            return False, error_msg

    def discard(self) -> None:
        """Remove backup files after successful install.

        Removes freeze.txt and resolved.txt.bak if they exist.
        """
        try:
            if self._freeze_file.exists():
                self._freeze_file.unlink()
                logger.debug(f"Removed freeze file: {self._freeze_file}")

            if self._resolved_backup.exists():
                self._resolved_backup.unlink()
                logger.debug(f"Removed resolved backup: {self._resolved_backup}")

        except Exception as e:
            # Non-fatal - just log the error
            logger.warning(f"Failed to clean up snapshot files: {e}")

    def has_snapshot(self) -> bool:
        """Check if a snapshot exists.

        Returns:
            True if freeze.txt exists, False otherwise
        """
        return self._freeze_file.exists()
