"""
Download progress tracking for pipeline model downloads.
"""

import threading


class DownloadProgressManager:
    """Simple progress tracker for pipeline downloads."""

    def __init__(self):
        self._progress = {}
        self._lock = threading.Lock()

    def update(
        self, pipeline_id: str, artifact: str, downloaded_mb: float, total_mb: float
    ):
        """Update download progress."""
        with self._lock:
            if pipeline_id not in self._progress:
                self._progress[pipeline_id] = {"artifacts": {}, "is_downloading": True}
            self._progress[pipeline_id]["artifacts"][artifact] = {
                "downloaded_mb": downloaded_mb,
                "total_mb": total_mb,
            }

    def get_progress(self, pipeline_id: str):
        """Get current artifact progress."""
        with self._lock:
            if pipeline_id not in self._progress:
                return None
            data = self._progress[pipeline_id]
            if not data["artifacts"]:
                return None

            # Calculate total downloaded MB across all artifacts
            total_downloaded_mb = sum(
                artifact_data["downloaded_mb"]
                for artifact_data in data["artifacts"].values()
            )

            return {
                "is_downloading": data["is_downloading"],
                "total_downloaded_mb": round(total_downloaded_mb, 2),
            }

    def mark_complete(self, pipeline_id: str):
        """Mark download as complete."""
        with self._lock:
            if pipeline_id in self._progress:
                self._progress[pipeline_id]["is_downloading"] = False

    def clear_progress(self, pipeline_id: str):
        """Clear progress data."""
        with self._lock:
            self._progress.pop(pipeline_id, None)


# Global singleton instance
download_progress_manager = DownloadProgressManager()
