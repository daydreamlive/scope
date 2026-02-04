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

            # If there's an error, return it even without artifact data
            if "error" in data:
                result = {
                    "is_downloading": False,
                    "percentage": 0,
                    "current_artifact": None,
                    "error": data["error"],
                }
                # Include last artifact info if available
                if data["artifacts"]:
                    *_, (current_artifact, current_data) = data["artifacts"].items()
                    result["current_artifact"] = current_artifact
                    if current_data["total_mb"] > 0:
                        result["percentage"] = round(
                            current_data["downloaded_mb"]
                            / current_data["total_mb"]
                            * 100,
                            1,
                        )
                return result

            if not data["artifacts"]:
                return None

            # The current artifact is the last one in the dict
            *_, (current_artifact, current_data) = data["artifacts"].items()

            # Calculate percentage for current artifact
            current_percentage = 0
            if current_data["total_mb"] > 0:
                current_percentage = (
                    current_data["downloaded_mb"] / current_data["total_mb"] * 100
                )

            return {
                "is_downloading": data["is_downloading"],
                "percentage": round(current_percentage, 1),
                "current_artifact": current_artifact,
            }

    def mark_complete(self, pipeline_id: str):
        """Mark download as complete."""
        with self._lock:
            if pipeline_id in self._progress:
                self._progress[pipeline_id]["is_downloading"] = False

    def mark_error(self, pipeline_id: str, error_message: str):
        """Mark download as failed with an error message."""
        with self._lock:
            if pipeline_id not in self._progress:
                self._progress[pipeline_id] = {"artifacts": {}, "is_downloading": False}
            self._progress[pipeline_id]["is_downloading"] = False
            self._progress[pipeline_id]["error"] = error_message

    def clear_progress(self, pipeline_id: str):
        """Clear progress data."""
        with self._lock:
            self._progress.pop(pipeline_id, None)


# Global singleton instance
download_progress_manager = DownloadProgressManager()
