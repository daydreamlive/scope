"""
Cross-platform model downloader using huggingface_hub for HF repo/files.
"""

import argparse
import logging
import os
import sys
import threading
from pathlib import Path

from .artifacts import Artifact, HuggingfaceRepoArtifact
from .download_progress_manager import download_progress_manager

# Disable hf_transfer to use standard download method
# This prevents errors when HF_HUB_ENABLE_HF_TRANSFER=1 is set but hf_transfer is not installed
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
# Disable xet for now because it seems to sometimes causes an issue with exiting the server after a download
# This has not been investigated thoroughly, but disabling it seems to fix the issue for now
os.environ["HF_HUB_ENABLE_HF_XET"] = "1"

from .models_config import (
    ensure_models_dir,
)

# Set up logger
logger = logging.getLogger(__name__)


# --- third-party libs ---
try:
    from huggingface_hub import snapshot_download
except Exception:
    print(
        "Error: huggingface_hub is required. Install with: pip install huggingface_hub",
        file=sys.stderr,
    )
    raise


def get_directory_size(path: Path) -> int:
    """Get total size of a directory in bytes."""
    total = 0
    logger.info(f"Getting directory size of {path}")
    if path.exists():
        if path.is_file():
            return path.stat().st_size
        for item in path.rglob("*"):
            if item.is_file():
                try:
                    total += item.stat().st_size
                except (OSError, FileNotFoundError):
                    # File may have been moved/deleted during iteration
                    pass
    return total


class DownloadProgressLogger:
    """Background thread that logs directory sizes every 5 seconds during downloads."""

    def __init__(
        self,
        pipeline_id: str,
        directories: list[tuple[str, Path]],
        interval: float = 5.0,
    ):
        """
        Args:
            pipeline_id: Pipeline ID for progress tracking
            directories: List of (artifact_name, directory_path) tuples to monitor
            interval: Logging interval in seconds
        """
        self._pipeline_id = pipeline_id
        self._directories = directories
        self._interval = interval
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self):
        """Start the background logging thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._log_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the background logging thread."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _log_loop(self):
        """Background loop that logs directory sizes."""
        while not self._stop_event.is_set():
            try:
                for artifact_name, dir_path in self._directories:
                    size_bytes = get_directory_size(dir_path)
                    size_mb = size_bytes / (1024 * 1024)
                    logger.info(f"[{artifact_name}] Downloaded: {size_mb:.2f} MB")

                    # Update progress manager for UI
                    # Note: We don't know the total size, so we just report current size
                    # The UI will show the current downloaded size
                    download_progress_manager.update(
                        self._pipeline_id,
                        artifact_name,
                        size_mb,
                        0,  # Total unknown when using directory size monitoring
                    )
            except Exception as e:
                logger.debug(f"Error logging directory sizes: {e}")

            # Wait for interval or until stopped
            self._stop_event.wait(self._interval)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False


def download_hf_repo(
    repo_id: str,
    local_dir: Path,
    filename: str | None = None,
    allow_patterns: list[str] | None = None,
    ignore_patterns: list[str] | None = None,
) -> None:
    """
    Download from HuggingFace repo - either a single file or repo snapshot with patterns.

    Args:
        repo_id: HuggingFace repository ID
        local_dir: Local directory to download to
        filename: Optional single filename to download (uses hf_hub_download)
        allow_patterns: Optional list of patterns to include (glob-like, relative to repo root)
        ignore_patterns: Optional list of patterns to exclude (glob-like, relative to repo root)
    """
    local_dir.mkdir(parents=True, exist_ok=True)

    if filename:
        # Single file download using snapshot_download with allow_patterns
        # (hf_hub_download doesn't support tqdm_class parameter)
        logger.info(f"Starting download of '{filename}' from '{repo_id}'")
        allow_patterns = [filename]
    else:
        # Repo snapshot download
        logger.info(f"Starting download of repo '{repo_id}' to: {local_dir}")

    snapshot_download(
        repo_id=repo_id,
        # In previous versions, we used local_dir_use_symlinks=False to copy files for portability.
        # However, this is not necessary anymore with snapshot_download
        local_dir=str(local_dir),
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
        # TODO: Remove this once we can support progress tracking when there are multiple workers
        max_workers=1,
        # token is picked up automatically from HUGGINGFACE_TOKEN if set
        # revision=None,  # optionally pin a commit/tag if you like
    )
    logger.info(f"Completed download of repo '{repo_id}' to: {local_dir}")


def download_artifact(artifact: Artifact, models_root: Path) -> None:
    """
    Download an artifact to the models directory.

    This is a generic dispatcher that routes to the appropriate download
    function based on the artifact type.

    Args:
        artifact: The artifact to download
        models_root: Root directory where models are stored

    Raises:
        ValueError: If artifact type is not supported
    """
    if isinstance(artifact, HuggingfaceRepoArtifact):
        download_hf_artifact(artifact, models_root)
    else:
        raise ValueError(f"Unsupported artifact type: {type(artifact)}")


def download_hf_artifact(
    artifact: HuggingfaceRepoArtifact, models_root: Path
) -> None:
    """
    Download a HuggingFace repository artifact.

    Downloads specific files/directories from a HuggingFace repository.

    Args:
        artifact: HuggingFace repo artifact
        models_root: Root directory where models are stored
    """
    local_dir = models_root / artifact.repo_id.split("/")[-1]

    # Convert file/directory specifications to glob patterns
    allow_patterns = []
    for file in artifact.files:
        # Add the file/directory itself
        allow_patterns.append(file)
        # If it's a directory, also include everything inside it
        # This handles both "google" and "google/" formats
        if not file.endswith(("/", ".pt", ".pth", ".safetensors", ".json")):
            # Likely a directory, add pattern to include its contents
            allow_patterns.append(f"{file}/*")
            allow_patterns.append(f"{file}/**/*")

    logger.info(f"Downloading from {artifact.repo_id}: {artifact.files}")
    download_hf_repo(
        repo_id=artifact.repo_id,
        local_dir=local_dir,
        allow_patterns=allow_patterns,
    )


def download_models(pipeline_id: str) -> None:
    """
    Download models for a specific pipeline.

    Args:
        pipeline_id: Pipeline ID to download models for.
    """
    from .pipeline_artifacts import PIPELINE_ARTIFACTS

    models_root = ensure_models_dir()

    logger.info(f"Downloading models for pipeline: {pipeline_id}")
    artifacts = PIPELINE_ARTIFACTS[pipeline_id]

    # Build list of directories to monitor for progress logging
    directories_to_monitor: list[tuple[str, Path]] = []
    for artifact in artifacts:
        if isinstance(artifact, HuggingfaceRepoArtifact):
            local_dir = models_root / artifact.repo_id.split("/")[-1]
            directories_to_monitor.append((artifact.repo_id, local_dir))

    # Start background progress logger and download all artifacts
    with DownloadProgressLogger(pipeline_id, directories_to_monitor):
        for artifact in artifacts:
            download_artifact(artifact, models_root)


def main():
    """Main entry point for the download_models script."""
    parser = argparse.ArgumentParser(
        description="Download models for Daydream Scope pipelines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download specific pipeline
  python download_models.py --pipeline streamdiffusionv2
  python download_models.py --pipeline longlive
  python download_models.py --pipeline krea-realtime-video
  python download_models.py --pipeline reward-forcing
  python download_models.py -p streamdiffusionv2
        """,
    )
    parser.add_argument(
        "--pipeline",
        "-p",
        type=str,
        default=None,
        required=True,
        help="Pipeline ID (e.g., 'streamdiffusionv2', 'longlive', 'krea-realtime-video', 'reward-forcing').",
    )

    args = parser.parse_args()

    try:
        download_models(args.pipeline)
    except KeyboardInterrupt:
        print("\nCancelled.")
        sys.exit(130)
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
