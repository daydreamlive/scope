"""
Cross-platform model downloader using huggingface_hub for HF repo/files.
"""

import argparse
import logging
import os
import sys
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

# Current download context for tqdm callback
_current_pipeline_id: str | None = None
_current_artifact: str | None = None


def set_download_context(pipeline_id: str, artifact: str):
    """Set the current download context for progress tracking."""
    global _current_pipeline_id, _current_artifact
    _current_pipeline_id = pipeline_id
    _current_artifact = artifact


def clear_download_context():
    """Clear the download context."""
    global _current_pipeline_id, _current_artifact
    _current_pipeline_id = None
    _current_artifact = None


# --- third-party libs ---
try:
    from huggingface_hub import snapshot_download
    from tqdm.auto import tqdm
except Exception:
    print(
        "Error: huggingface_hub and tqdm are required. Install with: pip install huggingface_hub tqdm",
        file=sys.stderr,
    )
    raise

# Ideally we would use a custom tqdm_class with HF, but the proper usage is unclear
# Instead we monkey patch tqdm.update to log progress every 5%
PROGRESS_LOG_INTERVAL_PERCENT = 5.0

_original_tqdm_update = tqdm.update

# Track last logged progress per tqdm instance (by id)
_last_logged_progress: dict[int, float] = {}


def _patched_tqdm_update(self, n: int = 1):
    """Patched tqdm update that logs progress every 5%."""
    # Call original update FIRST to increment self.n
    result = _original_tqdm_update(self, n)

    try:
        if self.n is not None and self.total is not None and self.total > 0:
            current_progress = (self.n / self.total) * 100

            # Skip logging at 0% progress
            if current_progress == 0.0:
                return result

            instance_id = id(self)
            last_logged = _last_logged_progress.get(instance_id, 0.0)

            # Only log if we've made at least PROGRESS_LOG_INTERVAL_PERCENT progress since last log
            if current_progress >= last_logged + PROGRESS_LOG_INTERVAL_PERCENT:
                downloaded = self.n / 1024 / 1024
                total_size = self.total / 1024 / 1024
                logger.info(f"Downloaded {downloaded:.2f}MB of {total_size:.2f}MB")
                _last_logged_progress[instance_id] = current_progress

                # Clear dict entry when progress reaches 100%
                if current_progress >= 100.0:
                    _last_logged_progress.pop(instance_id, None)

            # Update progress tracker for UI (now with the updated self.n value)
            if _current_pipeline_id and _current_artifact:
                try:
                    download_progress_manager.update(
                        _current_pipeline_id,
                        _current_artifact,
                        self.n / 1024 / 1024,
                        self.total / 1024 / 1024,
                    )
                except Exception:
                    # Don't let progress tracking interfere with download
                    pass
    except KeyboardInterrupt:
        # Re-raise KeyboardInterrupt to allow proper signal handling
        raise
    except Exception:
        # Don't let logging errors interfere with tqdm or signal handling
        pass

    return result


# Apply the monkey patch
tqdm.update = _patched_tqdm_update


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
        # token is picked up automatically from HUGGINGFACE_TOKEN if set
        # revision=None,  # optionally pin a commit/tag if you like
    )
    logger.info(f"Completed download of repo '{repo_id}' to: {local_dir}")


def download_artifact(artifact: Artifact, models_root: Path, pipeline_id: str) -> None:
    """
    Download an artifact to the models directory.

    This is a generic dispatcher that routes to the appropriate download
    function based on the artifact type.

    Args:
        artifact: The artifact to download
        models_root: Root directory where models are stored
        pipeline_id: Optional pipeline ID for progress tracking

    Raises:
        ValueError: If artifact type is not supported
    """
    if isinstance(artifact, HuggingfaceRepoArtifact):
        download_hf_artifact(artifact, models_root, pipeline_id)
    else:
        raise ValueError(f"Unsupported artifact type: {type(artifact)}")


def download_hf_artifact(
    artifact: HuggingfaceRepoArtifact, models_root: Path, pipeline_id: str
) -> None:
    """
    Download a HuggingFace repository artifact.

    Downloads specific files/directories from a HuggingFace repository.

    Args:
        artifact: HuggingFace repo artifact
        models_root: Root directory where models are stored
        pipeline_id: Pipeline ID to download models for
    """
    # Set up progress tracking context
    set_download_context(pipeline_id, artifact.repo_id)

    try:
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
    finally:
        # Always clear context after download
        if pipeline_id:
            clear_download_context()


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

    # Download each artifact (progress tracking starts in set_download_context)
    for artifact in artifacts:
        download_artifact(artifact, models_root, pipeline_id)


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
