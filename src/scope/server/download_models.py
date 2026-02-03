"""
Cross-platform model downloader
"""

import argparse
import http
import logging
import os
import re
import shutil
import sys
from collections.abc import Callable
from pathlib import Path

import httpx

from scope.core.pipelines.artifacts import (
    Artifact,
    GoogleDriveArtifact,
    HuggingfaceRepoArtifact,
)

from .download_progress_manager import download_progress_manager
from .models_config import ensure_models_dir

# Set up logger
logger = logging.getLogger(__name__)

# Download settings
CHUNK_SIZE = 10 * 1024 * 1024  # 10MB chunks
PROGRESS_LOG_INTERVAL_PERCENT = 5.0
DOWNLOAD_TIMEOUT = httpx.Timeout(connect=30.0, read=60.0, write=30.0, pool=30.0)


def get_repo_files(
    repo_id: str,
    allow_patterns: list[str] | None = None,
    ignore_patterns: list[str] | None = None,
) -> list[dict]:
    """
    Get list of files in a HuggingFace repo with their metadata.

    Args:
        repo_id: HuggingFace repository ID
        allow_patterns: Optional list of glob patterns to include
        ignore_patterns: Optional list of glob patterns to exclude

    Returns:
        List of dicts with 'path', 'size', and 'url' keys
    """
    from fnmatch import fnmatch

    from huggingface_hub import HfApi, hf_hub_url

    api = HfApi()

    # Get all file paths first (fast operation)
    all_paths = api.list_repo_files(repo_id)

    # Filter paths by patterns
    filtered_paths = []
    for path in all_paths:
        # Apply allow patterns filter
        if allow_patterns:
            if not any(fnmatch(path, pattern) for pattern in allow_patterns):
                continue

        # Apply ignore patterns filter
        if ignore_patterns:
            if any(fnmatch(path, pattern) for pattern in ignore_patterns):
                continue

        filtered_paths.append(path)

    if not filtered_paths:
        return []

    # Get file info (including sizes) only for filtered files
    paths_info = api.get_paths_info(repo_id, filtered_paths)

    files = []
    for info in paths_info:
        url = hf_hub_url(repo_id, info.path)
        files.append(
            {
                "path": info.path,
                "size": info.size,
                "url": url,
            }
        )

    return files


def http_get(
    url: str,
    dest_path: Path,
    expected_size: int | None = None,
    on_progress: Callable[[int], None] | None = None,
) -> None:
    """
    Download a file using httpx with streaming and resume support.

    Downloads to a temp file (.incomplete suffix) and moves to final location when complete.
    If a partial file exists, attempts to resume from where it left off.

    Args:
        url: URL to download from
        dest_path: Final destination path for the file
        expected_size: Expected file size in bytes (for progress tracking)
        on_progress: Optional callback(downloaded_bytes) for progress updates
    """
    temp_path = dest_path.with_suffix(dest_path.suffix + ".incomplete")
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if we already have the complete file
    if dest_path.exists():
        if expected_size is None or dest_path.stat().st_size == expected_size:
            logger.debug(f"File for {url} already exists: {dest_path}")
            if on_progress and expected_size:
                on_progress(expected_size)
            return

    # Check for existing partial download to resume
    resume_from = 0
    if temp_path.exists():
        resume_from = temp_path.stat().st_size
        logger.info(
            f"Resuming download for {url} from {resume_from / 1024 / 1024:.2f}MB"
        )

    headers = {}
    token = os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    # Add Range header for resuming
    if resume_from > 0:
        headers["Range"] = f"bytes={resume_from}-"

    try:
        with httpx.stream(
            "GET", url, headers=headers, timeout=DOWNLOAD_TIMEOUT, follow_redirects=True
        ) as response:
            # Handle resume response
            if (
                resume_from > 0
                and response.status_code
                == http.HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE
            ):
                # Range not satisfiable - file might be complete or changed
                logger.warning("Resume not supported or file changed, starting fresh")
                temp_path.unlink(missing_ok=True)
                resume_from = 0
                # Retry without Range header
                headers.pop("Range", None)
                with httpx.stream(
                    "GET",
                    url,
                    headers=headers,
                    timeout=DOWNLOAD_TIMEOUT,
                    follow_redirects=True,
                ) as retry_response:
                    retry_response.raise_for_status()
                    _write_stream_to_file(
                        retry_response, temp_path, 0, expected_size, on_progress
                    )
            elif response.status_code == http.HTTPStatus.PARTIAL_CONTENT:
                # Partial content - resume successful
                response.raise_for_status()
                _write_stream_to_file(
                    response, temp_path, resume_from, expected_size, on_progress
                )
            else:
                # Full download (200 OK)
                response.raise_for_status()
                # If we got a 200 when expecting to resume, start fresh
                if resume_from > 0:
                    temp_path.unlink(missing_ok=True)
                    resume_from = 0
                _write_stream_to_file(
                    response, temp_path, 0, expected_size, on_progress
                )

        # Move temp file to final location
        shutil.move(str(temp_path), str(dest_path))
        logger.debug(f"Download for {url} complete: {dest_path}")

    except KeyboardInterrupt:
        logger.info(f"Download interrupted, partial file for {url} saved for resume")
        raise
    except Exception as e:
        logger.error(f"Download failed for {url}: {e}")
        raise


def _write_stream_to_file(
    response: httpx.Response,
    temp_path: Path,
    resume_from: int,
    expected_size: int | None,
    on_progress: Callable[[int], None] | None,
) -> None:
    """Write streaming response to file with progress tracking."""
    downloaded = resume_from
    last_logged_percent = 0.0

    # Open in append mode if resuming, write mode otherwise
    mode = "ab" if resume_from > 0 else "wb"

    with open(temp_path, mode) as f:
        for chunk in response.iter_bytes(chunk_size=CHUNK_SIZE):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)

                # Progress logging
                if expected_size and expected_size > 0:
                    percent = (downloaded / expected_size) * 100

                    # Log every PROGRESS_LOG_INTERVAL_PERCENT
                    if percent >= last_logged_percent + PROGRESS_LOG_INTERVAL_PERCENT:
                        logger.info(
                            f"Downloaded {downloaded / 1024 / 1024:.2f}MB of "
                            f"{expected_size / 1024 / 1024:.2f}MB ({percent:.1f}%)"
                        )
                        last_logged_percent = percent

                    # Call progress callback
                    if on_progress:
                        on_progress(downloaded)


def download_hf_repo(
    repo_id: str,
    local_dir: Path,
    allow_patterns: list[str] | None = None,
    ignore_patterns: list[str] | None = None,
    pipeline_id: str | None = None,
) -> None:
    """
    Download from HuggingFace repo - either a single file or repo snapshot with patterns.

    Args:
        repo_id: HuggingFace repository ID
        local_dir: Local directory to download to
        allow_patterns: Optional list of patterns to include (glob-like, relative to repo root)
        ignore_patterns: Optional list of patterns to exclude (glob-like, relative to repo root)
        pipeline_id: Optional pipeline ID for progress tracking
    """
    local_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting download of repo '{repo_id}' to: {local_dir}")

    # Get list of files to download
    files = get_repo_files(repo_id, allow_patterns, ignore_patterns)

    if not files:
        logger.warning(f"No files matched patterns in {repo_id}")
        return

    # Calculate total size for progress tracking
    total_size = sum(f["size"] for f in files)
    total_downloaded = 0

    logger.info(
        f"Downloading {len(files)} files ({total_size / 1024 / 1024:.2f}MB total)"
    )

    def make_progress_callback(downloaded_offset: int):
        """Create a progress callback that accounts for already-downloaded files."""

        def on_progress(downloaded: int):
            total_downloaded = downloaded_offset + downloaded

            # Update progress manager for UI
            if pipeline_id:
                try:
                    download_progress_manager.update(
                        pipeline_id,
                        repo_id,
                        total_downloaded / 1024 / 1024,
                        total_size / 1024 / 1024,
                    )
                except Exception:
                    pass

        return on_progress

    # Download each file
    for i, file_info in enumerate(files, 1):
        file_path = file_info["path"]
        file_size = file_info["size"]
        file_url = file_info["url"]
        dest_path = local_dir / file_path

        logger.info(
            f"[{i}/{len(files)}] Downloading: {file_path} ({file_size / 1024 / 1024:.2f}MB)"
        )

        http_get(
            url=file_url,
            dest_path=dest_path,
            expected_size=file_size,
            on_progress=make_progress_callback(total_downloaded),
        )

        total_downloaded += file_size

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
    elif isinstance(artifact, GoogleDriveArtifact):
        download_google_drive_artifact(artifact, models_root, pipeline_id)
    else:
        raise ValueError(f"Unsupported artifact type: {type(artifact)}")


def download_google_drive_artifact(
    artifact: GoogleDriveArtifact, models_root: Path, pipeline_id: str
) -> None:
    """
    Download a Google Drive file artifact (supports large files and ZIP extraction).

    Args:
        artifact: Google Drive artifact with file_id and optional files to extract
        models_root: Root directory where models are stored
        pipeline_id: Pipeline ID for progress tracking
    """
    import zipfile

    # Extract file ID from URL or use directly
    file_id = artifact.file_id
    if "drive.google.com" in file_id:
        match = re.search(r"(?:/d/|id=)([a-zA-Z0-9_-]+)", file_id)
        if not match:
            raise ValueError(f"Could not extract file ID from URL: {artifact.file_id}")
        file_id = match.group(1)

    output_dir = models_root / (artifact.name or "")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading from Google Drive (file_id: {file_id}) to: {output_dir}")

    # Download to temp file
    temp_path = output_dir / f"{file_id}.tmp"
    _download_google_drive_raw(file_id, temp_path, pipeline_id)

    # Handle the downloaded file
    if not artifact.files:
        # No specific files requested - just rename temp to file_id
        dest = output_dir / file_id
        shutil.move(str(temp_path), str(dest))
        logger.info(f"Downloaded file to {dest}")
        return

    # Try to extract as ZIP
    try:
        with zipfile.ZipFile(temp_path, "r") as zf:
            logger.info(f"Extracting ZIP archive to {output_dir}")
            zf.extractall(output_dir)

            # Find and move target files to output_dir root
            found = set()
            for target in artifact.files:
                for name in zf.namelist():
                    if name.endswith(target) and not name.startswith("__MACOSX"):
                        src = output_dir / name
                        dest = output_dir / target
                        if src != dest:
                            shutil.move(str(src), str(dest))
                        logger.info(f"Extracted {target}")
                        found.add(target)
                        break

            if missing := set(artifact.files) - found:
                logger.warning(f"Files not found in ZIP: {missing}")

        temp_path.unlink()
    except zipfile.BadZipFile:
        # Not a ZIP - treat as single file
        if len(artifact.files) == 1:
            dest = output_dir / artifact.files[0]
            shutil.move(str(temp_path), str(dest))
            logger.info(f"Downloaded file to {dest}")
        else:
            logger.warning(
                f"Not a ZIP but multiple files expected, keeping as {temp_path.name}"
            )


def _download_google_drive_raw(file_id: str, dest_path: Path, pipeline_id: str) -> None:
    """
    Download a file from Google Drive, handling virus scan warnings for large files.

    Uses streaming download (http_get) which handles large files properly.
    """

    def on_progress(downloaded: int) -> None:
        if pipeline_id:
            try:
                download_progress_manager.update(
                    pipeline_id,
                    f"Google Drive ({file_id})",
                    downloaded / 1024 / 1024,
                    None,
                )
            except Exception:
                pass

    download_url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm=t"
    http_get(download_url, dest_path, expected_size=None, on_progress=on_progress)

    # Check if we got HTML (virus scan warning) instead of actual file
    with open(dest_path, "rb") as f:
        header = f.read(100)

    if not header.startswith((b"<!DOCTYPE", b"<html")):
        return  # Got actual file content

    # Handle virus scan warning - need to extract real download URL from HTML form
    logger.info("Handling Google Drive virus scan warning...")
    dest_path.unlink()

    html = httpx.get(
        f"https://drive.google.com/uc?export=download&id={file_id}",
        timeout=DOWNLOAD_TIMEOUT,
        follow_redirects=True,
    ).text

    download_url = _extract_google_drive_download_url(html, file_id)
    http_get(download_url, dest_path, expected_size=None, on_progress=on_progress)


def _extract_google_drive_download_url(html: str, file_id: str) -> str:
    """Extract the actual download URL from Google Drive's virus scan warning page."""
    # Try to extract form action and build URL from form params
    form_action = re.search(r'action="([^"]+)"', html)
    if form_action:
        params = {
            "id": _extract_form_value(html, "id") or file_id,
            "export": _extract_form_value(html, "export") or "download",
            "confirm": _extract_form_value(html, "confirm") or "t",
        }
        if uuid_val := _extract_form_value(html, "uuid"):
            params["uuid"] = uuid_val
        return (
            f"{form_action.group(1)}?{'&'.join(f'{k}={v}' for k, v in params.items())}"
        )

    # Fallback: try confirm token from URL
    if match := re.search(r"confirm=([a-zA-Z0-9_-]+)", html):
        return f"https://drive.google.com/uc?export=download&id={file_id}&confirm={match.group(1)}"

    # Last resort
    return f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"


def _extract_form_value(html: str, name: str) -> str | None:
    """Extract a form input value by name."""
    match = re.search(rf'name="{name}"\s+value="([^"]+)"', html)
    return match.group(1) if match else None


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
        pipeline_id=pipeline_id,
    )


def download_models(pipeline_id: str) -> None:
    """
    Download models for a specific pipeline.

    Args:
        pipeline_id: Pipeline ID to download models for.
    """
    from .artifact_registry import get_artifacts_for_pipeline

    models_root = ensure_models_dir()

    logger.info(f"Downloading models for pipeline: {pipeline_id}")
    artifacts = get_artifacts_for_pipeline(pipeline_id)

    if not artifacts:
        logger.warning(f"No artifacts defined for pipeline: {pipeline_id}")
        return

    # Download each artifact (progress tracking starts in set_download_context)
    for artifact in artifacts:
        download_artifact(artifact, models_root, pipeline_id)


def main():
    """Main entry point for the download_models script."""
    # Configure logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

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
