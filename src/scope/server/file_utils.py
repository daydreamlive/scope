"""Filesystem utilities for directory traversal with symlink support."""

import logging
import os
from collections.abc import Iterator
from pathlib import Path

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
LORA_EXTENSIONS = {".safetensors"}


def iter_files(
    directory: Path,
    extensions: set[str],
    *,
    follow_symlinks: bool = True,
) -> Iterator[Path]:
    """Walk *directory* yielding files whose suffix is in *extensions*.

    Uses :func:`os.walk` with ``followlinks=follow_symlinks`` so that
    symlinked directories are traversed by default.  This is important on
    Windows where ``pathlib.rglob`` does not follow symlinks.

    Broken symlinks and inaccessible files are silently skipped.
    """
    if not directory.is_dir():
        return

    for root, _dirs, files in os.walk(directory, followlinks=follow_symlinks):
        root_path = Path(root)
        for filename in files:
            if Path(filename).suffix.lower() not in extensions:
                continue
            file_path = root_path / filename
            try:
                if file_path.is_file():
                    yield file_path
            except (OSError, PermissionError) as e:
                logger.debug(
                    "iter_files: skipping inaccessible file %s: %s", file_path, e
                )
