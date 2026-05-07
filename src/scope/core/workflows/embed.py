"""Embed referenced media files into a workflow JSON and extract them back.

A shareable workflow file references media (reference images, audio, video) by
filesystem path. Those paths only resolve on the machine that exported the
workflow, so an importer ends up missing the assets the workflow author saw.

These helpers walk the workflow JSON, base64-encode every existing media file
they find into a top-level ``embedded_assets`` list, and on import write each
embedded asset to the local assets directory and rewrite path references by
basename so they resolve again.

The functions operate on plain ``dict``/``list`` JSON values to avoid coupling
to either the client-side schema or :mod:`scope.core.workflows.resolve`'s
strictly-typed models. ``WorkflowRequest`` already uses ``extra="ignore"``,
so a workflow with ``embedded_assets`` resolves fine without further changes.
"""

from __future__ import annotations

import base64
import hashlib
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Mirrors scope.server.file_utils to avoid importing server-side modules from
# core. Keeping the lists in sync is a documented invariant — both files name
# the same set of extensions and ship together.
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg"}
MEDIA_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS | AUDIO_EXTENSIONS

CURRENT_FORMAT_VERSION = "1.1"

# Cap on basename-collision rename attempts before we give up. Hitting more
# than a handful means something is structurally wrong with the assets dir.
_MAX_RENAME_ATTEMPTS = 1000

_MIME_BY_EXT: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".mp4": "video/mp4",
    ".avi": "video/x-msvideo",
    ".mov": "video/quicktime",
    ".mkv": "video/x-matroska",
    ".webm": "video/webm",
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
}


def _normalize_separators(value: str) -> str:
    """Map Windows-style backslash separators to forward slashes so a single
    POSIX-style ``Path`` can extract a basename regardless of the OS that
    wrote the workflow. Only converts ``\\`` between path-like segments; leaves
    free-form strings alone since the caller filters those out first."""
    return value.replace("\\", "/")


def _basename(value: str) -> str:
    """Cross-OS basename: split on either separator and return the trailing
    component. ``Path(value).name`` on POSIX would treat backslashes as
    regular characters, which breaks Windows-exported workflows."""
    return Path(_normalize_separators(value)).name


def _looks_path_shaped(value: str, known_filenames: set[str]) -> bool:
    """Stricter than ``_has_media_extension``: require the string to actually
    look like a path. Either it contains a path separator (so it's clearly
    addressing a file location), or its full content exactly equals a known
    embedded filename (so a bare ``ref.png`` reference still gets rewritten).

    This guards against rewriting prompt strings or other free-form text that
    happens to end with a media extension (``"… see attached cat.png"``).
    """
    if "/" in value or "\\" in value:
        return True
    return value in known_filenames


def _has_media_extension(value: Any) -> bool:
    """A string carries a media extension if its last path component ends in
    one of our known media suffixes and it isn't a URL/data scheme."""
    if not isinstance(value, str) or not value:
        return False
    if value.startswith(("http://", "https://", "blob:", "data:")):
        return False
    suffix = Path(_normalize_separators(value)).suffix.lower()
    return suffix in MEDIA_EXTENSIONS


def _walk_strings(obj: Any, visit: Any) -> Any:
    """Recursively walk a JSON-like value, replacing each string with
    ``visit(string)`` (must return a string). Mutates a copy, leaves input
    alone."""
    if isinstance(obj, str):
        return visit(obj)
    if isinstance(obj, list):
        return [_walk_strings(x, visit) for x in obj]
    if isinstance(obj, dict):
        return {k: _walk_strings(v, visit) for k, v in obj.items()}
    return obj


def _resolve_media_path(value: str, assets_dir: Path) -> Path | None:
    """Map a workflow path string to an existing file on disk.

    Tries the literal path first, then falls back to looking up the basename
    in ``assets_dir`` (handles workflows whose absolute paths point at a
    different machine, including Windows-style paths viewed from POSIX).
    """
    candidate = Path(value)
    if candidate.is_absolute() and candidate.is_file():
        return candidate

    # Either the absolute path doesn't exist locally, or the value is
    # relative / Windows-style. Fall back to basename lookup in assets_dir.
    base = _basename(value)
    if base:
        base_match = assets_dir / base
        if base_match.is_file():
            return base_match

    if not candidate.is_absolute():
        rel_match = assets_dir / value
        if rel_match.is_file():
            return rel_match

    return None


def _next_higher_version(current: Any, minimum: str) -> str:
    """Return the higher of *current* and *minimum* using a tuple-of-ints
    compare. When *current* is missing or unparseable we substitute *minimum*
    so embedded files always carry a version we recognize."""
    if not isinstance(current, str):
        return minimum
    try:
        cur_parts = tuple(int(p) for p in current.split("."))
        min_parts = tuple(int(p) for p in minimum.split("."))
    except ValueError:
        return minimum
    return current if cur_parts >= min_parts else minimum


def embed_workflow_assets(workflow: dict[str, Any], assets_dir: Path) -> dict[str, Any]:
    """Walk *workflow*, base64-encode every referenced media file that exists
    on disk into a new top-level ``embedded_assets`` list, and bump
    ``format_version`` to ``"1.1"`` (preserving any higher value already set).

    Returns a new dict; *workflow* is not mutated.

    Embedded entry shape::

        {
          "filename": "ref.png",
          "mime_type": "image/png",
          "size_bytes": 1234,
          "sha256": "<hex>",
          "data": "<base64>"
        }

    Multiple references to the same file (by basename) collapse to one entry.
    Paths whose target file we can't locate are left as-is — the importer
    will see them and report a missing-asset condition like before.
    """
    embedded: dict[str, dict[str, Any]] = {}

    def visit(value: str) -> str:
        if not _has_media_extension(value):
            return value
        resolved = _resolve_media_path(value, assets_dir)
        if resolved is None:
            return value
        filename = resolved.name
        if filename in embedded:
            return value
        try:
            data = resolved.read_bytes()
        except OSError as exc:
            logger.warning("embed: cannot read %s: %s", resolved, exc)
            return value
        embedded[filename] = {
            "filename": filename,
            "mime_type": _MIME_BY_EXT.get(
                resolved.suffix.lower(), "application/octet-stream"
            ),
            "size_bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "data": base64.b64encode(data).decode("ascii"),
        }
        return value

    rewritten = _walk_strings(workflow, visit)

    if not embedded:
        return rewritten

    result = dict(rewritten)
    # Preserve any pre-existing embedded_assets a caller already attached
    # (e.g. partial embeds), keyed by filename so we don't double up.
    existing = result.get("embedded_assets")
    if isinstance(existing, list):
        for entry in existing:
            if isinstance(entry, dict) and isinstance(entry.get("filename"), str):
                embedded.setdefault(entry["filename"], entry)
    result["embedded_assets"] = list(embedded.values())
    result["format_version"] = _next_higher_version(
        result.get("format_version"), CURRENT_FORMAT_VERSION
    )
    return result


def extract_workflow_assets(
    workflow: dict[str, Any], assets_dir: Path
) -> dict[str, Any]:
    """Write each ``embedded_assets`` entry to *assets_dir*, then rewrite
    every path-shaped reference in *workflow* whose basename matches an
    embedded filename to point at the resulting on-disk path. Strips
    ``embedded_assets`` from the returned workflow.

    Path rewriting only fires for strings that look path-shaped (contain a
    separator, or exactly match an embedded filename). Free-form strings that
    happen to end with a media extension — e.g. a prompt mentioning
    ``cat.png`` — are left alone.

    Idempotent: if an existing file with matching SHA-256 is already present
    we reuse it. If a different file with the same basename exists we save
    under a deduplicated name (``foo_imported1.png``) and rewrite paths to
    that new name.

    Returns a new dict; *workflow* is not mutated.
    """
    embedded = workflow.get("embedded_assets")
    if not isinstance(embedded, list) or not embedded:
        return workflow

    assets_dir.mkdir(parents=True, exist_ok=True)
    assets_dir_resolved = assets_dir.resolve()

    def _confine(target: Path, label: str) -> Path | None:
        """Reject paths that escape assets_dir after symlink resolution."""
        resolved = target.resolve()
        if not resolved.is_relative_to(assets_dir_resolved):
            logger.warning("extract: refusing to write outside assets dir: %s", label)
            return None
        return resolved

    written: dict[str, str] = {}  # original filename -> resolved path
    for entry in embedded:
        if not isinstance(entry, dict):
            continue
        filename = entry.get("filename")
        data_b64 = entry.get("data")
        expected_sha = entry.get("sha256")
        if not isinstance(filename, str) or not isinstance(data_b64, str):
            logger.warning("extract: skipping malformed embedded asset entry")
            continue
        try:
            data = base64.b64decode(data_b64)
        except (ValueError, TypeError) as exc:
            logger.warning("extract: invalid base64 for %s: %s", filename, exc)
            continue

        target = _confine(assets_dir / filename, filename)
        if target is None:
            continue

        if target.exists():
            try:
                existing_sha = hashlib.sha256(target.read_bytes()).hexdigest()
            except OSError:
                existing_sha = None
            if existing_sha and existing_sha == expected_sha:
                written[filename] = str(target)
                continue
            # Same name, different content — pick a non-conflicting name and
            # re-validate confinement on every attempt so the safety check is
            # local to each write.
            stem, suffix = target.stem, target.suffix
            counter = 1
            while True:
                candidate = _confine(
                    assets_dir / f"{stem}_imported{counter}{suffix}",
                    f"{filename} (rename #{counter})",
                )
                if candidate is None:
                    target = None
                    break
                if not candidate.exists():
                    target = candidate
                    break
                counter += 1
                if counter > _MAX_RENAME_ATTEMPTS:
                    raise RuntimeError(
                        f"Could not find non-conflicting name for {filename}"
                    )
            if target is None:
                continue

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        written[filename] = str(target)

    if not written:
        result = dict(workflow)
        result.pop("embedded_assets", None)
        return result

    known_filenames = set(written.keys())

    def visit(value: str) -> str:
        if not _has_media_extension(value):
            return value
        if not _looks_path_shaped(value, known_filenames):
            return value
        basename = _basename(value)
        if basename in written:
            return written[basename]
        return value

    stripped = {k: v for k, v in workflow.items() if k != "embedded_assets"}
    return _walk_strings(stripped, visit)
