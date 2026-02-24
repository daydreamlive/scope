import asyncio
import logging
from pathlib import Path
from typing import Literal
from urllib.parse import unquote, urlparse

from pydantic import BaseModel

from scope.core.lora.manifest import (
    LoRAProvenance,
    add_manifest_entry,
    compute_sha256,
)

from .download_models import http_get

logger = logging.getLogger(__name__)


class LoRADownloadRequest(BaseModel):
    source: Literal["huggingface", "civitai", "url"]
    repo_id: str | None = None
    hf_filename: str | None = None
    model_id: str | None = None
    version_id: str | None = None
    url: str | None = None
    subfolder: str | None = None
    expected_sha256: str | None = None


class LoRADownloadResult(BaseModel):
    filename: str
    path: str
    sha256: str
    size_bytes: int


def _resolve_hf_url(repo_id: str, hf_filename: str) -> str:
    from huggingface_hub import hf_hub_url

    return hf_hub_url(repo_id, hf_filename)


def _resolve_civitai_download_url(version_id: str) -> tuple[str, str | None]:
    """Resolve CivitAI version ID to download URL. Returns (url, filename)."""
    import httpx

    api_url = f"https://civitai.com/api/v1/model-versions/{version_id}"
    headers = {}
    from .models_config import get_civitai_token

    token = get_civitai_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"

    response = httpx.get(api_url, headers=headers, timeout=30.0, follow_redirects=True)
    response.raise_for_status()
    data = response.json()

    files = data.get("files", [])
    if not files:
        raise ValueError(f"No files found for CivitAI version {version_id}")

    primary = files[0]
    download_url = primary.get("downloadUrl")
    if not download_url:
        raise ValueError(f"No download URL for CivitAI version {version_id}")

    # Add token to download URL if available
    if token:
        separator = "&" if "?" in download_url else "?"
        download_url = f"{download_url}{separator}token={token}"

    filename = primary.get("name")
    return download_url, filename


def _filename_from_url(url: str) -> str | None:
    """Try to extract a filename from the URL path. Returns None if not possible."""
    parsed = urlparse(url)
    path_part = unquote(parsed.path.split("/")[-1])
    if path_part and "." in path_part:
        return path_part
    return None


def _filename_from_response(url: str) -> str:
    """Resolve filename by making a HEAD/streaming request and reading Content-Disposition."""
    import re

    import httpx

    with httpx.Client(follow_redirects=True, timeout=30.0) as client:
        with client.stream("GET", url) as response:
            response.raise_for_status()
            # Try Content-Disposition header
            content_disp = response.headers.get("content-disposition", "")
            match = re.search(r'filename[*]?=["\']?([^"\';]+)["\']?', content_disp)
            if match:
                return unquote(match.group(1).strip())
            # Try the final URL after redirects
            name = _filename_from_url(str(response.url))
            if name:
                return name
    raise ValueError(f"Cannot determine filename from URL: {url}")


async def download_lora(
    request: LoRADownloadRequest, lora_dir: Path
) -> LoRADownloadResult:
    loop = asyncio.get_running_loop()

    if request.source == "huggingface":
        if not request.repo_id or not request.hf_filename:
            raise ValueError("repo_id and hf_filename required for HuggingFace source")
        url = _resolve_hf_url(request.repo_id, request.hf_filename)
        filename = request.hf_filename.split("/")[-1]

    elif request.source == "civitai":
        if not request.version_id:
            raise ValueError("version_id required for CivitAI source")
        url, civitai_filename = await loop.run_in_executor(
            None, _resolve_civitai_download_url, request.version_id
        )
        filename = civitai_filename  # may be None, resolved below

    elif request.source == "url":
        if not request.url:
            raise ValueError("url required for URL source")
        url = request.url
        filename = _filename_from_url(url)  # may be None, resolved below

    else:
        raise ValueError(f"Unknown source: {request.source}")

    # Resolve filename from the actual download response if we don't have one yet
    if not filename:
        filename = await loop.run_in_executor(None, _filename_from_response, url)

    # Build destination path
    if request.subfolder:
        dest_dir = lora_dir / request.subfolder
    else:
        dest_dir = lora_dir
    dest_path = dest_dir / filename

    # Download
    await loop.run_in_executor(None, http_get, url, dest_path)

    # Compute hash and verify
    sha256 = await loop.run_in_executor(None, compute_sha256, dest_path)
    if request.expected_sha256 and sha256 != request.expected_sha256:
        dest_path.unlink(missing_ok=True)
        raise ValueError(
            f"SHA256 mismatch: expected {request.expected_sha256}, got {sha256}"
        )

    size_bytes = dest_path.stat().st_size
    relative_filename = dest_path.relative_to(lora_dir).as_posix()

    # Update manifest
    provenance = LoRAProvenance(
        source=request.source,
        repo_id=request.repo_id,
        hf_filename=request.hf_filename,
        model_id=request.model_id,
        version_id=request.version_id,
        url=request.url if request.source == "url" else None,
    )
    add_manifest_entry(lora_dir, relative_filename, provenance, sha256, size_bytes)

    return LoRADownloadResult(
        filename=relative_filename,
        path=str(dest_path),
        sha256=sha256,
        size_bytes=size_bytes,
    )
