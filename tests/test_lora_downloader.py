import asyncio
from pathlib import Path
from unittest.mock import patch

import pytest

from scope.core.lora.manifest import load_manifest
from scope.server.lora_downloader import (
    LoRADownloadRequest,
    _filename_from_url,
    _resolve_hf_url,
    download_lora,
)


def test_filename_from_url():
    assert (
        _filename_from_url("https://example.com/path/model.safetensors")
        == "model.safetensors"
    )


def test_filename_from_url_no_extension():
    with pytest.raises(ValueError, match="Cannot determine filename"):
        _filename_from_url("https://example.com/path/noextension")


def test_resolve_hf_url():
    with patch("huggingface_hub.hf_hub_url", return_value="https://hf.co/file") as mock:
        result = _resolve_hf_url("user/repo", "model.safetensors")
        assert result == "https://hf.co/file"
        mock.assert_called_once_with("user/repo", "model.safetensors")


def test_download_lora_hf(tmp_path: Path):
    request = LoRADownloadRequest(
        source="huggingface",
        repo_id="user/repo",
        hf_filename="model.safetensors",
    )

    def fake_http_get(url, dest_path, **kwargs):
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(b"fake model data")

    with (
        patch(
            "scope.server.lora_downloader._resolve_hf_url",
            return_value="https://hf.co/file",
        ),
        patch("scope.server.lora_downloader.http_get", side_effect=fake_http_get),
    ):
        result = asyncio.run(download_lora(request, tmp_path))

    assert result.filename == "model.safetensors"
    assert result.size_bytes == len(b"fake model data")
    assert result.sha256  # not empty

    manifest = load_manifest(tmp_path)
    assert "model.safetensors" in manifest.entries
    assert manifest.entries["model.safetensors"].provenance.source == "huggingface"


def test_download_lora_sha256_mismatch(tmp_path: Path):
    request = LoRADownloadRequest(
        source="huggingface",
        repo_id="user/repo",
        hf_filename="model.safetensors",
        expected_sha256="wrong_hash",
    )

    def fake_http_get(url, dest_path, **kwargs):
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(b"fake model data")

    with (
        patch(
            "scope.server.lora_downloader._resolve_hf_url",
            return_value="https://hf.co/file",
        ),
        patch("scope.server.lora_downloader.http_get", side_effect=fake_http_get),
    ):
        with pytest.raises(ValueError, match="SHA256 mismatch"):
            asyncio.run(download_lora(request, tmp_path))


def test_download_lora_civitai(tmp_path: Path):
    request = LoRADownloadRequest(
        source="civitai",
        version_id="12345",
        model_id="999",
    )

    def fake_http_get(url, dest_path, **kwargs):
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(b"civitai data")

    with (
        patch(
            "scope.server.lora_downloader._resolve_civitai_download_url",
            return_value=("https://civitai.com/dl/12345", "style.safetensors"),
        ),
        patch("scope.server.lora_downloader.http_get", side_effect=fake_http_get),
    ):
        result = asyncio.run(download_lora(request, tmp_path))

    assert result.filename == "style.safetensors"
    manifest = load_manifest(tmp_path)
    assert manifest.entries["style.safetensors"].provenance.source == "civitai"


def test_download_lora_url(tmp_path: Path):
    request = LoRADownloadRequest(
        source="url",
        url="https://example.com/lora.safetensors",
    )

    def fake_http_get(url, dest_path, **kwargs):
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(b"url data")

    with patch("scope.server.lora_downloader.http_get", side_effect=fake_http_get):
        result = asyncio.run(download_lora(request, tmp_path))

    assert result.filename == "lora.safetensors"
    manifest = load_manifest(tmp_path)
    assert (
        manifest.entries["lora.safetensors"].provenance.url
        == "https://example.com/lora.safetensors"
    )


def test_download_lora_with_subfolder(tmp_path: Path):
    request = LoRADownloadRequest(
        source="huggingface",
        repo_id="user/repo",
        hf_filename="model.safetensors",
        subfolder="anime",
    )

    def fake_http_get(url, dest_path, **kwargs):
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(b"data")

    with (
        patch(
            "scope.server.lora_downloader._resolve_hf_url",
            return_value="https://hf.co/file",
        ),
        patch("scope.server.lora_downloader.http_get", side_effect=fake_http_get),
    ):
        result = asyncio.run(download_lora(request, tmp_path))

    # On Windows the path separator is \, normalize for comparison
    assert Path(result.filename) == Path("anime/model.safetensors")


def test_download_lora_missing_params():
    with pytest.raises(ValueError, match="repo_id and hf_filename required"):
        asyncio.run(
            download_lora(LoRADownloadRequest(source="huggingface"), Path("/tmp"))
        )

    with pytest.raises(ValueError, match="version_id required"):
        asyncio.run(download_lora(LoRADownloadRequest(source="civitai"), Path("/tmp")))

    with pytest.raises(ValueError, match="url required"):
        asyncio.run(download_lora(LoRADownloadRequest(source="url"), Path("/tmp")))
