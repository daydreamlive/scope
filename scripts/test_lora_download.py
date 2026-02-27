"""Manual integration test for LoRA downloads from real sources.

Usage:
    uv run python scripts/test_lora_download.py

Requires the server to be running for the API endpoint test:
    uv run daydream-scope

Downloads real LoRAs from HuggingFace and CivitAI to a temp directory,
verifies manifest creation, tests the list API, then cleans up.

Note: CivitAI public models work without an API token.
"""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import patch

from scope.core.lora.manifest import load_manifest
from scope.server.lora_downloader import LoRADownloadRequest, download_lora


async def test_downloads():
    with tempfile.TemporaryDirectory() as tmpdir:
        lora_dir = Path(tmpdir)

        # HuggingFace
        print("=== HuggingFace Download ===")
        req = LoRADownloadRequest(
            source="huggingface",
            repo_id="shauray/Origami_WanLora",
            hf_filename="origami_000000500.safetensors",
        )
        result = await download_lora(req, lora_dir)
        print(f"  filename: {result.filename}")
        print(f"  sha256:   {result.sha256[:16]}...")
        print(f"  size:     {result.size_bytes / 1024 / 1024:.1f} MB")

        # CivitAI (public model, no token required)
        print("\n=== CivitAI Download (no token) ===")
        req = LoRADownloadRequest(
            source="civitai",
            version_id="2680702",
            model_id="2383884",
        )
        result = await download_lora(req, lora_dir)
        print(f"  filename: {result.filename}")
        print(f"  sha256:   {result.sha256[:16]}...")
        print(f"  size:     {result.size_bytes / 1024 / 1024:.1f} MB")

        # Verify manifest
        print("\n=== Manifest ===")
        manifest = load_manifest(lora_dir)
        for key, entry in manifest.entries.items():
            print(
                f"  {key}: source={entry.provenance.source}, sha256={entry.sha256[:16]}..."
            )

        assert len(manifest.entries) == 2, (
            f"Expected 2 entries, got {len(manifest.entries)}"
        )

        # Verify GET /api/v1/loras returns provenance data
        print("\n=== GET /api/v1/loras (via test client) ===")
        with patch("scope.server.app.get_lora_dir", return_value=lora_dir):
            from fastapi.testclient import TestClient

            from scope.server.app import app

            client = TestClient(app)
            resp = client.get("/api/v1/loras")
            assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
            data = resp.json()
            lora_files = data["lora_files"]
            print(f"  returned {len(lora_files)} files")
            for f in lora_files:
                print(
                    f"  {f['name']}: sha256={f['sha256'][:16] if f['sha256'] else None}..., "
                    f"provenance={f['provenance']['source'] if f['provenance'] else None}"
                )
                assert f["sha256"] is not None, f"Missing sha256 for {f['name']}"
                assert f["provenance"] is not None, (
                    f"Missing provenance for {f['name']}"
                )

        print("\nAll tests passed!")


if __name__ == "__main__":
    asyncio.run(test_downloads())
