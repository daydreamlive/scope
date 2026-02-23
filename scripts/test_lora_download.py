"""Manual integration test for LoRA downloads from real sources.

Usage:
    uv run python scripts/test_lora_download.py

Downloads real LoRAs from HuggingFace and CivitAI to a temp directory,
verifies manifest creation, then cleans up.
"""

import asyncio
import tempfile
from pathlib import Path

from scope.core.lora.manifest import load_manifest
from scope.server.lora_downloader import LoRADownloadRequest, download_lora


async def main():
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

        # CivitAI
        print("\n=== CivitAI Download ===")
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
        print("\nAll downloads successful!")


if __name__ == "__main__":
    asyncio.run(main())
