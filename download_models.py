"""
Cross-platform model downloader using huggingface_hub for HF repo/files.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Disable hf_transfer to use standard download method
# This prevents errors when HF_HUB_ENABLE_HF_TRANSFER=1 is set but hf_transfer is not installed
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

from lib.models_config import (
    ensure_models_dir,
    models_are_downloaded,
)

# Set up logger
logger = logging.getLogger(__name__)

# --- third-party libs ---
try:
    from huggingface_hub import hf_hub_download, snapshot_download
except Exception:
    print(
        "Error: huggingface_hub is required. Install with: pip install huggingface_hub",
        file=sys.stderr,
    )
    raise


def download_hf_repo_excluding(
    repo_id: str, local_dir: Path, ignore_patterns: list[str]
) -> None:
    """
    Download an entire HF repo snapshot while excluding specific files.
    """
    local_dir.mkdir(parents=True, exist_ok=True)
    # snapshot_download supports exclude via `ignore_patterns`
    # (patterns are glob-like, relative to the repo root)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,  # copy files for portability
        ignore_patterns=ignore_patterns,
        # token is picked up automatically from HUGGINGFACE_TOKEN if set
        # revision=None,  # optionally pin a commit/tag if you like
    )
    print(f"[OK] Downloaded repo '{repo_id}' to: {local_dir}")


def download_hf_single_file(repo_id: str, filename: str, local_dir: Path) -> None:
    """
    Download a single file from an HF repo into a target folder.
    """
    local_dir.mkdir(parents=True, exist_ok=True)
    out_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
    )
    print(f"[OK] Downloaded file '{filename}' from '{repo_id}' to: {out_path}")


def download_required_models():
    """Download required models if they are not already present."""
    if models_are_downloaded():
        logger.info("Models already downloaded, skipping download")
        return

    logger.info("Downloading required models...")
    try:
        download_models()
        logger.info("Model download completed successfully")
    except Exception as e:
        logger.error(f"Error downloading models: {e}")
        raise


def download_streamdiffusionv2_pipeline() -> None:
    """Download models for the StreamDiffusionV2 pipeline."""
    wan_video_repo = "Wan-AI/Wan2.1-T2V-1.3B"
    wan_video_comfy_repo = "Kijai/WanVideo_comfy"
    wan_video_comfy_file = "umt5-xxl-enc-fp8_e4m3fn.safetensors"
    stream_diffusion_repo = "jerryfeng/StreamDiffusionV2"

    # Ensure models directory exists and get paths
    models_root = ensure_models_dir()
    wan_video_dst = models_root / "Wan2.1-T2V-1.3B"
    wan_video_comfy_dst = models_root / "WanVideo_comfy"
    stream_diffusion_dst = models_root / "StreamDiffusionV2"

    # 1) HF repo download excluding a large file
    wan_video_exclude = ["models_t5_umt5-xxl-enc-bf16.pth"]
    download_hf_repo_excluding(
        wan_video_repo, wan_video_dst, ignore_patterns=wan_video_exclude
    )

    # 2) HF single file download into a folder
    download_hf_single_file(
        wan_video_comfy_repo, wan_video_comfy_file, wan_video_comfy_dst
    )

    # 3) HF repo download for StreamDiffusionV2 (1.3b only)
    snapshot_download(
        repo_id=stream_diffusion_repo,
        local_dir=stream_diffusion_dst,
        allow_patterns=["wan_causal_dmd_v2v/model.pt"],
    )


def download_longlive_pipeline() -> None:
    """Download models for the LongLive pipeline."""
    wan_video_repo = "Wan-AI/Wan2.1-T2V-1.3B"
    wan_video_comfy_repo = "Kijai/WanVideo_comfy"
    wan_video_comfy_file = "umt5-xxl-enc-fp8_e4m3fn.safetensors"
    longlive_repo = "Efficient-Large-Model/LongLive-1.3B"

    # Ensure models directory exists and get paths
    models_root = ensure_models_dir()
    wan_video_dst = models_root / "Wan2.1-T2V-1.3B"
    wan_video_comfy_dst = models_root / "WanVideo_comfy"
    longlive_dst = models_root / "LongLive-1.3B"

    # 1) HF repo download for Wan2.1-T2V-1.3B, excluding large file
    wan_video_exclude = ["models_t5_umt5-xxl-enc-bf16.pth"]
    download_hf_repo_excluding(
        wan_video_repo, wan_video_dst, ignore_patterns=wan_video_exclude
    )

    # 2) HF single file download for UMT5 encoder
    download_hf_single_file(
        wan_video_comfy_repo, wan_video_comfy_file, wan_video_comfy_dst
    )

    # 3) HF repo download for LongLive-1.3B
    download_hf_repo_excluding(longlive_repo, longlive_dst, ignore_patterns=[])


def download_krea_realtime_video_pipeline() -> None:
    """
    Download models for the krea-realtime-video pipeline.
    """
    # HuggingFace repos
    krea_rt_repo = "krea/krea-realtime-video"
    wan_video_1_3b_repo = "Wan-AI/Wan2.1-T2V-1.3B"
    wan_video_14b_repo = "Wan-AI/Wan2.1-T2V-14B"
    wan_video_comfy_repo = "Kijai/WanVideo_comfy"
    wan_video_comfy_file = "umt5-xxl-enc-fp8_e4m3fn.safetensors"

    # Ensure models directory exists and get paths
    models_root = ensure_models_dir()
    krea_rt_dst = models_root / "krea-realtime-video"
    wan_video_1_3b_dst = models_root / "Wan2.1-T2V-1.3B"
    wan_video_14b_dst = models_root / "Wan2.1-T2V-14B"
    wan_video_comfy_dst = models_root / "WanVideo_comfy"

    # 1) Download only krea-realtime-video
    snapshot_download(
        repo_id=krea_rt_repo,
        local_dir=krea_rt_dst,
        allow_patterns=["krea-realtime-video-14b.safetensors"],
    )

    # 2) Download VAE and text encoder from Wan2.1-T2V-1.3B
    wan_video_exclude = ["models_t5_umt5-xxl-enc-bf16.pth"]
    download_hf_repo_excluding(
        wan_video_1_3b_repo, wan_video_1_3b_dst, ignore_patterns=wan_video_exclude
    )

    # 3) Download only config.json from Wan2.1-T2V-14B (no model weights needed)
    snapshot_download(
        repo_id=wan_video_14b_repo,
        local_dir=wan_video_14b_dst,
        allow_patterns=["config.json"],
    )

    # 4) HF single file download for UMT5 encoder
    download_hf_single_file(
        wan_video_comfy_repo, wan_video_comfy_file, wan_video_comfy_dst
    )


def download_models(pipeline_id: str | None = None) -> None:
    """
    Download models. If pipeline_id is None, downloads all pipelines.
    If pipeline_id is specified, downloads that specific pipeline.

    Args:
        pipeline_id: Optional pipeline ID. Supports "streamdiffusionv2" or "longlive".
                    If None, downloads all pipelines.
    """
    if pipeline_id is None:
        # Download all pipelines
        download_streamdiffusionv2_pipeline()
        download_longlive_pipeline()
    elif pipeline_id == "streamdiffusionv2":
        download_streamdiffusionv2_pipeline()
    elif pipeline_id == "longlive":
        download_longlive_pipeline()
    elif pipeline_id == "krea-realtime-video":
        download_krea_realtime_video_pipeline()
    else:
        raise ValueError(
            f"Unknown pipeline: {pipeline_id}. Supported pipelines: streamdiffusionv2, longlive, krea-realtime-video"
        )

    print("\nAll downloads complete.")


def main():
    """Main entry point for the download_models script."""
    parser = argparse.ArgumentParser(
        description="Download models for Daydream Scope pipelines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all pipelines
  python download_models.py

  # Download specific pipeline
  python download_models.py --pipeline streamdiffusionv2
  python download_models.py --pipeline longlive
  python download_models.py -p streamdiffusionv2
        """,
    )
    parser.add_argument(
        "--pipeline",
        "-p",
        type=str,
        default=None,
        help="Pipeline ID to download (e.g., 'streamdiffusionv2', 'longlive'). If not specified, downloads all pipelines.",
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
