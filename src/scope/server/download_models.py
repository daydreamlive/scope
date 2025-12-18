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
# Disable xet for now because it seems to sometimes causes an issue with exiting the server after a download
# This has not been investigated thoroughly, but disabling it seems to fix the issue for now
os.environ["HF_HUB_ENABLE_HF_XET"] = "1"

from .models_config import (
    ensure_models_dir,
    models_are_downloaded,
)

# Set up logger
logger = logging.getLogger(__name__)

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
    try:
        if self.n is not None and self.total is not None and self.total > 0:
            current_progress = (self.n / self.total) * 100

            # Skip logging at 0% progress
            if current_progress == 0.0:
                return _original_tqdm_update(self, n)

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
    except KeyboardInterrupt:
        # Re-raise KeyboardInterrupt to allow proper signal handling
        raise
    except Exception:
        # Don't let logging errors interfere with tqdm or signal handling
        pass

    # Always call original update, even if our logging failed
    return _original_tqdm_update(self, n)


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


def download_hf_repo_excluding(
    repo_id: str, local_dir: Path, ignore_patterns: list[str]
) -> None:
    """
    Download an entire HF repo snapshot while excluding specific files.
    This is a convenience wrapper around download_hf_repo.
    """
    download_hf_repo(repo_id, local_dir, ignore_patterns=ignore_patterns)


def download_hf_single_file(repo_id: str, filename: str, local_dir: Path) -> None:
    """
    Download a single file from an HF repo into a target folder.
    This is a convenience wrapper around download_hf_repo.
    """
    download_hf_repo(repo_id, local_dir, filename=filename)


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

    # 1) HF repo download for Wan2.1-T2V-1.3B VAE + config
    wan_video_exclude = [
        "models_t5_umt5-xxl-enc-bf16.pth",
        "diffusion_pytorch_model.safetensors",
    ]
    download_hf_repo_excluding(
        wan_video_repo, wan_video_dst, ignore_patterns=wan_video_exclude
    )

    # 2) HF single file download into a folder
    download_hf_single_file(
        wan_video_comfy_repo, wan_video_comfy_file, wan_video_comfy_dst
    )

    # 3) HF repo download for StreamDiffusionV2 (1.3b only)
    download_hf_repo(
        stream_diffusion_repo,
        stream_diffusion_dst,
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

    # 1) HF repo download for Wan2.1-T2V-1.3B VAE + config
    wan_video_exclude = [
        "models_t5_umt5-xxl-enc-bf16.pth",
        "diffusion_pytorch_model.safetensors",
    ]
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
    download_hf_repo(
        krea_rt_repo,
        krea_rt_dst,
        allow_patterns=["krea-realtime-video-14b.safetensors"],
    )

    # 2) HF repo download for Wan2.1-T2V-1.3B VAE + config
    wan_video_exclude = [
        "models_t5_umt5-xxl-enc-bf16.pth",
        "diffusion_pytorch_model.safetensors",
    ]
    download_hf_repo_excluding(
        wan_video_1_3b_repo, wan_video_1_3b_dst, ignore_patterns=wan_video_exclude
    )

    # 3) Download only config.json from Wan2.1-T2V-14B (no model weights needed)
    download_hf_repo(
        wan_video_14b_repo, wan_video_14b_dst, allow_patterns=["config.json"]
    )

    # 4) HF single file download for UMT5 encoder
    download_hf_single_file(
        wan_video_comfy_repo, wan_video_comfy_file, wan_video_comfy_dst
    )


def download_reward_forcing_pipeline() -> None:
    """
    Download models for the RewardForcing pipeline.
    """
    # HuggingFace repos
    reward_forcing_repo = "JaydenLu666/Reward-Forcing-T2V-1.3B"
    wan_video_repo = "Wan-AI/Wan2.1-T2V-1.3B"
    wan_video_comfy_repo = "Kijai/WanVideo_comfy"
    wan_video_comfy_file = "umt5-xxl-enc-fp8_e4m3fn.safetensors"

    # Ensure models directory exists and get paths
    models_root = ensure_models_dir()
    reward_forcing_dst = models_root / "Reward-Forcing-T2V-1.3B"
    wan_video_dst = models_root / "Wan2.1-T2V-1.3B"
    wan_video_comfy_dst = models_root / "WanVideo_comfy"

    # 1) Download Reward-Forcing model
    download_hf_repo(
        reward_forcing_repo, reward_forcing_dst, allow_patterns=["rewardforcing.pt"]
    )

    # 2) HF repo download for Wan2.1-T2V-1.3B VAE + config
    wan_video_exclude = [
        "models_t5_umt5-xxl-enc-bf16.pth",
        "diffusion_pytorch_model.safetensors",
    ]
    download_hf_repo_excluding(
        wan_video_repo, wan_video_dst, ignore_patterns=wan_video_exclude
    )

    # 3) HF single file download for UMT5 encoder
    download_hf_single_file(
        wan_video_comfy_repo, wan_video_comfy_file, wan_video_comfy_dst
    )


def download_models(pipeline_id: str | None = None) -> None:
    """
    Download models. If pipeline_id is None, downloads all pipelines.
    If pipeline_id is specified, downloads that specific pipeline.

    Args:
        pipeline_id: Optional pipeline ID. Supports "streamdiffusionv2", "longlive",
                    "krea-realtime-video", or "reward-forcing".
                    If None, downloads all pipelines.
    """
    if pipeline_id is None:
        # Download all pipelines
        download_streamdiffusionv2_pipeline()
        download_longlive_pipeline()
        download_krea_realtime_video_pipeline()
        download_reward_forcing_pipeline()
    elif pipeline_id == "streamdiffusionv2":
        download_streamdiffusionv2_pipeline()
    elif pipeline_id == "longlive":
        download_longlive_pipeline()
    elif pipeline_id == "krea-realtime-video":
        download_krea_realtime_video_pipeline()
    elif pipeline_id == "reward-forcing":
        download_reward_forcing_pipeline()
    else:
        raise ValueError(
            f"Unknown pipeline: {pipeline_id}. Supported pipelines: streamdiffusionv2, longlive, krea-realtime-video, reward-forcing"
        )

    logger.info("All downloads complete.")


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
        help="Pipeline ID to download (e.g., 'streamdiffusionv2', 'longlive', 'krea-realtime-video', 'reward-forcing'). If not specified, downloads all pipelines.",
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
