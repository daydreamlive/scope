import asyncio
import copy
import logging
import os
import queue
import threading
import time
import uuid
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from scope.realtime import (
    CompiledPrompt,
    StyleManifest,
    StyleRegistry,
    TemplateCompiler,
    WorldState,
    create_compiler,
)

try:
    from aiortc.mediastreams import VideoFrame
except ImportError:  # pragma: no cover
    VideoFrame = Any  # type: ignore[misc,assignment]

from scope.realtime.control_bus import ControlBus, EventType

from .models_config import get_model_file_path
from .pipeline_manager import PipelineManager, PipelineNotAvailableException
from .session_recorder import SessionRecorder

logger = logging.getLogger(__name__)


def _is_env_true(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


# Multiply the # of output frames from pipeline by this to get the max size of the output queue
OUTPUT_QUEUE_MAX_SIZE_FACTOR = 3

# FPS calculation constants
MIN_FPS = 1.0  # Minimum FPS to prevent division by zero
MAX_FPS = 60.0  # Maximum FPS cap
DEFAULT_FPS = 30.0  # Default FPS
SLEEP_TIME = 0.01

# Input FPS measurement constants
INPUT_FPS_SAMPLE_SIZE = 30  # Number of frame intervals to track
INPUT_FPS_MIN_SAMPLES = 5  # Minimum samples needed before using input FPS

# Snapshot constants
MAX_SNAPSHOTS = 10  # Maximum number of snapshots to keep (LRU eviction)

# Continuity keys from pipeline.state that define generation continuity
CONTINUITY_KEYS = [
    "current_start_frame",
    "first_context_frame",
    "context_frame_buffer",
    "decoded_frame_buffer",
    "context_frame_buffer_max_size",
    "decoded_frame_buffer_max_size",
]


# VACE control map modes
VACE_CONTROL_MAP_MODES = ["none", "canny", "pidinet", "depth", "composite", "external"]

# Control-map preview worker policy:
# - "canny" is CPU-only and usually cheap enough to run at input FPS for preview.
# - "pidinet" / "depth" / "composite" can be GPU-heavy; running them concurrently with
#   generation can materially reduce end-to-end FPS.
VACE_CONTROL_MAP_WORKER_HEAVY_MODES = {"pidinet", "depth", "composite"}


def apply_canny_edges(
    frames: list[torch.Tensor],
    low_threshold: int | None = None,
    high_threshold: int | None = None,
    blur_kernel_size: int = 5,
    blur_sigma: float = 1.4,
    adaptive_thresholds: bool = True,
    dilate_edges: bool = False,
    dilate_kernel_size: int = 2,
) -> list[torch.Tensor]:
    """Apply Canny edge detection to video frames for VACE control.

    Args:
        frames: List of tensors, each (1, H, W, C) `uint8` or float in [0, 255].
        low_threshold: Canny low threshold. If None and adaptive_thresholds=True,
            computed as 0.66 * median pixel value.
        high_threshold: Canny high threshold. If None and adaptive_thresholds=True,
            computed as 1.33 * median pixel value.
        blur_kernel_size: Gaussian blur kernel size (must be odd). Set to 0 to disable.
        blur_sigma: Gaussian blur sigma. Higher = more blur.
        adaptive_thresholds: If True and thresholds are None, compute thresholds
            based on image statistics (median-based). Produces cleaner edges.
        dilate_edges: If True, dilate edges to make them thicker/more visible.
        dilate_kernel_size: Size of dilation kernel.

    Returns:
        List of edge tensors, each (1, H, W, 3) uint8 in [0, 255].
    """
    result = []

    # Prepare dilation kernel if needed
    dilate_kernel = None
    if dilate_edges and dilate_kernel_size > 0:
        dilate_kernel = np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8)

    for frame in frames:
        # frame is (1, H, W, C) uint8 or float in [0, 255]
        img_t = frame.squeeze(0)
        if img_t.dtype == torch.uint8:
            img = img_t.numpy()
        else:
            img = img_t.clamp(0, 255).to(dtype=torch.uint8).numpy()
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Apply Gaussian blur to reduce noise (improves edge quality significantly)
        if blur_kernel_size > 0:
            # Ensure kernel size is odd
            k = blur_kernel_size if blur_kernel_size % 2 == 1 else blur_kernel_size + 1
            gray = cv2.GaussianBlur(gray, (k, k), blur_sigma)

        # Compute adaptive thresholds if not provided
        if adaptive_thresholds and (low_threshold is None or high_threshold is None):
            # Median-based adaptive thresholds (common technique for Canny)
            median_val = np.median(gray)
            low_t = int(max(0, 0.66 * median_val)) if low_threshold is None else low_threshold
            high_t = int(min(255, 1.33 * median_val)) if high_threshold is None else high_threshold
        else:
            # Use provided thresholds or defaults
            low_t = low_threshold if low_threshold is not None else 100
            high_t = high_threshold if high_threshold is not None else 200

        edges = cv2.Canny(gray, low_t, high_t)

        # Optional dilation to thicken edges
        if dilate_kernel is not None:
            edges = cv2.dilate(edges, dilate_kernel, iterations=1)

        # Convert back to 3-channel RGB
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        result.append(torch.from_numpy(edges_rgb).unsqueeze(0))
    return result


def soft_max_fusion(
    depth: torch.Tensor,
    edges: torch.Tensor,
    edge_strength: float = 0.6,
    sharpness: float = 10.0,
) -> torch.Tensor:
    """Fuse depth and edges using soft max for smooth transitions.

    Soft max avoids hard discontinuities at transition boundaries.
    Edges "punch through" where strong, depth dominates elsewhere.

    Args:
        depth: Depth tensor normalized to [0, 1], shape (H, W) or (1, H, W, C).
        edges: Edge tensor normalized to [0, 1], shape (H, W) or (1, H, W, C).
        edge_strength: Scale factor for edges (0.5-0.7 recommended).
        sharpness: Controls transition sharpness (higher = sharper).

    Returns:
        Fused tensor in [0, 1], same shape as input.
    """
    # Inputs should already be normalized, but clamp defensively.
    depth = torch.clamp(depth, 0.0, 1.0)
    edges = torch.clamp(edges, 0.0, 1.0)

    edge_strength = max(0.0, float(edge_strength))
    sharpness = float(sharpness)
    if sharpness <= 0:
        return torch.clamp(depth, 0.0, 1.0)

    scaled_edges = edges * edge_strength

    # Soft max (stable): logaddexp(a, b) / sharpness.
    d_scaled = torch.clamp(sharpness * depth, -20, 20)
    e_scaled = torch.clamp(sharpness * scaled_edges, -20, 20)
    fused = torch.logaddexp(d_scaled, e_scaled) / sharpness

    # Stabilize to [0, 1] without per-frame min/max normalization.
    # For a=b=0, fused=log(2)/sharpness; for depth=1 and edges=1, fused reaches a fixed max.
    fused_min = float(np.log(2.0) / sharpness)
    fused_max = float(np.logaddexp(sharpness, sharpness * edge_strength) / sharpness)
    denom = fused_max - fused_min
    if denom <= 1e-6:
        return torch.clamp(fused, 0.0, 1.0)
    return torch.clamp((fused - fused_min) / denom, 0.0, 1.0)


class VDADepthControlMapGenerator:
    """Video Depth Anything streaming depth generator for VACE control maps.

    This class owns the VDA model and its streaming cache state. It should be
    owned by a FrameProcessor (per-session) and reset on hard cuts.

    The model is lazy-loaded on first use to avoid GPU memory allocation
    until depth mode is actually enabled.
    """

    # Model configs from VDA repo
    MODEL_CONFIGS = {
        "vits": {
            "encoder": "vits",
            "features": 64,
            "out_channels": [48, 96, 192, 384],
        },
        "vitb": {
            "encoder": "vitb",
            "features": 128,
            "out_channels": [96, 192, 384, 768],
        },
        "vitl": {
            "encoder": "vitl",
            "features": 256,
            "out_channels": [256, 512, 1024, 1024],
        },
    }

    def __init__(
        self,
        encoder: str = "vits",
        checkpoint_path: str | Path | None = None,
        device: str = "cuda",
        input_size: int = 518,
    ):
        self.encoder = encoder
        self.checkpoint_path = Path(
            checkpoint_path
            if checkpoint_path is not None
            else get_model_file_path("vda/video_depth_anything_vits.pth")
        )
        self.device = device
        self.input_size = input_size

        self._model = None
        self._model_loaded = False

        # Stabilization state (running quantiles for normalization)
        self._q_lo: float | torch.Tensor | None = None
        self._q_hi: float | torch.Tensor | None = None
        self._quantile_momentum = 0.95  # EMA momentum for quantile updates

        # Contrast adjustment: gamma curve applied after normalization.
        # Values > 1.0 increase mid-tone contrast (emphasize subtle depth differences).
        # Values < 1.0 reduce contrast. Default 1.0 = no change.
        self._depth_contrast: float = 1.0

    @property
    def depth_contrast(self) -> float:
        """Get current depth contrast (gamma) value."""
        return self._depth_contrast

    @depth_contrast.setter
    def depth_contrast(self, value: float) -> None:
        """Set depth contrast (gamma). Values > 1.0 increase contrast."""
        self._depth_contrast = max(0.1, min(5.0, float(value)))  # Clamp to sane range

    def _load_model(self):
        """Lazy-load the VDA model on first use."""
        if self._model_loaded:
            return

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"VDA checkpoint not found: {self.checkpoint_path} "
                "(expected under DAYDREAM_SCOPE_MODELS_DIR)."
            )

        try:
            from scope.vendored.video_depth_anything import VideoDepthAnything

            config = self.MODEL_CONFIGS[self.encoder]
            self._model = VideoDepthAnything(**config)
            self._model.load_state_dict(
                torch.load(self.checkpoint_path, map_location="cpu"),
                strict=True,
            )
            self._model = self._model.to(self.device).eval()

            if _is_env_true("SCOPE_VDA_COMPILE", default="0"):
                fullgraph = _is_env_true("SCOPE_VDA_TORCH_COMPILE_FULLGRAPH", default="0")
                dynamic = _is_env_true("SCOPE_VDA_TORCH_COMPILE_DYNAMIC", default="0")
                compiled_any = False
                try:
                    self._model.forward_features = torch.compile(
                        self._model.forward_features,
                        fullgraph=fullgraph,
                        dynamic=dynamic,
                    )
                    compiled_any = True
                except Exception as e:
                    logger.warning("VDA torch.compile forward_features failed: %s", e)

                try:
                    self._model.forward_depth = torch.compile(
                        self._model.forward_depth,
                        fullgraph=fullgraph,
                        dynamic=dynamic,
                    )
                    compiled_any = True
                except Exception as e:
                    logger.warning("VDA torch.compile forward_depth failed: %s", e)

                if compiled_any:
                    logger.info(
                        "VDA torch.compile enabled (fullgraph=%s dynamic=%s)",
                        fullgraph,
                        dynamic,
                    )

            self._model_loaded = True
            logger.info(
                "VDA model loaded: encoder=%s device=%s checkpoint=%s",
                self.encoder,
                self.device,
                self.checkpoint_path,
            )
        except Exception as e:
            logger.error(f"Failed to load VDA model: {e}")
            raise

    def _reset_model_temporal_cache(self) -> None:
        """Reset only the VDA streaming cache (not normalization state)."""
        if self._model is None:
            return
        # Reset VDA streaming cache
        self._model.transform = None
        self._model.frame_id_list = []
        self._model.frame_cache_list = []
        self._model.id = -1

    def reset_cache(self):
        """Reset streaming cache and stabilization state.

        Call this on hard cuts (when init_cache=True) to prevent blending
        depth across discontinuities.
        """
        self._reset_model_temporal_cache()

        # Reset stabilization state
        self._q_lo = None
        self._q_hi = None
        logger.debug("VDA depth cache reset")

    def process_frames(
        self,
        frames: list[torch.Tensor],
        hard_cut: bool = False,
        *,
        input_size: int | None = None,
        fp32: bool | None = None,
        temporal_mode: str | None = None,
        output_device: str = "cpu",
    ) -> list[torch.Tensor]:
        """Process frames through VDA and return depth maps as control frames.

        Args:
            frames: List of tensors, each (1, H, W, C) uint8 or float in [0, 255].
            hard_cut: If True, reset cache before processing.
            input_size: Override VDA resize target (lower is faster). If changed mid-stream,
                the generator treats it like a hard cut and resets streaming state.
            fp32: If True, force FP32 (disables autocast). Default is False (autocast enabled).
            temporal_mode: Controls whether VDA uses its streaming temporal cache.
                - "stream" (default): uses temporal cache (more stable, can trail/ghost on fast motion)
                - "stateless": disables temporal cache (no trails, can be noisier)
            output_device: Device for returned control frames ("cpu" or "cuda"). Default: "cpu".

        Returns:
            List of depth tensors, each (1, H, W, 3) uint8 in [0, 255].
            Depth is normalized per-session using running quantiles.
        """
        self._load_model()

        if input_size is not None:
            input_size_int = int(input_size)
            if input_size_int <= 0:
                raise ValueError(f"input_size must be > 0, got {input_size_int}")
            # VDA's transform is initialized only on the first frame; changing the
            # input_size mid-stream requires a reset so the transform + caches are consistent.
            if input_size_int != self.input_size:
                self.input_size = input_size_int
                hard_cut = True

        fp32_flag = bool(fp32) if fp32 is not None else False

        temporal_mode_env = os.getenv("SCOPE_VACE_DEPTH_TEMPORAL_MODE", "").strip()
        temporal_mode_norm = (
            (temporal_mode if temporal_mode is not None else temporal_mode_env) or "stream"
        ).strip().lower()
        if temporal_mode_norm in ("stream", "streaming", "temporal", "1", "true", "yes", "on"):
            use_temporal_cache = True
        elif temporal_mode_norm in (
            "stateless",
            "single",
            "single_frame",
            "no_cache",
            "0",
            "false",
            "no",
            "off",
        ):
            use_temporal_cache = False
        else:
            logger.warning(
                "Invalid vace_depth_temporal_mode=%r; expected 'stream' or 'stateless' (falling back to 'stream')",
                temporal_mode_norm,
            )
            use_temporal_cache = True

        if hard_cut:
            self.reset_cache()

        output_device_norm = (output_device or "cpu").strip().lower()
        output_on_gpu = output_device_norm not in ("", "cpu")

        result = []
        quantile_stride = int(os.getenv("SCOPE_VACE_DEPTH_QUANTILE_STRIDE", "4") or "4")
        if quantile_stride < 1:
            quantile_stride = 1
        for frame in frames:
            # frame is (1, H, W, C) uint8 or float in [0, 255]
            # VDA expects RGB numpy array (H, W, 3) uint8
            img_t = frame.squeeze(0)
            if img_t.dtype == torch.uint8:
                img = img_t.numpy()
            else:
                img = img_t.clamp(0, 255).to(dtype=torch.uint8).numpy()

            # Infer depth (H, W): numpy array by default, or a torch tensor when output_on_gpu.
            try:
                with torch.no_grad():
                    if not use_temporal_cache:
                        # Stateless mode: force VDA to treat every frame as a "first frame" by
                        # resetting its temporal cache before each inference call.
                        self._reset_model_temporal_cache()
                    depth = self._model.infer_video_depth_one(
                        img,
                        input_size=self.input_size,
                        device=self.device,
                        fp32=fp32_flag,
                        return_torch=output_on_gpu,
                    )
            except AssertionError:
                # VDA streaming asserts frame size consistency; treat size changes
                # as a hard cut and re-run as "first frame".
                self.reset_cache()
                with torch.no_grad():
                    if not use_temporal_cache:
                        self._reset_model_temporal_cache()
                    depth = self._model.infer_video_depth_one(
                        img,
                        input_size=self.input_size,
                        device=self.device,
                        fp32=fp32_flag,
                        return_torch=output_on_gpu,
                    )

            if not output_on_gpu:
                # -------------------------------
                # CPU output path (legacy)
                # -------------------------------
                depth_np = depth

                # Stabilize using running quantiles (avoid per-frame normalization)
                depth_sample = depth_np
                if quantile_stride > 1:
                    depth_sample = depth_np[::quantile_stride, ::quantile_stride]
                q_lo_frame, q_hi_frame = np.percentile(depth_sample, [2, 98])

                # Initialize quantiles on first frame to avoid startup saturation.
                if self._q_lo is None or self._q_hi is None:
                    self._q_lo = float(q_lo_frame)
                    self._q_hi = float(q_hi_frame)
                else:
                    q_lo_prev = (
                        float(self._q_lo.detach().float().cpu().item())
                        if isinstance(self._q_lo, torch.Tensor)
                        else float(self._q_lo)
                    )
                    q_hi_prev = (
                        float(self._q_hi.detach().float().cpu().item())
                        if isinstance(self._q_hi, torch.Tensor)
                        else float(self._q_hi)
                    )
                    self._q_lo = (
                        self._quantile_momentum * q_lo_prev
                        + (1 - self._quantile_momentum) * float(q_lo_frame)
                    )
                    self._q_hi = (
                        self._quantile_momentum * q_hi_prev
                        + (1 - self._quantile_momentum) * float(q_hi_frame)
                    )

                # Normalize to [0, 1] using running quantiles
                q_lo = float(self._q_lo)
                q_hi = float(self._q_hi)
                depth_range = max(q_hi - q_lo, 1e-6)
                depth_norm = np.clip((depth_np - q_lo) / depth_range, 0.0, 1.0)

                # Apply contrast (gamma) curve: values > 1.0 increase mid-tone contrast
                if self._depth_contrast != 1.0:
                    depth_norm = np.power(depth_norm, 1.0 / self._depth_contrast)

                # Convert to [0, 255] and replicate to 3 channels
                depth_uint8 = (depth_norm * 255.0).astype(np.uint8)
                depth_rgb = np.repeat(depth_uint8[:, :, None], 3, axis=2)

                result.append(torch.from_numpy(depth_rgb).unsqueeze(0))
                continue

            # -------------------------------
            # GPU output path
            # -------------------------------
            if not isinstance(depth, torch.Tensor):
                raise TypeError(
                    f"Expected torch depth when output_device={output_device!r}, got: {type(depth)}"
                )

            depth_t = depth
            if output_device_norm not in ("cuda", "gpu") and not output_device_norm.startswith(
                "cuda:"
            ):
                # Only "cuda*" devices are supported for output_on_gpu (caller requested non-cpu).
                raise ValueError(
                    f"Unsupported output_device={output_device!r}; expected 'cpu' or 'cuda'"
                )

            use_gpu_quantiles = os.getenv("SCOPE_VACE_DEPTH_GPU_QUANTILES", "").lower() in (
                "1",
                "true",
                "yes",
                "on",
            )
            if use_gpu_quantiles:
                # Keep the normalization path on GPU to avoid the `.cpu().numpy()`
                # sync point when output_device=cuda.
                depth_sample_t = depth_t
                if quantile_stride > 1:
                    depth_sample_t = depth_sample_t[::quantile_stride, ::quantile_stride]

                sample_flat = depth_sample_t.detach().to(dtype=torch.float32).flatten()
                q_lo_frame_t = torch.quantile(sample_flat, 0.02)
                q_hi_frame_t = torch.quantile(sample_flat, 0.98)

                # Ensure cached quantiles are torch scalars on the right device.
                if self._q_lo is None:
                    self._q_lo = q_lo_frame_t
                elif not isinstance(self._q_lo, torch.Tensor):
                    self._q_lo = torch.tensor(
                        float(self._q_lo),
                        device=depth_t.device,
                        dtype=torch.float32,
                    )

                if self._q_hi is None:
                    self._q_hi = q_hi_frame_t
                elif not isinstance(self._q_hi, torch.Tensor):
                    self._q_hi = torch.tensor(
                        float(self._q_hi),
                        device=depth_t.device,
                        dtype=torch.float32,
                    )

                self._q_lo = (
                    self._quantile_momentum * self._q_lo
                    + (1 - self._quantile_momentum) * q_lo_frame_t
                )
                self._q_hi = (
                    self._quantile_momentum * self._q_hi
                    + (1 - self._quantile_momentum) * q_hi_frame_t
                )

                depth_range_t = (self._q_hi - self._q_lo).clamp_min(1e-6)
                depth_norm_t = ((depth_t - self._q_lo) / depth_range_t).clamp(0.0, 1.0)
            else:
                # Stabilize quantiles on CPU using a downsampled grid to minimize sync cost.
                depth_sample_t = depth_t
                if quantile_stride > 1:
                    depth_sample_t = depth_sample_t[::quantile_stride, ::quantile_stride]
                q_lo_frame, q_hi_frame = np.percentile(
                    depth_sample_t.detach().float().cpu().numpy(), [2, 98]
                )

                if self._q_lo is None or self._q_hi is None:
                    self._q_lo = float(q_lo_frame)
                    self._q_hi = float(q_hi_frame)
                else:
                    q_lo_prev = (
                        float(self._q_lo.detach().float().cpu().item())
                        if isinstance(self._q_lo, torch.Tensor)
                        else float(self._q_lo)
                    )
                    q_hi_prev = (
                        float(self._q_hi.detach().float().cpu().item())
                        if isinstance(self._q_hi, torch.Tensor)
                        else float(self._q_hi)
                    )
                    self._q_lo = (
                        self._quantile_momentum * q_lo_prev
                        + (1 - self._quantile_momentum) * float(q_lo_frame)
                    )
                    self._q_hi = (
                        self._quantile_momentum * q_hi_prev
                        + (1 - self._quantile_momentum) * float(q_hi_frame)
                    )

                q_lo = float(self._q_lo)
                q_hi = float(self._q_hi)
                depth_range = max(q_hi - q_lo, 1e-6)

                depth_norm_t = ((depth_t - q_lo) / depth_range).clamp(0.0, 1.0)

            # Apply contrast (gamma) curve: values > 1.0 increase mid-tone contrast
            if self._depth_contrast != 1.0:
                depth_norm_t = torch.pow(depth_norm_t, 1.0 / self._depth_contrast)

            depth_uint8_t = (depth_norm_t * 255.0).to(dtype=torch.uint8)
            depth_rgb_t = depth_uint8_t.unsqueeze(-1).repeat(1, 1, 3)

            result.append(depth_rgb_t.unsqueeze(0))

        return result


class PiDiNetEdgeGenerator:
    """PiDiNet neural edge detector for high-quality VACE control maps.

    Uses the PiDiNet model from controlnet_aux for learned edge detection
    that produces cleaner, more semantically meaningful edges than Canny.

    Requires: controlnet_aux. If missing, run `uv sync` or `pip install controlnet_aux`.
    """

    def __init__(
        self,
        device: str = "cuda",
        safe_mode: bool = True,
    ):
        """Initialize PiDiNet edge generator.

        Args:
            device: Device to run model on ("cuda" or "cpu").
            safe_mode: If True, use safe/cleaner edge detection mode.
        """
        self.device = device
        self.safe_mode = safe_mode
        self._model = None
        self._model_loaded = False

    def _load_model(self):
        """Lazy-load the PiDiNet model on first use."""
        if self._model_loaded:
            return

        try:
            from controlnet_aux import PidiNetDetector

            self._model = PidiNetDetector.from_pretrained("lllyasviel/Annotators")
            # Move to device if possible (some models support .to())
            if hasattr(self._model, "to"):
                self._model = self._model.to(self.device)
            self._model_loaded = True
            logger.info(
                "PiDiNet model loaded: device=%s safe_mode=%s",
                self.device,
                self.safe_mode,
            )
        except ImportError as e:
            raise ImportError(
                "controlnet_aux not installed. Install with: pip install controlnet_aux "
                "(or run: uv sync)."
            ) from e
        except Exception as e:
            logger.error(f"Failed to load PiDiNet model: {e}")
            raise

    def process_frames(
        self, frames: list[torch.Tensor], apply_filter: bool = True
    ) -> list[torch.Tensor]:
        """Process frames through PiDiNet and return edge maps.

        Args:
            frames: List of tensors, each (1, H, W, C) `uint8` or float in [0, 255].
            apply_filter: If True, apply post-processing filter for cleaner edges.

        Returns:
            List of edge tensors, each (1, H, W, 3) float in [0, 255].
        """
        self._load_model()

        from PIL import Image

        result = []
        for frame in frames:
            # frame is (1, H, W, C) uint8 or float in [0, 255]
            img_t = frame.squeeze(0)
            if img_t.dtype == torch.uint8:
                img = img_t.numpy()
            else:
                img = img_t.clamp(0, 255).to(dtype=torch.uint8).numpy()

            # Convert to PIL Image for controlnet_aux
            pil_img = Image.fromarray(img)

            # Run PiDiNet detection
            edge_pil = self._model(
                pil_img,
                detect_resolution=min(pil_img.width, pil_img.height),
                image_resolution=min(pil_img.width, pil_img.height),
                safe=self.safe_mode,
                apply_filter=apply_filter,
            )

            # Convert back to numpy/tensor
            edge_np = np.array(edge_pil)

            # Ensure 3-channel output
            if len(edge_np.shape) == 2:
                edge_np = cv2.cvtColor(edge_np, cv2.COLOR_GRAY2RGB)
            elif edge_np.shape[2] == 4:
                edge_np = cv2.cvtColor(edge_np, cv2.COLOR_RGBA2RGB)

            # Ensure output matches input resolution (required for composite mode).
            if edge_np.shape[0] != img.shape[0] or edge_np.shape[1] != img.shape[1]:
                edge_np = cv2.resize(
                    edge_np,
                    (img.shape[1], img.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )

            result.append(torch.from_numpy(edge_np).unsqueeze(0).float())

        return result


@dataclass
class Snapshot:
    """Server-side snapshot of generation state at a chunk boundary.

    Snapshots are stored in-memory and contain cloned GPU tensors.
    Clients receive only snapshot_id + metadata, not the actual tensor data.
    """

    snapshot_id: str
    chunk_index: int
    created_at: float

    # Continuity state (cloned tensors from pipeline.state)
    current_start_frame: int = 0
    first_context_frame: torch.Tensor | None = None
    context_frame_buffer: torch.Tensor | None = None
    decoded_frame_buffer: torch.Tensor | None = None
    context_frame_buffer_max_size: int = 0
    decoded_frame_buffer_max_size: int = 0

    # Control state (deep copy of parameters)
    parameters: dict[str, Any] = field(default_factory=dict)
    paused: bool = False
    video_mode: bool = False

    # Style layer state (minimal, deterministic)
    world_state_json: str | None = None  # JSON string for thread-safe restore
    active_style_name: str | None = None  # Used for edge-triggering LoRA updates
    compiled_prompt_text: str | None = None  # For debugging

    # Compatibility metadata (for future validation)
    pipeline_id: str | None = None
    resolution: tuple[int, int] | None = None


class _FrameWithID:
    """Attach a monotonically increasing frame_id to any input frame-like object.

    The wrapped object must expose: to_ndarray(format="rgb24") -> np.ndarray
    Phase 2.1b: Used for frame ID tracking in control buffer architecture.
    """

    __slots__ = ["frame_id", "_frame"]

    def __init__(self, frame: Any, frame_id: int):
        self._frame = frame
        self.frame_id = int(frame_id)

    def to_ndarray(self, format: str = "rgb24"):
        return self._frame.to_ndarray(format=format)

    def __getattr__(self, name: str) -> Any:
        # Delegate any other access (pts, time_base, etc.) to the underlying frame.
        return getattr(self._frame, name)


class _SpoutFrame:
    """Lightweight wrapper for Spout frames to match VideoFrame interface."""

    __slots__ = ["_data"]

    def __init__(self, data):
        self._data = data

    def to_ndarray(self, format="rgb24"):
        return self._data


class _NDIFrame:
    """Lightweight wrapper for NDI frames to match VideoFrame interface."""

    __slots__ = ["_data"]

    def __init__(self, data):
        self._data = data

    def to_ndarray(self, format="rgb24"):
        return self._data


class ControlMapWorker:
    """Background worker that generates control maps from raw frames.

    This worker runs independently of chunk processing, receiving raw frames and
    producing control maps continuously.

    - Phase 2.1a: feeds the MJPEG preview stream at input FPS.
    - Phase 2.1b (when control buffer is enabled): also writes per-frame control
      maps into a ring buffer keyed by `frame_id`, so generation can sample
      precomputed control maps instead of doing chunk-time control-map compute.

    Note: In `vace_control_map_mode="external"`, incoming frames are already
    control maps and are not re-processed by this worker.
    """

    def __init__(
        self,
        latest_control_frame_lock: threading.Lock,
        parameters_getter: callable,
        max_queue_size: int = 60,
        control_buffer_size: int = 120,
    ):
        """Initialize control map worker.

        Args:
            latest_control_frame_lock: Lock for updating latest_control_frame_cpu.
            parameters_getter: Callable that returns current parameters dict.
            max_queue_size: Max frames to buffer (small = low latency).
            control_buffer_size: Ring buffer size (frames) for generation sampling by frame_id.
        """
        self._lock = latest_control_frame_lock
        self._get_params = parameters_getter
        self._input_queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._latest_frame: torch.Tensor | None = None

        # Phase 2.1b: generation ring buffer keyed by frame_id
        self._control_buffer_maxlen = int(control_buffer_size)
        self._control_buffer: OrderedDict[int, torch.Tensor] = OrderedDict()
        self._control_buffer_lock = threading.Lock()
        self._last_frame_id: int | None = None

        self._thread: threading.Thread | None = None
        self._shutdown = threading.Event()
        self._running = False

        # Own generator instances (not shared with chunk processor)
        self._depth_generator: VDADepthControlMapGenerator | None = None
        self._pidinet_generator: PiDiNetEdgeGenerator | None = None

        # Stats
        self._frames_processed = 0
        self._last_mode: str | None = None
        self._dropped_input_frames = 0

        # Hard cut request (applied on worker thread to avoid cross-thread generator mutation)
        self._hard_cut_requested = False

        # Throttle: cap processing rate to reduce GPU contention.
        # Disabled by default (max_fps <= 0).
        self._max_fps: float | None = None
        self._min_process_interval_s = 0.0
        self._last_process_t = 0.0

        # Phase 2.1b: worker is the primary producer for generation when buffer enabled,
        # so heavy modes should be allowed by default when buffer is on.
        # When buffer disabled: default allow_heavy=0 (old behavior, preview-only)
        # When buffer enabled: default allow_heavy=1 (worker feeds generation)
        # Explicit SCOPE_VACE_CONTROL_MAP_WORKER_ALLOW_HEAVY always wins.
        buffer_enabled = _is_env_true("SCOPE_VACE_CONTROL_BUFFER_ENABLED", default="0")
        allow_heavy_default = "1" if buffer_enabled else "0"
        self._allow_heavy = _is_env_true(
            "SCOPE_VACE_CONTROL_MAP_WORKER_ALLOW_HEAVY", default=allow_heavy_default
        )

    def set_max_fps(self, max_fps: float | None) -> None:
        if max_fps is None:
            self._max_fps = None
            self._min_process_interval_s = 0.0
            return

        max_fps_f = float(max_fps)
        if max_fps_f <= 0:
            self._max_fps = None
            self._min_process_interval_s = 0.0
            return

        self._max_fps = max_fps_f
        self._min_process_interval_s = 1.0 / max_fps_f

    def set_allow_heavy(self, allow_heavy: bool) -> None:
        """Enable/disable GPU-heavy control-map modes in the worker (depth/pidinet/composite)."""
        self._allow_heavy = bool(allow_heavy)

    def clear_latest(self, *, lock_held: bool = False) -> None:
        """Clear cached preview output so callers fall back to chunk outputs."""
        if lock_held:
            self._latest_frame = None
            return
        with self._lock:
            self._latest_frame = None

    def clear_generation_buffer(self) -> None:
        """Clear the generation ring buffer (Phase 2.1b)."""
        with self._control_buffer_lock:
            self._control_buffer.clear()
            self._last_frame_id = None

    def _drain_input_queue(self) -> int:
        """Best-effort drain of pending frames (used on hard cuts)."""
        drained = 0
        while True:
            try:
                item = self._input_queue.get_nowait()
                if item is None:
                    continue
                drained += 1
            except queue.Empty:
                break
        return drained

    def request_hard_cut(self, *, clear_queue: bool = True, reason: str | None = None) -> None:
        """Request a hard cut:
        - clears generation buffer + preview immediately,
        - optionally drains queued frames,
        - resets VDA streaming caches on the worker thread before the next processed frame.
        """
        _ = reason  # reserved for future debug logging
        self._hard_cut_requested = True
        self.clear_latest()
        self.clear_generation_buffer()
        if clear_queue:
            self._drain_input_queue()

    def get_debug_info(self) -> dict[str, object]:
        try:
            queue_depth = int(self._input_queue.qsize())
        except Exception:
            queue_depth = -1

        with self._control_buffer_lock:
            buffer_depth = len(self._control_buffer)
            last_frame_id = self._last_frame_id

        return {
            "running": bool(self._running),
            "last_mode": self._last_mode,
            "frames_processed": int(self._frames_processed),
            "dropped_input_frames": int(self._dropped_input_frames),
            "queue_depth": queue_depth,
            "control_buffer_depth": int(buffer_depth),
            "last_frame_id": int(last_frame_id) if last_frame_id is not None else None,
            "max_fps": float(self._max_fps) if self._max_fps is not None else None,
            "allow_heavy": bool(self._allow_heavy),
            "heavy_modes": sorted(VACE_CONTROL_MAP_WORKER_HEAVY_MODES),
        }

    def start(self):
        """Start the control map worker thread."""
        if self._running:
            return
        self._shutdown.clear()
        self._running = True
        self._thread = threading.Thread(
            target=self._worker_loop, name="ControlMapWorker", daemon=True
        )
        self._thread.start()
        logger.info("ControlMapWorker started")

    def stop(self):
        """Stop the control map worker thread."""
        if not self._running:
            return
        self._running = False
        self._shutdown.set()
        # Unblock queue.get() by putting sentinel
        try:
            self._input_queue.put_nowait(None)
        except queue.Full:
            pass
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        # Clear generators to free GPU memory
        self._depth_generator = None
        self._pidinet_generator = None
        self.clear_generation_buffer()
        logger.info(f"ControlMapWorker stopped after {self._frames_processed} frames")

    def put(self, frame) -> bool:
        """Enqueue a raw frame for control map processing.

        Non-blocking: drops frame if queue is full (preview can skip frames).

        Args:
            frame: VideoFrame, _SpoutFrame, or _FrameWithID with to_ndarray() method.

        Returns:
            True if queued, False if dropped.
        """
        if not self._running:
            return False
        try:
            self._input_queue.put_nowait(frame)
            return True
        except queue.Full:
            self._dropped_input_frames += 1
            # Note: We do NOT trigger hard cut on single drop (thrashing risk).
            # Future: could add threshold-based hard cut if drops are sustained.
            return False

    def get_latest(self) -> torch.Tensor | None:
        """Get the most recent control frame.

        Returns (H, W, 3) uint8 tensor or None.
        """
        with self._lock:
            if self._latest_frame is not None:
                return self._latest_frame.clone()
            return None

    def reset_cache(self):
        """Reset streaming caches (call on hard cuts). Backward-compatible alias."""
        self.request_hard_cut(clear_queue=True, reason="reset_cache")

    def sample_control_frames(self, frame_ids: list[int]) -> list[torch.Tensor] | None:
        """Sample generation control frames by exact frame_id (Phase 2.1b).

        Returns:
            List of (1, H, W, 3) tensors aligned to frame_ids, or None if any missing.
        """
        if not frame_ids:
            return []
        with self._control_buffer_lock:
            # Exact-match only (missing policy is handled by FrameProcessor).
            for fid in frame_ids:
                if fid not in self._control_buffer:
                    return None
            return [self._control_buffer[fid].clone() for fid in frame_ids]

    def _worker_loop(self):
        """Main worker loop: process frames continuously."""
        while self._running and not self._shutdown.is_set():
            try:
                # Block with timeout so we can check shutdown
                frame = self._input_queue.get(timeout=0.1)
                if frame is None:  # Sentinel
                    continue

                # Apply pending hard cut on worker thread (safe point)
                if self._hard_cut_requested:
                    self._hard_cut_requested = False
                    if self._depth_generator is not None:
                        self._depth_generator.reset_cache()
                    self.clear_generation_buffer()
                    self.clear_latest()
                    logger.debug("ControlMapWorker hard cut applied")

                # Get current mode from parameters
                params = self._get_params()
                mode = params.get("vace_control_map_mode", "none")

                if mode != self._last_mode:
                    logger.info(
                        "ControlMapWorker mode changed: %s -> %s", self._last_mode, mode
                    )
                    self._last_mode = mode
                    # Prevent stale preview frames when switching modes (or disabling control maps).
                    self.clear_latest()
                    # Also clear generation buffer and reset streaming caches.
                    self.request_hard_cut(clear_queue=False, reason="mode_change")

                # Skip when control maps are disabled or passthrough-only.
                # "external" mode means frames are already control maps and should not be
                # re-processed by the worker.
                if mode in ("none", "external"):
                    continue

                # Default: avoid duplicating GPU-heavy annotators in the preview worker.
                # Chunk processing will still generate control frames at chunk rate.
                if (not self._allow_heavy) and mode in VACE_CONTROL_MAP_WORKER_HEAVY_MODES:
                    continue

                if self._min_process_interval_s > 0:
                    now = time.perf_counter()
                    if now - self._last_process_t < self._min_process_interval_s:
                        continue

                # Process frame
                gen_frame, preview_frame = self._process_frame(frame, mode, params)
                if gen_frame is not None and preview_frame is not None:
                    with self._lock:
                        self._latest_frame = preview_frame

                    # Phase 2.1b: write to generation ring buffer keyed by frame_id
                    frame_id = getattr(frame, "frame_id", None)
                    if frame_id is not None:
                        with self._control_buffer_lock:
                            self._control_buffer[int(frame_id)] = gen_frame.detach().cpu()
                            self._control_buffer.move_to_end(int(frame_id))
                            while len(self._control_buffer) > self._control_buffer_maxlen:
                                self._control_buffer.popitem(last=False)
                            self._last_frame_id = int(frame_id)

                    self._frames_processed += 1
                    self._last_process_t = time.perf_counter()

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"ControlMapWorker error: {e}", exc_info=True)
                time.sleep(0.01)

    def _process_frame(
        self, frame, mode: str, params: dict
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
        """Process a single frame through the appropriate control map generator.

        Args:
            frame: VideoFrame, _SpoutFrame, or _FrameWithID.
            mode: Control map mode ("canny", "pidinet", "depth", "composite").
            params: Current parameters dict.

        Returns:
            Tuple of (gen_frame, preview_frame):
              - gen_frame: (1, H, W, 3) tensor for generation (dtype depends on mode)
              - preview_frame: (H, W, 3) uint8 tensor for MJPEG preview
            Or (None, None) on error.
        """
        try:
            # Extract RGB data from frame
            rgb = frame.to_ndarray(format="rgb24")
            # Create tensor in format expected by generators: (1, H, W, C) uint8 [0, 255]
            frame_tensor = torch.from_numpy(rgb).unsqueeze(0)
            frames = [frame_tensor]

            if mode == "canny":
                low = params.get("vace_canny_low_threshold")
                high = params.get("vace_canny_high_threshold")
                blur_kernel = params.get("vace_canny_blur_kernel", 5)
                blur_sigma = params.get("vace_canny_blur_sigma", 1.4)
                adaptive = params.get("vace_canny_adaptive", True)
                dilate = params.get("vace_canny_dilate", False)
                dilate_size = params.get("vace_canny_dilate_size", 2)

                control_frames = apply_canny_edges(
                    frames,
                    low_threshold=low,
                    high_threshold=high,
                    blur_kernel_size=blur_kernel,
                    blur_sigma=blur_sigma,
                    adaptive_thresholds=adaptive,
                    dilate_edges=dilate,
                    dilate_kernel_size=dilate_size,
                )

            elif mode == "pidinet":
                if self._pidinet_generator is None:
                    self._pidinet_generator = PiDiNetEdgeGenerator()
                safe_mode = params.get("vace_pidinet_safe", True)
                apply_filter = params.get("vace_pidinet_filter", True)
                self._pidinet_generator.safe_mode = safe_mode
                control_frames = self._pidinet_generator.process_frames(
                    frames, apply_filter=apply_filter
                )

            elif mode == "depth":
                if self._depth_generator is None:
                    self._depth_generator = VDADepthControlMapGenerator()
                # Note: hard_cut should be signaled via reset_cache() externally
                depth_input_size = params.get("vace_depth_input_size")
                depth_fp32 = params.get("vace_depth_fp32")
                depth_temporal_mode = params.get("vace_depth_temporal_mode")
                depth_contrast = params.get("vace_depth_contrast")
                if depth_contrast is not None:
                    self._depth_generator.depth_contrast = depth_contrast
                control_frames = self._depth_generator.process_frames(
                    frames,
                    hard_cut=False,
                    input_size=depth_input_size,
                    fp32=depth_fp32,
                    temporal_mode=depth_temporal_mode,
                )

            elif mode == "composite":
                # Composite mode: depth + edges fused with soft max
                if self._depth_generator is None:
                    self._depth_generator = VDADepthControlMapGenerator()

                # Get composite parameters
                edge_strength = params.get("composite_edge_strength", 0.6)
                edge_thickness = params.get("composite_edge_thickness", 8)
                sharpness = params.get("composite_sharpness", 10.0)
                edge_source = params.get("composite_edge_source", "canny")

                # Generate depth
                depth_input_size = params.get("vace_depth_input_size")
                depth_fp32 = params.get("vace_depth_fp32")
                depth_temporal_mode = params.get("vace_depth_temporal_mode")
                depth_contrast = params.get("vace_depth_contrast")
                if depth_contrast is not None:
                    self._depth_generator.depth_contrast = depth_contrast
                depth_frames = self._depth_generator.process_frames(
                    frames,
                    hard_cut=False,
                    input_size=depth_input_size,
                    fp32=depth_fp32,
                    temporal_mode=depth_temporal_mode,
                )

                # Generate edges based on source
                if edge_source == "pidinet":
                    if self._pidinet_generator is None:
                        self._pidinet_generator = PiDiNetEdgeGenerator()
                    safe_mode = params.get("vace_pidinet_safe", True)
                    apply_filter = params.get("vace_pidinet_filter", True)
                    self._pidinet_generator.safe_mode = safe_mode
                    edge_frames = self._pidinet_generator.process_frames(
                        frames, apply_filter=apply_filter
                    )
                else:
                    # Default to canny with dilation for thickness
                    edge_frames = apply_canny_edges(
                        frames,
                        adaptive_thresholds=True,
                        dilate_edges=True,
                        dilate_kernel_size=edge_thickness,
                    )

                # Fuse depth + edges
                depth_f = depth_frames[0]  # (1, H, W, 3) float [0, 255]
                edge_f = edge_frames[0]

                # Normalize to [0, 1] using first channel (grayscale)
                depth_norm = depth_f[:, :, :, 0] / 255.0
                edge_norm = edge_f[:, :, :, 0] / 255.0

                # Soft max fusion
                fused = soft_max_fusion(
                    depth_norm,
                    edge_norm,
                    edge_strength=edge_strength,
                    sharpness=sharpness,
                )

                # Convert back to (1, H, W, 3) float [0, 255]
                fused_uint8 = (fused * 255.0).clamp(0, 255)
                fused_rgb = fused_uint8.unsqueeze(-1).expand(-1, -1, -1, 3)
                control_frames = [fused_rgb]

            else:
                return None, None

            # control_frames[-1] is (1, H, W, 3) (uint8 for canny/depth; float for pidinet/composite)
            gen_frame = control_frames[-1]
            preview = gen_frame.squeeze(0)
            if preview.dtype != torch.uint8:
                preview = preview.clamp(0, 255).to(torch.uint8)
            return gen_frame, preview

        except Exception as e:
            logger.error(f"ControlMapWorker._process_frame error: {e}", exc_info=True)
            return None, None


class FrameProcessor:
    def __init__(
        self,
        pipeline_manager: PipelineManager,
        max_output_queue_size: int = 8,
        max_parameter_queue_size: int = 8,
        max_buffer_size: int = 30,
        initial_parameters: dict = None,
        notification_callback: callable = None,
    ):
        self.pipeline_manager = pipeline_manager

        output_queue_env = (
            os.getenv("SCOPE_OUTPUT_QUEUE_MAX_FRAMES", "").strip()
            or os.getenv("SCOPE_OUTPUT_QUEUE_SIZE", "").strip()
        )
        output_queue_max_frames: int | None = None
        if output_queue_env:
            try:
                output_queue_max_frames = max(1, int(output_queue_env))
                max_output_queue_size = output_queue_max_frames
            except ValueError:
                logger.warning(
                    "Invalid SCOPE_OUTPUT_QUEUE_MAX_FRAMES/SCOPE_OUTPUT_QUEUE_SIZE=%r; expected int",
                    output_queue_env,
                )

        self.frame_buffer = deque(maxlen=max_buffer_size)
        self.frame_buffer_lock = threading.Lock()
        self.output_queue = queue.Queue(maxsize=max_output_queue_size)
        self.output_queue_lock = threading.Lock()  # Protects queue resize and flush

        # Output queue drop counter (when consumer can't keep up).
        # Written by worker thread, read by debug endpoint.
        self.output_frames_dropped = 0

        # Low-latency mode: drop old input frames to reduce lag
        # Opt-in via SCOPE_LOW_LATENCY_INPUT=1 env var or parameter
        self._low_latency_mode = (
            os.environ.get("SCOPE_LOW_LATENCY_INPUT", "0") == "1"
            or (initial_parameters or {}).get("low_latency_input", False)
        )
        # Low-latency output: prefer newest generated frames over smooth playback.
        # When enabled, the output queue drops the oldest frames to keep latency bounded.
        self._low_latency_output_mode = (
            os.environ.get("SCOPE_LOW_LATENCY_OUTPUT", "0") == "1"
            or (initial_parameters or {}).get("low_latency_output", False)
        )
        # Optional cap on output queue max size (prevents auto-resize from increasing latency).
        self._output_queue_maxsize_cap: int | None = (
            output_queue_max_frames if output_queue_max_frames is not None else None
        )
        self._low_latency_buffer_factor = 2  # Keep chunk_size * factor frames max
        self.input_frames_dropped = 0  # Counter for dropped input frames

        # Non-destructive latest frame buffer for REST /api/frame/latest
        self.latest_frame_cpu: torch.Tensor | None = None
        self.latest_frame_lock = threading.Lock()
        # Monotonic version counter for latest_frame_cpu updates (observer pacing / wait).
        self.latest_frame_id = 0
        self._latest_frame_event = asyncio.Event()
        try:
            self._latest_frame_event_loop: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
        except RuntimeError:
            self._latest_frame_event_loop = None

        # Latest control frame for VACE preview (MJPEG streaming)
        self.latest_control_frame_cpu: torch.Tensor | None = None
        self.latest_control_frame_lock = threading.Lock()

        # Phase 2.1b: Frame ID tracking for control buffer alignment
        self._frame_id_lock = threading.Lock()
        self._next_frame_id = 0

        # Phase 2.1b: Feature flag to enable control buffer generation path
        # Default OFF for safety - enables scaffolding without changing behavior
        self._control_buffer_enabled = _is_env_true(
            "SCOPE_VACE_CONTROL_BUFFER_ENABLED", default="0"
        )

        # Control map worker for high-FPS preview (Phase 2.1a) and generation (Phase 2.1b)
        self._control_map_worker = ControlMapWorker(
            latest_control_frame_lock=self.latest_control_frame_lock,
            parameters_getter=lambda: self.parameters,
            max_queue_size=int(
                os.getenv("SCOPE_VACE_CONTROL_MAP_WORKER_QUEUE_SIZE", "60") or "60"
            ),
            control_buffer_size=int(
                os.getenv("SCOPE_VACE_CONTROL_BUFFER_MAXLEN", "120") or "120"
            ),
        )

        # VDA depth generator for VACE control maps (lazy-loaded)
        self._depth_generator: VDADepthControlMapGenerator | None = None

        # PiDiNet neural edge generator for VACE control maps (lazy-loaded)
        self._pidinet_generator: PiDiNetEdgeGenerator | None = None

        # Temporal EMA state for control map smoothing
        self._prev_control_frames: list[torch.Tensor] | None = None
        self._prev_control_map_mode: str | None = None

        # Current parameters used by processing thread
        self.parameters = initial_parameters or {}
        # Queue for parameter updates from external threads
        self.parameters_queue = queue.Queue(maxsize=max_parameter_queue_size)

        self.worker_thread: threading.Thread | None = None
        self.shutdown_event = threading.Event()
        self.running = False

        self.is_prepared = False

        # Callback to notify when frame processor stops
        self.notification_callback = notification_callback

        # FPS tracking variables
        self.processing_time_per_frame = deque(
            maxlen=2
        )  # Keep last 2 processing_time/num_frames values for averaging
        self.last_fps_update = time.time()
        self.fps_update_interval = 0.5  # Update FPS every 0.5 seconds
        self.min_fps = MIN_FPS
        self.max_fps = MAX_FPS
        self.current_pipeline_fps = DEFAULT_FPS
        self.fps_lock = threading.Lock()  # Lock for thread-safe FPS updates

        # Input FPS tracking variables
        self.input_frame_times = deque(maxlen=INPUT_FPS_SAMPLE_SIZE)
        self.current_input_fps = DEFAULT_FPS
        self.last_input_fps_update = time.time()
        self.input_fps_lock = threading.Lock()

        self.paused = False

        # Control bus for deterministic event ordering at chunk boundaries
        self.control_bus = ControlBus()
        self.chunk_index = 0

        # Style layer: WorldState + StyleManifest + TemplateCompiler
        self.world_state: WorldState = WorldState()
        self.style_manifest: StyleManifest | None = None
        self.style_registry: StyleRegistry = StyleRegistry()
        self.prompt_compiler: TemplateCompiler = TemplateCompiler()
        self._compiled_prompt: CompiledPrompt | None = None
        self._active_style_name: str | None = None  # For edge-triggering LoRA updates

        # Step mode: allow generating N chunks even while paused.
        # Stored on the worker thread for deterministic semantics.
        self._pending_steps = 0

        # Soft transition state (temporary KV cache bias adjustment)
        self._soft_transition_active: bool = False
        self._soft_transition_chunks_remaining: int = 0
        self._soft_transition_temp_bias: float | None = None
        self._soft_transition_original_bias: float | None = None
        self._soft_transition_original_bias_was_set: bool = False
        # Soft transition recording latch: record softCut once at the next generated chunk.
        self._soft_transition_record_pending: bool = False

        # Style switch behavior: when True, reset cache on style change (clean transition)
        # When False, allow blend artifacts during style transitions (artistic effect)
        self.reset_cache_on_style_switch: bool = True

        # Session recorder (server-side timeline export)
        self.session_recorder = SessionRecorder()
        self._last_recording_path: Path | None = None

        # Snapshot store (server-side, in-memory)
        # Keys are snapshot_id, values are Snapshot objects with cloned tensors
        self.snapshots: dict[str, Snapshot] = {}
        self.snapshot_order: list[str] = []  # For LRU eviction (oldest first)
        self.snapshot_response_callback: callable | None = None

        # Spout integration
        self.spout_sender = None
        self.spout_sender_enabled = False
        self.spout_sender_name = "ScopeSyphonSpoutOut"
        self._frame_spout_count = 0
        self.spout_sender_queue = queue.Queue(
            maxsize=30
        )  # Queue for async Spout sending
        self.spout_sender_thread = None

        # Spout input
        self.spout_receiver = None
        self.spout_receiver_enabled = False
        self.spout_receiver_name = ""
        self.spout_receiver_thread = None

        # NDI input
        self.ndi_receiver = None
        self.ndi_receiver_enabled = False
        self.ndi_receiver_source = ""
        self.ndi_receiver_extra_ips: list[str] | None = None
        self.ndi_receiver_thread = None

        # NDI stats (exposed via debug/status endpoints)
        self.ndi_frames_received: int = 0
        self.ndi_frames_dropped: int = 0
        self.ndi_last_frame_ts_s: float = 0.0
        self.ndi_frames_reused: int = 0
        self.ndi_connected_source: str | None = None
        self.ndi_connected_url: str | None = None
        self.ndi_reconnects: int = 0

        # "Hold last input frame" for NDI external/passthrough control mode.
        # This decouples generation from NDI jitter / lower FPS producers.
        self._ndi_hold_last_input_frame: torch.Tensor | None = None
        self._ndi_hold_last_input_frame_id: int | None = None
        self.external_input_stale: bool = False
        self._external_resume_hard_cut_pending: bool = False

        # Input source exclusivity (Phase 0 for external video inputs)
        # Prevent mixing WebRTC + Spout (+ future NDI) frames in the same buffer.
        self._input_source_lock = threading.Lock()
        self._active_input_source: str = "webrtc"  # "webrtc" | "spout" | "ndi"

        # Input mode is signaled by the frontend at stream start.
        # This determines whether we wait for video frames or generate immediately.
        self._video_mode = (initial_parameters or {}).get("input_mode") == "video"

        # Hard cut: if reset_cache is requested while waiting for video input,
        # flush once (no log spam) but keep reset_cache pending until applied.
        self._hard_cut_flushed_pending = False

    def _get_current_effective_prompt(self) -> tuple[str | None, float]:
        """Best-effort extraction of the current pipeline-facing prompt.

        Precedence:
        1) transition.target_prompts[0]
        2) parameters["prompts"][0]
        3) pipeline.state["prompts"][0] (fallback)
        4) style layer compiled prompt (multiple shapes)
        """
        transition = self.parameters.get("transition")
        if isinstance(transition, dict):
            targets = transition.get("target_prompts")
            if isinstance(targets, list) and targets:
                first = targets[0]
                if isinstance(first, dict):
                    return first.get("text"), float(first.get("weight", 1.0))

        prompts = self.parameters.get("prompts")
        if isinstance(prompts, list) and prompts:
            first = prompts[0]
            if isinstance(first, dict):
                return first.get("text"), float(first.get("weight", 1.0))
            if hasattr(first, "text"):
                return getattr(first, "text", None), float(getattr(first, "weight", 1.0))

        pipeline = None
        try:
            pipeline = self.pipeline_manager.get_pipeline()
        except Exception:
            pipeline = None
        if pipeline is not None and hasattr(pipeline, "state"):
            state = getattr(pipeline, "state", None)
            state_prompts = None
            if hasattr(state, "get"):
                state_prompts = state.get("prompts")
            if isinstance(state_prompts, list) and state_prompts:
                first = state_prompts[0]
                if isinstance(first, dict):
                    return first.get("text"), float(first.get("weight", 1.0))

        compiled = getattr(self, "_compiled_prompt", None)
        if compiled is not None:
            cps = getattr(compiled, "prompts", None)
            if isinstance(cps, list) and cps:
                first = cps[0]
                if hasattr(first, "text"):
                    return getattr(first, "text", None), float(getattr(first, "weight", 1.0))
                if isinstance(first, dict):
                    return first.get("text"), float(first.get("weight", 1.0))

            pos = getattr(compiled, "positive", None)
            if isinstance(pos, list) and pos:
                first = pos[0]
                if isinstance(first, dict):
                    return first.get("text"), float(first.get("weight", 1.0))

            prompt_str = getattr(compiled, "prompt", None)
            if isinstance(prompt_str, str) and prompt_str.strip():
                return prompt_str, 1.0

        return None, 1.0

    def start(self):
        if self.running:
            return

        self.running = True
        self.shutdown_event.clear()

        # Process any Spout settings from initial parameters
        if "spout_sender" in self.parameters:
            spout_config = self.parameters.pop("spout_sender")
            self._update_spout_sender(spout_config)

        if "spout_receiver" in self.parameters:
            spout_config = self.parameters.pop("spout_receiver")
            self._update_spout_receiver(spout_config)

        # Load style manifests from styles/ directory
        try:
            self.style_registry.load_from_style_dirs()
            if len(self.style_registry) > 0:
                logger.info(
                    "Loaded %d styles: %s",
                    len(self.style_registry),
                    self.style_registry.list_styles(),
                )
        except Exception as e:
            logger.warning("Failed to load styles from style dirs: %s", e)

        if default_style := os.getenv("STYLE_DEFAULT"):
            style_name = default_style.strip()
            if style_name and self.style_registry.get(style_name):
                logger.info("STYLE_DEFAULT=%s: applying initial style", style_name)
                self.update_parameters({"_rcp_set_style": style_name})
            elif style_name:
                logger.warning(
                    "STYLE_DEFAULT=%s: style not found in registry, ignoring",
                    style_name,
                )

        self.worker_thread = threading.Thread(target=self.worker_loop, daemon=True)
        self.worker_thread.start()

        # Start control map worker for high-FPS preview (Phase 2.1a)
        enable_control_map_worker = _is_env_true(
            "SCOPE_VACE_CONTROL_MAP_WORKER", default="1"
        )
        auto_disable_above_pixels = os.getenv(
            "SCOPE_VACE_CONTROL_MAP_WORKER_AUTO_DISABLE_ABOVE_PIXELS", ""
        ).strip()
        if enable_control_map_worker and auto_disable_above_pixels:
            try:
                threshold_pixels = int(auto_disable_above_pixels)
            except ValueError:
                logger.warning(
                    "Invalid SCOPE_VACE_CONTROL_MAP_WORKER_AUTO_DISABLE_ABOVE_PIXELS=%r; expected int",
                    auto_disable_above_pixels,
                )
            else:
                if threshold_pixels > 0:
                    width, height = self._get_pipeline_dimensions()
                    pixels = int(width) * int(height)
                    if pixels >= threshold_pixels:
                        enable_control_map_worker = False
                        logger.info(
                            "ControlMapWorker auto-disabled for high-res: %dx%d (%d px) >= %d px "
                            "(SCOPE_VACE_CONTROL_MAP_WORKER_AUTO_DISABLE_ABOVE_PIXELS)",
                            width,
                            height,
                            pixels,
                            threshold_pixels,
                        )

        max_fps_env = os.getenv("SCOPE_VACE_CONTROL_MAP_WORKER_MAX_FPS", "").strip()
        if max_fps_env:
            try:
                max_fps = float(max_fps_env)
            except ValueError:
                logger.warning(
                    "Invalid SCOPE_VACE_CONTROL_MAP_WORKER_MAX_FPS=%r; expected float",
                    max_fps_env,
                )
            else:
                self._control_map_worker.set_max_fps(max_fps)
                logger.info(
                    "ControlMapWorker max_fps=%.2f (SCOPE_VACE_CONTROL_MAP_WORKER_MAX_FPS)",
                    max_fps,
                )

        if enable_control_map_worker:
            self._control_map_worker.start()
        else:
            logger.info("ControlMapWorker disabled")

        logger.info("FrameProcessor started")

    def stop(self, error_message: str = None):
        if not self.running:
            return

        self.running = False
        self.shutdown_event.set()

        if self.worker_thread and self.worker_thread.is_alive():
            # Don't join if we're calling stop() from within the worker thread
            if threading.current_thread() != self.worker_thread:
                self.worker_thread.join(timeout=5.0)

        self.flush_output_queue()

        with self.frame_buffer_lock:
            self.frame_buffer.clear()

        # Stop control map worker (Phase 2.1a)
        self._control_map_worker.stop()

        # Clean up Spout sender
        self.spout_sender_enabled = False
        if self.spout_sender_thread and self.spout_sender_thread.is_alive():
            # Signal thread to stop by putting None in queue
            try:
                self.spout_sender_queue.put_nowait(None)
            except queue.Full:
                pass
            self.spout_sender_thread.join(timeout=2.0)
        if self.spout_sender is not None:
            try:
                self.spout_sender.release()
            except Exception as e:
                logger.error(f"Error releasing Spout sender: {e}")
            self.spout_sender = None

        # Clean up Spout receiver
        self.spout_receiver_enabled = False
        if self.spout_receiver_thread and self.spout_receiver_thread.is_alive():
            if threading.current_thread() != self.spout_receiver_thread:
                self.spout_receiver_thread.join(timeout=2.0)
        self.spout_receiver_thread = None
        if self.spout_receiver is not None:
            try:
                self.spout_receiver.release()
            except Exception as e:
                logger.error(f"Error releasing Spout receiver: {e}")
            self.spout_receiver = None

        # Clean up NDI receiver
        self.ndi_receiver_enabled = False
        if self.ndi_receiver_thread and self.ndi_receiver_thread.is_alive():
            if threading.current_thread() != self.ndi_receiver_thread:
                self.ndi_receiver_thread.join(timeout=2.0)
        self.ndi_receiver_thread = None
        # Receiver resources are released by the receiver thread (see _ndi_receiver_loop()).
        self.ndi_receiver = None
        if self.get_active_input_source() == "ndi":
            self._set_active_input_source("webrtc")

        # Clear input frame times
        with self.input_fps_lock:
            self.input_frame_times.clear()

        logger.info("FrameProcessor stopped")

        # Notify callback that frame processor has stopped
        if self.notification_callback:
            try:
                message = {"type": "stream_stopped"}
                if error_message:
                    message["error_message"] = error_message
                self.notification_callback(message)
            except Exception as e:
                logger.error(f"Error in frame processor stop callback: {e}")

    def put(self, frame: VideoFrame) -> bool:
        if not self.running:
            return False

        if self.get_active_input_source() != "webrtc":
            # Ignore WebRTC input when another input source is active (e.g. Spout/NDI).
            # Do not call track_input_frame(): input_fps should reflect the active source.
            return False

        # Track input frame timestamp for FPS measurement
        self.track_input_frame()

        # Phase 2.1b: Assign a stable monotonic frame_id
        with self._frame_id_lock:
            frame_id = self._next_frame_id
            self._next_frame_id += 1

        wrapped = _FrameWithID(frame, frame_id)

        # Enqueue to control map worker (preview + generation ring buffer)
        # Non-blocking: worker may drop frames if it falls behind.
        self._control_map_worker.put(wrapped)

        with self.frame_buffer_lock:
            self.frame_buffer.append(wrapped)
            return True

    def get_active_input_source(self) -> str:
        with self._input_source_lock:
            return self._active_input_source

    def _set_active_input_source(self, source: str) -> None:
        normalized = (source or "").strip().lower()
        if normalized not in ("webrtc", "spout", "ndi"):
            logger.warning("Ignoring unknown input source %r", source)
            return
        with self._input_source_lock:
            self._active_input_source = normalized

    def flush_output_queue(self) -> int:
        """Flush all frames from output queue.

        Thread-safe: uses output_queue_lock to prevent race with queue resize.

        Returns:
            Number of frames flushed
        """
        count = 0
        with self.output_queue_lock:
            while True:
                try:
                    self.output_queue.get_nowait()
                    count += 1
                except queue.Empty:
                    break
        return count

    def get_latest_frame(self) -> torch.Tensor | None:
        """Get the most recent frame without consuming from output queue.

        Returns a clone of the latest frame, or None if no frames produced yet.
        Thread-safe: uses latest_frame_lock.
        """
        with self.latest_frame_lock:
            if self.latest_frame_cpu is not None:
                return self.latest_frame_cpu.clone()
            return None

    def set_latest_frame_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Set the asyncio loop used to signal latest-frame updates.

        FrameProcessor output is produced on a worker thread. Observers can await
        `wait_for_frame()` without polling by having the worker thread signal an
        asyncio.Event via `loop.call_soon_threadsafe(...)`.
        """
        self._latest_frame_event_loop = loop

    def _signal_latest_frame_available(self) -> None:
        """Best-effort signal to wake any `wait_for_frame()` awaiters."""
        loop = self._latest_frame_event_loop
        if loop is None:
            return
        try:
            loop.call_soon_threadsafe(self._latest_frame_event.set)
        except RuntimeError:
            # Loop may be closed during shutdown; observers will fall back to polling/timeouts.
            self._latest_frame_event_loop = None

    async def wait_for_frame(
        self,
        after_id: int,
        *,
        timeout: float = 0.1,
    ) -> tuple[torch.Tensor | None, int]:
        """Wait until latest_frame_id advances beyond `after_id`, or timeout.

        Returns:
            (latest_frame_clone_or_none, latest_frame_id)
        """
        if self._latest_frame_event_loop is None:
            try:
                self._latest_frame_event_loop = asyncio.get_running_loop()
            except RuntimeError:
                self._latest_frame_event_loop = None

        deadline = time.monotonic() + max(0.0, float(timeout))
        while True:
            with self.latest_frame_lock:
                current_id = int(self.latest_frame_id)
                if current_id > int(after_id):
                    frame = (
                        self.latest_frame_cpu.clone() if self.latest_frame_cpu is not None else None
                    )
                    return frame, current_id

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                with self.latest_frame_lock:
                    current_id = int(self.latest_frame_id)
                    frame = (
                        self.latest_frame_cpu.clone() if self.latest_frame_cpu is not None else None
                    )
                return frame, current_id

            # Avoid missing a signal by clearing + re-checking before awaiting.
            self._latest_frame_event.clear()
            with self.latest_frame_lock:
                current_id = int(self.latest_frame_id)
                if current_id > int(after_id):
                    continue

            try:
                await asyncio.wait_for(self._latest_frame_event.wait(), timeout=remaining)
            except asyncio.TimeoutError:
                # Loop back and return best-effort latest frame.
                continue

    def get_latest_control_frame(self) -> torch.Tensor | None:
        """Get the most recent VACE control frame for preview.

        Returns a clone of the latest control frame (H, W, 3) uint8,
        or None if no control frames generated yet.

        Phase 2.1a: Prefers worker output (high-FPS) over chunk output.
        Falls back to chunk output if worker hasn't produced anything yet.
        """
        if (self.parameters.get("vace_control_map_mode") or "none") == "none":
            return None

        # Prefer worker output (higher FPS)
        worker_frame = self._control_map_worker.get_latest()
        if worker_frame is not None:
            return worker_frame

        # Fallback to chunk-generated output
        with self.latest_control_frame_lock:
            if self.latest_control_frame_cpu is not None:
                return self.latest_control_frame_cpu.clone()
            return None

    def get(self) -> torch.Tensor | None:
        if not self.running:
            return None

        try:
            frame = self.output_queue.get_nowait()
            # Enqueue frame for async Spout sending (non-blocking)
            if self.spout_sender_enabled and self.spout_sender is not None:
                try:
                    # Frame is (H, W, C) uint8 [0, 255]
                    frame_np = frame.numpy()
                    self.spout_sender_queue.put_nowait(frame_np)
                except queue.Full:
                    # Queue full, drop frame (non-blocking)
                    logger.debug("Spout output queue full, dropping frame")
                except Exception as e:
                    logger.error(f"Error enqueueing Spout frame: {e}")

            return frame
        except queue.Empty:
            return None

    def get_current_pipeline_fps(self) -> float:
        """Get the current dynamically calculated pipeline FPS"""
        with self.fps_lock:
            return self.current_pipeline_fps

    def get_output_fps(self) -> float:
        """Get the output FPS that frames should be sent at.

        Returns the minimum of input FPS and pipeline FPS to ensure:
        1. We don't send frames faster than they were captured (maintains temporal accuracy)
        2. We don't try to output faster than the pipeline can produce (prevents frame starvation)
        """
        input_fps = self._get_input_fps()
        pipeline_fps = self.get_current_pipeline_fps()

        # In external control mode, input frames are control signals (not a "video clock").
        # Allow output to run at pipeline FPS, reusing control maps as needed.
        if self._ndi_external_hold_last_enabled() and self._ndi_hold_last_input_frame is not None:
            base_fps = pipeline_fps
        elif input_fps is None:
            base_fps = pipeline_fps
        else:
            # Use minimum to respect both input rate and pipeline capacity
            base_fps = min(input_fps, pipeline_fps)

        pacing_fps = self._get_output_pacing_fps()
        if pacing_fps is not None:
            return min(base_fps, pacing_fps)

        return base_fps

    def get_fps_debug(self) -> dict[str, object]:
        """Return a debug snapshot of the FPS signals used for WebRTC pacing.

        This is intended for diagnosing cases where the end-to-end server FPS
        differs from pipeline-only measurements.
        """
        with self.input_fps_lock:
            input_samples = len(self.input_frame_times)
            input_fps = (
                float(self.current_input_fps)
                if input_samples >= INPUT_FPS_MIN_SAMPLES
                else None
            )

        pipeline_fps = float(self.get_current_pipeline_fps())
        output_pacing_fps = self._get_output_pacing_fps()

        external_timebase = (
            self._ndi_external_hold_last_enabled() and self._ndi_hold_last_input_frame is not None
        )

        base_output_fps = pipeline_fps
        if not external_timebase and input_fps is not None:
            base_output_fps = float(min(input_fps, pipeline_fps))

        output_fps = float(self.get_output_fps())

        bottleneck = "pipeline_fps"
        if output_pacing_fps is not None and float(output_pacing_fps) <= float(base_output_fps):
            bottleneck = "output_pacing_fps"
        elif external_timebase:
            bottleneck = "pipeline_fps"
        elif input_fps is None:
            bottleneck = "pipeline_fps"
        elif input_fps <= pipeline_fps:
            bottleneck = "input_fps"
        else:
            bottleneck = "pipeline_fps"

        with self.output_queue_lock:
            output_queue_depth = int(self.output_queue.qsize())
            output_queue_max = int(self.output_queue.maxsize)

        with self.frame_buffer_lock:
            frame_buffer_depth = int(len(self.frame_buffer))

        ndi_last_frame_age_ms: float | None = None
        if self.ndi_last_frame_ts_s > 0:
            ndi_last_frame_age_ms = (time.monotonic() - float(self.ndi_last_frame_ts_s)) * 1000.0

        estimated_input_buffer_window_ms: float | None = None
        if input_fps is not None and input_fps > 0:
            estimated_input_buffer_window_ms = (frame_buffer_depth / input_fps) * 1000.0

        estimated_output_queue_window_ms: float | None = None
        if output_fps > 0:
            estimated_output_queue_window_ms = (output_queue_depth / output_fps) * 1000.0

        estimated_server_buffer_window_ms: float | None = None
        if (
            estimated_input_buffer_window_ms is not None
            and estimated_output_queue_window_ms is not None
        ):
            estimated_server_buffer_window_ms = (
                estimated_input_buffer_window_ms + estimated_output_queue_window_ms
            )

        try:
            parameters_queue_depth = int(self.parameters_queue.qsize())
        except Exception:
            parameters_queue_depth = 0

        backend_report: dict[str, object] | None = None

        return {
            "input_fps": input_fps,
            "input_fps_samples": input_samples,
            "pipeline_fps": pipeline_fps,
            "output_base_fps": float(base_output_fps),
            "output_fps": output_fps,
            "output_pacing_fps": float(output_pacing_fps)
            if output_pacing_fps is not None
            else None,
            "output_fps_bottleneck": bottleneck,
            "active_input_source": self.get_active_input_source(),
            "frame_buffer_depth": frame_buffer_depth,
            # Back-compat for scripts/check_fps.sh
            "frame_buffer_size": frame_buffer_depth,
            "output_queue_depth": output_queue_depth,
            # Back-compat for scripts/check_fps.sh
            "output_queue_size": output_queue_depth,
            "output_queue_max": output_queue_max,
            # Stopgap latency estimates (buffer windows only; excludes compute + network).
            "estimated_input_buffer_window_ms": estimated_input_buffer_window_ms,
            "estimated_output_queue_window_ms": estimated_output_queue_window_ms,
            "estimated_server_buffer_window_ms": estimated_server_buffer_window_ms,
            "output_frames_dropped": int(self.output_frames_dropped),
            "input_frames_dropped": int(self.input_frames_dropped),
            "low_latency_mode": bool(self._low_latency_mode),
            "low_latency_output_mode": bool(self._low_latency_output_mode),
            "output_queue_maxsize_cap": int(self._output_queue_maxsize_cap)
            if self._output_queue_maxsize_cap is not None
            else None,
            "parameters_queue_depth": parameters_queue_depth,
            "control_map_worker": self._control_map_worker.get_debug_info(),
            "backend_report": backend_report,
            "video_mode": bool(self._video_mode),
            "ndi": {
                "enabled": bool(self.ndi_receiver_enabled),
                "source": str(self.ndi_receiver_source or ""),
                "extra_ips": list(self.ndi_receiver_extra_ips) if self.ndi_receiver_extra_ips else None,
                "connected_source": str(self.ndi_connected_source) if self.ndi_connected_source else None,
                "connected_url": str(self.ndi_connected_url) if self.ndi_connected_url else None,
                "reconnects": int(self.ndi_reconnects),
                "frames_received": int(self.ndi_frames_received),
                "frames_dropped_during_drain": int(self.ndi_frames_dropped),
                "last_frame_age_ms": float(ndi_last_frame_age_ms) if ndi_last_frame_age_ms is not None else None,
                "frames_reused_total": int(self.ndi_frames_reused),
                "external_stale_ms": float(self._get_vace_external_stale_ms()),
                "external_input_stale": bool(self.external_input_stale),
                "external_resume_hard_cut": bool(self._get_vace_external_resume_hard_cut_enabled()),
            },
        }

    def _get_input_fps(self) -> float | None:
        """Get the current measured input FPS.

        Returns the measured input FPS if enough samples are available,
        otherwise returns None to indicate fallback should be used.
        """
        with self.input_fps_lock:
            if len(self.input_frame_times) < INPUT_FPS_MIN_SAMPLES:
                return None
            return self.current_input_fps

    def _calculate_input_fps(self):
        """Calculate and update input FPS from recent frame timestamps.

        Uses the same time-based update logic as pipeline FPS for consistency.
        Only updates if enough time has passed since the last update.
        """
        # Update FPS if enough time has passed
        current_time = time.time()
        if current_time - self.last_input_fps_update >= self.fps_update_interval:
            with self.input_fps_lock:
                if len(self.input_frame_times) >= INPUT_FPS_MIN_SAMPLES:
                    # Calculate FPS from frame intervals
                    times = list(self.input_frame_times)
                    if len(times) >= 2:
                        # Time span from first to last frame
                        time_span = times[-1] - times[0]
                        if time_span > 0:
                            # FPS = (number of intervals) / time_span
                            num_intervals = len(times) - 1
                            estimated_fps = num_intervals / time_span

                            # Clamp to reasonable bounds (same as pipeline FPS)
                            estimated_fps = max(
                                self.min_fps, min(self.max_fps, estimated_fps)
                            )
                            self.current_input_fps = estimated_fps

            self.last_input_fps_update = current_time

    def track_input_frame(self):
        """Track timestamp of an incoming frame for FPS measurement"""
        with self.input_fps_lock:
            self.input_frame_times.append(time.time())

        # Update input FPS calculation using same logic as pipeline FPS
        self._calculate_input_fps()

    def _calculate_pipeline_fps(self, start_time: float, num_frames: int):
        """Calculate FPS based on processing time and number of frames created"""
        processing_time = time.time() - start_time
        if processing_time <= 0 or num_frames <= 0:
            return

        # Store processing time per frame for averaging
        time_per_frame = processing_time / num_frames
        self.processing_time_per_frame.append(time_per_frame)

        # Update FPS if enough time has passed
        current_time = time.time()
        if current_time - self.last_fps_update >= self.fps_update_interval:
            if len(self.processing_time_per_frame) >= 1:
                # Calculate average processing time per frame
                avg_time_per_frame = sum(self.processing_time_per_frame) / len(
                    self.processing_time_per_frame
                )

                # Calculate FPS: 1 / average_time_per_frame
                # This gives us the actual frames per second output
                with self.fps_lock:
                    current_fps = self.current_pipeline_fps
                estimated_fps = (
                    1.0 / avg_time_per_frame if avg_time_per_frame > 0 else current_fps
                )

                # Clamp to reasonable bounds
                estimated_fps = max(self.min_fps, min(self.max_fps, estimated_fps))
                with self.fps_lock:
                    self.current_pipeline_fps = estimated_fps

            self.last_fps_update = current_time

    def _get_pipeline_dimensions(self) -> tuple[int, int]:
        """Get current pipeline dimensions from pipeline manager."""
        try:
            status_info = self.pipeline_manager.get_status_info()
            load_params = status_info.get("load_params") or {}
            width = load_params.get("width", 512)
            height = load_params.get("height", 512)
            return width, height
        except Exception as e:
            logger.warning(f"Could not get pipeline dimensions: {e}")
            return 512, 512

    def _apply_temporal_ema(
        self,
        control_frames: list[torch.Tensor],
        mode: str,
        ema: float,
        hard_cut: bool = False,
    ) -> list[torch.Tensor]:
        """Apply temporal EMA smoothing to control frames.

        Args:
            control_frames: List of control frame tensors (1, H, W, 3) float [0, 255]
            mode: Current control map mode (for detecting mode changes)
            ema: EMA momentum (0.0 = no smoothing, 0.9 = heavy smoothing)
            hard_cut: If True, reset EMA state (e.g., on cache reset)

        Returns:
            Smoothed control frames
        """
        if ema <= 0.0 or ema >= 1.0:
            # No smoothing or invalid value
            return control_frames

        # Reset on mode change or hard cut
        if hard_cut or mode != self._prev_control_map_mode:
            self._prev_control_frames = None
            self._prev_control_map_mode = mode

        # If no previous frames, just store current and return
        if self._prev_control_frames is None:
            self._prev_control_frames = [f.clone() for f in control_frames]
            return control_frames

        # Apply EMA: smoothed = ema * prev + (1 - ema) * current
        smoothed_frames = []
        for i, current in enumerate(control_frames):
            if i < len(self._prev_control_frames):
                prev = self._prev_control_frames[i]
                # Ensure same shape
                if prev.shape == current.shape:
                    smoothed = ema * prev + (1.0 - ema) * current
                else:
                    # Shape mismatch (resolution change), reset
                    smoothed = current
            else:
                smoothed = current
            smoothed_frames.append(smoothed)

        # Store for next iteration
        self._prev_control_frames = [f.clone() for f in smoothed_frames]

        return smoothed_frames

    def _try_sample_control_frames(
        self, frame_ids: list[int] | None
    ) -> list[torch.Tensor] | None:
        """Try to sample control frames from worker buffer with block+timeout policy.

        Phase 2.1b: Attempts to retrieve pre-computed control frames from the
        ControlMapWorker's ring buffer. If the frames aren't available yet,
        blocks briefly (configurable) then returns None to signal fallback
        to chunk-time compute.

        Args:
            frame_ids: List of frame IDs to sample, or None if not available.

        Returns:
            List of control frame tensors if all frame_ids are available,
            or None to signal fallback to chunk-time compute.
        """
        if not self._control_buffer_enabled:
            return None
        if frame_ids is None or not frame_ids:
            return None
        # Skip if all frame_ids are -1 (stub/unknown)
        if all(fid == -1 for fid in frame_ids):
            return None

        policy = (
            self.parameters.get("vace_control_buffer_missing_policy")
            or os.getenv("SCOPE_VACE_CONTROL_BUFFER_MISSING_POLICY", "block")
            or "block"
        ).strip().lower()

        timeout_s = float(
            self.parameters.get("vace_control_buffer_block_timeout_s")
            or os.getenv("SCOPE_VACE_CONTROL_BUFFER_BLOCK_TIMEOUT_S", "0.25")
            or "0.25"
        )

        t0 = time.perf_counter()
        while True:
            control_frames = self._control_map_worker.sample_control_frames(frame_ids)
            if control_frames is not None:
                return control_frames
            if not policy.startswith("block"):
                return None
            if self.shutdown_event.is_set():
                return None
            if (time.perf_counter() - t0) >= timeout_s:
                return None
            self.shutdown_event.wait(0.005)

    def update_parameters(self, parameters: dict[str, Any]) -> bool:
        """Update parameters that will be used in the next pipeline call.

        Returns:
            True if the update was queued successfully, False otherwise.
        """
        # Handle Spout output settings
        if "spout_sender" in parameters:
            spout_config = parameters.pop("spout_sender")
            self._update_spout_sender(spout_config)

        # Handle Spout input settings
        if "spout_receiver" in parameters:
            spout_config = parameters.pop("spout_receiver")
            self._update_spout_receiver(spout_config)

        # Handle NDI input settings
        if "ndi_receiver" in parameters:
            ndi_config = parameters.pop("ndi_receiver")
            self._update_ndi_receiver(ndi_config)

        # Put new parameters in queue with mailbox semantics:
        # If queue is full, drop oldest (not newest) to ensure latest control commands apply
        try:
            self.parameters_queue.put_nowait(parameters)
        except queue.Full:
            # Drop oldest to make room for newest (mailbox semantics)
            try:
                self.parameters_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.parameters_queue.put_nowait(parameters)
            except queue.Full:
                logger.warning("Parameter queue still full after dropping oldest")
                return False
        return True

    def _update_spout_sender(self, config: dict):
        """Update Spout output configuration."""
        logger.info(f"Spout output config received: {config}")

        enabled = config.get("enabled", False)
        sender_name = config.get("name", "ScopeSyphonSpoutOut")

        # Get dimensions from active pipeline
        width, height = self._get_pipeline_dimensions()

        logger.info(
            f"Spout output: enabled={enabled}, name={sender_name}, size={width}x{height}"
        )

        # Lazy import SpoutSender
        try:
            from scope.server.spout import SpoutSender
        except ImportError:
            if enabled:
                logger.warning("Spout module not available on this platform")
            return

        if enabled and not self.spout_sender_enabled:
            # Enable Spout output
            try:
                self.spout_sender = SpoutSender(sender_name, width, height)
                if self.spout_sender.create():
                    self.spout_sender_enabled = True
                    self.spout_sender_name = sender_name
                    # Start background thread for async sending
                    if (
                        self.spout_sender_thread is None
                        or not self.spout_sender_thread.is_alive()
                    ):
                        self.spout_sender_thread = threading.Thread(
                            target=self._spout_sender_loop, daemon=True
                        )
                        self.spout_sender_thread.start()
                    logger.info(f"Spout output enabled: '{sender_name}'")
                else:
                    logger.error("Failed to create Spout sender")
                    self.spout_sender = None
            except Exception as e:
                logger.error(f"Error creating Spout sender: {e}")
                self.spout_sender = None

        elif not enabled and self.spout_sender_enabled:
            # Disable Spout output
            if self.spout_sender is not None:
                self.spout_sender.release()
                self.spout_sender = None
            self.spout_sender_enabled = False
            logger.info("Spout output disabled")

        elif enabled and (
            sender_name != self.spout_sender_name
            or (
                self.spout_sender
                and (
                    self.spout_sender.width != width
                    or self.spout_sender.height != height
                )
            )
        ):
            # Name or dimensions changed, recreate sender
            if self.spout_sender is not None:
                self.spout_sender.release()
            try:
                self.spout_sender = SpoutSender(sender_name, width, height)
                if self.spout_sender.create():
                    self.spout_sender_name = sender_name
                    # Ensure output thread is running
                    if (
                        self.spout_sender_thread is None
                        or not self.spout_sender_thread.is_alive()
                    ):
                        self.spout_sender_thread = threading.Thread(
                            target=self._spout_sender_loop, daemon=True
                        )
                        self.spout_sender_thread.start()
                    logger.info(
                        f"Spout output updated: '{sender_name}' ({width}x{height})"
                    )
                else:
                    logger.error("Failed to recreate Spout sender")
                    self.spout_sender = None
                    self.spout_sender_enabled = False
            except Exception as e:
                logger.error(f"Error recreating Spout sender: {e}")
                self.spout_sender = None
                self.spout_sender_enabled = False

    def _update_spout_receiver(self, config: dict):
        """Update Spout input configuration."""
        enabled = config.get("enabled", False)
        sender_name = config.get("name", "")

        # Lazy import SpoutReceiver
        try:
            from scope.server.spout import SpoutReceiver
        except ImportError:
            if enabled:
                logger.warning("Spout module not available on this platform")
            return

        def stop_receiver_thread() -> None:
            self.spout_receiver_enabled = False
            if self.spout_receiver_thread and self.spout_receiver_thread.is_alive():
                if threading.current_thread() != self.spout_receiver_thread:
                    self.spout_receiver_thread.join(timeout=2.0)
            self.spout_receiver_thread = None

        if enabled and not self.spout_receiver_enabled:
            # Enable Spout input
            try:
                self.spout_receiver = SpoutReceiver(sender_name, 512, 512)
                if self.spout_receiver.create():
                    self.spout_receiver_enabled = True
                    self.spout_receiver_name = sender_name
                    self._set_active_input_source("spout")
                    # Start receiving thread
                    self.spout_receiver_thread = threading.Thread(
                        target=self._spout_receiver_loop, daemon=True
                    )
                    self.spout_receiver_thread.start()
                    logger.info(f"Spout input enabled: '{sender_name or 'any'}'")
                else:
                    logger.error("Failed to create Spout receiver")
                    self.spout_receiver = None
            except Exception as e:
                logger.error(f"Error creating Spout receiver: {e}")
                self.spout_receiver = None

        elif not enabled and self.spout_receiver_enabled:
            # Disable Spout input
            stop_receiver_thread()
            if self.spout_receiver is not None:
                self.spout_receiver.release()
                self.spout_receiver = None
            if self.get_active_input_source() == "spout":
                self._set_active_input_source("webrtc")
            logger.info("Spout input disabled")

        elif enabled and sender_name != self.spout_receiver_name:
            # Name changed, recreate receiver
            stop_receiver_thread()
            if self.spout_receiver is not None:
                self.spout_receiver.release()
            try:
                self.spout_receiver = SpoutReceiver(sender_name, 512, 512)
                if self.spout_receiver.create():
                    self.spout_receiver_enabled = True
                    self.spout_receiver_name = sender_name
                    self._set_active_input_source("spout")
                    # Restart receiving thread
                    self.spout_receiver_thread = threading.Thread(
                        target=self._spout_receiver_loop, daemon=True
                    )
                    self.spout_receiver_thread.start()
                    logger.info(f"Spout input changed to: '{sender_name or 'any'}'")
                else:
                    logger.error("Failed to recreate Spout receiver")
                    self.spout_receiver = None
                    if self.get_active_input_source() == "spout":
                        self._set_active_input_source("webrtc")
            except Exception as e:
                logger.error(f"Error recreating Spout receiver: {e}")
                self.spout_receiver = None
                if self.get_active_input_source() == "spout":
                    self._set_active_input_source("webrtc")
        elif enabled and self.spout_receiver_enabled:
            # Receiver enabled but thread may have died; ensure it is running.
            if self.spout_receiver is not None and (
                self.spout_receiver_thread is None or not self.spout_receiver_thread.is_alive()
            ):
                self._set_active_input_source("spout")
                self.spout_receiver_thread = threading.Thread(
                    target=self._spout_receiver_loop, daemon=True
                )
                self.spout_receiver_thread.start()

    def _update_ndi_receiver(self, config: dict):
        """Update NDI input configuration."""
        enabled = bool(config.get("enabled", False))
        source = str(config.get("source", "") or "")
        extra_ips_raw = config.get("extra_ips", None)
        extra_ips: list[str] | None = None
        if extra_ips_raw is not None:
            if isinstance(extra_ips_raw, str):
                extra_ips = [s.strip() for s in extra_ips_raw.split(",") if s.strip()]
            elif isinstance(extra_ips_raw, list):
                extra_ips = [str(s).strip() for s in extra_ips_raw if str(s).strip()]

        def stop_receiver_thread() -> None:
            self.ndi_receiver_enabled = False
            self._ndi_hold_last_input_frame = None
            self._ndi_hold_last_input_frame_id = None
            self.external_input_stale = False
            self._external_resume_hard_cut_pending = False
            self.ndi_connected_source = None
            self.ndi_connected_url = None
            if self.ndi_receiver_thread and self.ndi_receiver_thread.is_alive():
                if threading.current_thread() != self.ndi_receiver_thread:
                    self.ndi_receiver_thread.join(timeout=2.0)
            self.ndi_receiver_thread = None

        if enabled and not self.ndi_receiver_enabled:
            self.ndi_receiver_enabled = True
            self.ndi_receiver_source = source
            self.ndi_receiver_extra_ips = extra_ips
            self._ndi_hold_last_input_frame = None
            self._ndi_hold_last_input_frame_id = None
            self.external_input_stale = False
            self._external_resume_hard_cut_pending = False
            self.ndi_frames_reused = 0
            self.ndi_connected_source = None
            self.ndi_connected_url = None
            self.ndi_reconnects = 0
            self._set_active_input_source("ndi")

            self.ndi_receiver_thread = threading.Thread(
                target=self._ndi_receiver_loop,
                daemon=True,
            )
            self.ndi_receiver_thread.start()
            logger.info("NDI input enabled (source=%r, extra_ips=%r)", source, extra_ips)

        elif not enabled and self.ndi_receiver_enabled:
            stop_receiver_thread()
            if self.get_active_input_source() == "ndi":
                self._set_active_input_source("webrtc")
            logger.info("NDI input disabled")

        elif enabled and self.ndi_receiver_enabled:
            # Config changed: restart receiver.
            if source != self.ndi_receiver_source or extra_ips != self.ndi_receiver_extra_ips:
                stop_receiver_thread()
                self.ndi_receiver_enabled = True
                self.ndi_receiver_source = source
                self.ndi_receiver_extra_ips = extra_ips
                self._ndi_hold_last_input_frame = None
                self._ndi_hold_last_input_frame_id = None
                self.external_input_stale = False
                self._external_resume_hard_cut_pending = False
                self.ndi_frames_reused = 0
                self.ndi_connected_source = None
                self.ndi_connected_url = None
                self.ndi_reconnects = 0
                self._set_active_input_source("ndi")

                self.ndi_receiver_thread = threading.Thread(
                    target=self._ndi_receiver_loop,
                    daemon=True,
                )
                self.ndi_receiver_thread.start()
                logger.info(
                    "NDI input reconfigured (source=%r, extra_ips=%r)", source, extra_ips
                )

            # Receiver enabled but thread may have died; ensure it is running.
            elif self.ndi_receiver_thread is None or not self.ndi_receiver_thread.is_alive():
                self._set_active_input_source("ndi")
                self.ndi_receiver_thread = threading.Thread(
                    target=self._ndi_receiver_loop,
                    daemon=True,
                )
                self.ndi_receiver_thread.start()

    def _spout_sender_loop(self):
        """Background thread that sends frames to Spout asynchronously."""
        logger.info("Spout output thread started")
        frame_count = 0

        while (
            self.running and self.spout_sender_enabled and self.spout_sender is not None
        ):
            try:
                # Get frame from queue (blocking with timeout)
                try:
                    frame_np = self.spout_sender_queue.get(timeout=0.1)
                    # None is a sentinel value to stop the thread
                    if frame_np is None:
                        break
                except queue.Empty:
                    continue

                # Send frame to Spout
                success = self.spout_sender.send(frame_np)
                frame_count += 1
                if frame_count % 100 == 0:
                    logger.info(
                        f"Spout sent frame {frame_count}, "
                        f"shape={frame_np.shape}, success={success}"
                    )
                self._frame_spout_count = frame_count

            except Exception as e:
                logger.error(f"Error in Spout output loop: {e}")
                time.sleep(0.01)

        logger.info(f"Spout output thread stopped after {frame_count} frames")

    def _spout_receiver_loop(self):
        """Background thread that receives frames from Spout and adds to buffer."""
        logger.info("Spout input thread started")

        # Initial target frame rate
        target_fps = self.get_current_pipeline_fps()
        frame_interval = 1.0 / target_fps
        last_frame_time = 0.0
        frame_count = 0

        while (
            self.running
            and self.spout_receiver_enabled
            and self.spout_receiver is not None
        ):
            try:
                # Update target FPS dynamically from pipeline performance
                current_pipeline_fps = self.get_current_pipeline_fps()
                if current_pipeline_fps > 0:
                    target_fps = current_pipeline_fps
                    frame_interval = 1.0 / target_fps

                current_time = time.time()

                # Frame rate limiting - don't receive faster than target FPS
                time_since_last = current_time - last_frame_time
                if time_since_last < frame_interval:
                    time.sleep(frame_interval - time_since_last)
                    continue

                if self.get_active_input_source() != "spout":
                    # Avoid mixing sources: keep receiver loop alive but do not feed the buffer.
                    time.sleep(0.01)
                    continue

                # Receive directly as RGB (avoids extra copy from RGBA slice)
                rgb_frame = self.spout_receiver.receive(as_rgb=True)
                if rgb_frame is not None:
                    last_frame_time = time.time()

                    # Phase 2.1b: track input FPS and assign frame_id
                    self.track_input_frame()

                    with self._frame_id_lock:
                        frame_id = self._next_frame_id
                        self._next_frame_id += 1

                    # Wrap in _FrameWithID for buffer sampling alignment
                    wrapped_frame = _FrameWithID(_SpoutFrame(rgb_frame), frame_id)

                    # Enqueue to control map worker (preview + generation buffer)
                    self._control_map_worker.put(wrapped_frame)

                    with self.frame_buffer_lock:
                        self.frame_buffer.append(wrapped_frame)

                    frame_count += 1
                    if frame_count % 100 == 0:
                        logger.debug(f"Spout input received {frame_count} frames")
                else:
                    time.sleep(0.001)  # Small sleep when no frame available

            except Exception as e:
                logger.error(f"Error in Spout input loop: {e}")
                time.sleep(0.01)

        logger.info(f"Spout input thread stopped after {frame_count} frames")

    def _ndi_receiver_loop(self):
        """Background thread that receives frames from NDI and adds to buffer."""
        logger.info("NDI input thread started")

        frame_count = 0

        try:
            from scope.server.ndi import NDIReceiver

            receiver = NDIReceiver(recv_name="ScopeNDIRecv")
            self.ndi_receiver = receiver

            if not receiver.create():
                logger.error("NDI receiver create() failed")
                return

            # Discover + connect (retry until disabled/stopped)
            while self.running and self.ndi_receiver_enabled:
                try:
                    src = receiver.connect_discovered(
                        source_substring=self.ndi_receiver_source,
                        extra_ips=self.ndi_receiver_extra_ips,
                        timeout_ms=1500,
                    )
                    logger.info("NDI connected: %s (%s)", src.name, src.url_address)
                    self.ndi_connected_source = src.name
                    self.ndi_connected_url = src.url_address
                    self.ndi_reconnects += 1
                    break
                except Exception as e:
                    logger.warning("NDI connect failed: %s", e)
                    time.sleep(0.5)

            while self.running and self.ndi_receiver_enabled:
                if self.get_active_input_source() != "ndi":
                    time.sleep(0.01)
                    continue

                try:
                    rgb_frame = receiver.receive_latest_rgb24(timeout_ms=50)
                except Exception as e:
                    logger.warning("NDI receive error: %s", e)
                    time.sleep(0.05)
                    continue

                if rgb_frame is None:
                    continue

                # Update stats
                stats = receiver.get_stats()
                self.ndi_frames_received = stats.frames_received
                self.ndi_frames_dropped = stats.frames_dropped_during_drain
                self.ndi_last_frame_ts_s = stats.last_frame_ts_s

                # Track input frame timestamp for FPS measurement
                self.track_input_frame()

                with self._frame_id_lock:
                    frame_id = self._next_frame_id
                    self._next_frame_id += 1

                wrapped_frame = _FrameWithID(_NDIFrame(rgb_frame), frame_id)

                # Enqueue to control map worker (preview + generation ring buffer)
                self._control_map_worker.put(wrapped_frame)

                with self.frame_buffer_lock:
                    self.frame_buffer.append(wrapped_frame)

                frame_count += 1
        finally:
            try:
                if self.ndi_receiver is not None:
                    self.ndi_receiver.release()
            except Exception as e:
                logger.warning("Failed to release NDI receiver: %s", e)
            self.ndi_receiver = None
            logger.info("NDI input thread stopped after %d frames", frame_count)

    def worker_loop(self):
        logger.info("Worker thread started")

        while self.running and not self.shutdown_event.is_set():
            try:
                self.process_chunk()

            except PipelineNotAvailableException as e:
                logger.debug(f"Pipeline temporarily unavailable: {e}")
                # Flush frame buffer to prevent buildup
                with self.frame_buffer_lock:
                    if self.frame_buffer:
                        logger.debug(
                            f"Flushing {len(self.frame_buffer)} frames due to pipeline unavailability"
                        )
                        self.frame_buffer.clear()
                continue
            except Exception as e:
                if self._is_recoverable(e):
                    logger.error(f"Error in worker loop: {e}")
                    continue
                else:
                    logger.error(
                        f"Non-recoverable error in worker loop: {e}, stopping frame processor"
                    )
                    self.stop(error_message=str(e))
                    break
        logger.info("Worker thread stopped")

    def process_chunk(self):
        start_time = time.time()

        # Legacy safety: ensure we don't persist "paused" inside self.parameters.
        # Pause state is tracked separately in self.paused and updated via events.
        paused = self.parameters.pop("paused", None)
        if paused is not None and paused != self.paused:
            self.paused = paused

        # ========================================================================
        # INGEST: Drain ALL pending queue entries (mailbox semantics)
        # ========================================================================
        # Intentional behavior change from "drain 1" to "drain all":
        # - Old: at most 1 update per chunk (10 rapid updates → 10 chunks to apply)
        # - New: all pending updates per chunk (commit at boundary)
        merged_updates: dict = {}
        while True:
            try:
                update = self.parameters_queue.get_nowait()
                # Last-write-wins merge
                merged_updates = {**merged_updates, **update}
            except queue.Empty:
                break

        # ========================================================================
        # RESERVED KEYS: Handle snapshot/restore commands (not forwarded to pipeline)
        # ========================================================================
        # These reserved keys route through parameters_queue for thread safety,
        # but are consumed here and never forwarded to the pipeline or events.
        if "_rcp_snapshot_request" in merged_updates:
            merged_updates.pop("_rcp_snapshot_request")
            try:
                snapshot = self._create_snapshot()
                # Send response via callback if registered
                if self.snapshot_response_callback:
                    self.snapshot_response_callback(
                        {
                            "type": "snapshot_response",
                            "snapshot_id": snapshot.snapshot_id,
                            "chunk_index": snapshot.chunk_index,
                            "current_start_frame": snapshot.current_start_frame,
                        }
                    )
            except Exception as e:
                logger.error(f"Error creating snapshot: {e}")
                if self.snapshot_response_callback:
                    self.snapshot_response_callback(
                        {"type": "snapshot_response", "error": str(e)}
                    )

        if "_rcp_restore_snapshot" in merged_updates:
            restore_data = merged_updates.pop("_rcp_restore_snapshot")
            snapshot_id = restore_data.get("snapshot_id") if restore_data else None
            if snapshot_id:
                success = self._restore_snapshot(snapshot_id)
                if self.snapshot_response_callback:
                    self.snapshot_response_callback(
                        {
                            "type": "restore_response",
                            "snapshot_id": snapshot_id,
                            "success": success,
                        }
                    )
            else:
                logger.warning("restore_snapshot called without snapshot_id")
                if self.snapshot_response_callback:
                    self.snapshot_response_callback(
                        {
                            "type": "restore_response",
                            "error": "snapshot_id required",
                            "success": False,
                        }
                    )

        # Step: generate exactly one chunk even while paused.
        # Keep a small backlog so step isn't dropped when input frames aren't ready.
        if "_rcp_step" in merged_updates:
            step_val = merged_updates.pop("_rcp_step")
            step_count = 1
            if isinstance(step_val, int) and not isinstance(step_val, bool):
                step_count = max(1, step_val)
            self._pending_steps += step_count

        # Session recording start/stop (consumed here; never forwarded to pipeline)
        if "_rcp_session_recording_start" in merged_updates:
            merged_updates.pop("_rcp_session_recording_start", None)
            try:
                status = (
                    self.pipeline_manager.peek_status_info()
                    if hasattr(self.pipeline_manager, "peek_status_info")
                    else self.pipeline_manager.get_status_info()
                )
            except Exception as e:
                logger.warning(
                    "Session recording start: failed to read pipeline status: %s", e
                )
                status = {}

            if status.get("status") != "loaded":
                logger.warning(
                    "Session recording start ignored: pipeline not loaded (status=%s)",
                    status.get("status"),
                )
            else:
                pipeline_id = status.get("pipeline_id")
                if not pipeline_id:
                    logger.warning(
                        "Session recording start ignored: missing pipeline_id in status"
                    )
                else:
                    lp = status.get("load_params") or {}
                    runtime_params: dict[str, Any] = (
                        dict(lp) if isinstance(lp, dict) else {"load_params": lp}
                    )

                    # Include key runtime params for timeline settings/replay
                    if "kv_cache_attention_bias" in self.parameters:
                        runtime_params["kv_cache_attention_bias"] = self.parameters.get(
                            "kv_cache_attention_bias"
                        )
                    if "denoising_step_list" in self.parameters:
                        runtime_params["denoising_step_list"] = self.parameters.get(
                            "denoising_step_list"
                        )
                    if "seed" not in runtime_params:
                        if "seed" in self.parameters:
                            runtime_params["seed"] = self.parameters.get("seed")
                        elif "base_seed" in self.parameters:
                            runtime_params["seed"] = self.parameters.get("base_seed")

                    baseline_prompt, baseline_weight = self._get_current_effective_prompt()
                    try:
                        self.session_recorder.start(
                            chunk_index=self.chunk_index,
                            pipeline_id=pipeline_id,
                            load_params=runtime_params,
                            baseline_prompt=baseline_prompt,
                            baseline_weight=baseline_weight,
                        )
                        self._last_recording_path = None
                        self._soft_transition_record_pending = bool(
                            self._soft_transition_active
                        )
                        logger.info(
                            "Session recording started at chunk=%d", self.chunk_index
                        )
                    except Exception as e:
                        logger.error("Session recording start failed: %s", e)

        if "_rcp_session_recording_stop" in merged_updates:
            merged_updates.pop("_rcp_session_recording_stop", None)
            try:
                recording = self.session_recorder.stop(chunk_index=self.chunk_index)
            except Exception as e:
                logger.error("Session recording stop failed: %s", e)
                recording = None

            if recording is not None:
                ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
                path = (
                    Path.home()
                    / ".daydream-scope"
                    / "recordings"
                    / f"session_{ts}.timeline.json"
                )
                try:
                    saved = self.session_recorder.save(recording, path)
                    self._last_recording_path = saved
                    logger.info("Session recording saved: %s", saved)
                except Exception as e:
                    logger.error("Failed to save session recording timeline: %s", e)

        # Soft transition: temporarily lower KV cache bias for N chunks
        if "_rcp_soft_transition" in merged_updates:
            soft_data = merged_updates.pop("_rcp_soft_transition")
            if isinstance(soft_data, dict):
                temp_bias = soft_data.get("temp_bias", 0.1)
                num_chunks = soft_data.get("num_chunks", 2)

                # Handle precedence: if explicit kv_cache_attention_bias in same message,
                # treat it as the base bias to restore to (and don't let it override temp)
                explicit_bias = merged_updates.pop("kv_cache_attention_bias", None)

                # Coerce + clamp inputs (avoid log(<=0) downstream)
                try:
                    temp_bias = float(temp_bias)
                except (TypeError, ValueError):
                    temp_bias = 0.1
                temp_bias = max(0.01, min(temp_bias, 1.0))

                try:
                    num_chunks = int(num_chunks)
                except (TypeError, ValueError):
                    num_chunks = 2
                num_chunks = max(1, min(num_chunks, 10))

                if explicit_bias is not None:
                    try:
                        explicit_bias = float(explicit_bias)
                    except (TypeError, ValueError):
                        explicit_bias = None
                    if explicit_bias is not None:
                        explicit_bias = max(0.01, min(explicit_bias, 1.0))

                # Re-entrancy: don't overwrite original if already in soft transition
                if not self._soft_transition_active:
                    # First trigger: save current bias as original
                    if explicit_bias is not None:
                        self._soft_transition_original_bias = explicit_bias
                        self._soft_transition_original_bias_was_set = True
                    else:
                        # Preserve "unset": if the key wasn't present, restore by deleting it.
                        if "kv_cache_attention_bias" in self.parameters:
                            self._soft_transition_original_bias = self.parameters.get(
                                "kv_cache_attention_bias"
                            )
                            self._soft_transition_original_bias_was_set = True
                        else:
                            self._soft_transition_original_bias = None
                            self._soft_transition_original_bias_was_set = False
                elif explicit_bias is not None:
                    # Re-trigger with explicit bias: update restore target
                    self._soft_transition_original_bias = explicit_bias
                    self._soft_transition_original_bias_was_set = True

                # (Re)start countdown
                self._soft_transition_temp_bias = temp_bias
                self._soft_transition_chunks_remaining = num_chunks
                self._soft_transition_active = True

                # Apply temporary bias immediately
                self.parameters["kv_cache_attention_bias"] = temp_bias
                self._soft_transition_record_pending = True
                logger.info(
                    f"Soft transition: bias -> {temp_bias} for {num_chunks} chunks "
                    f"(will restore to "
                    f"{self._soft_transition_original_bias if self._soft_transition_original_bias_was_set else '<unset>'})"
                )

        # If an explicit bias update arrives while a soft transition is active (and it wasn't
        # consumed above), treat it as an override and cancel the soft transition so we
        # don't later restore over the user's explicit change.
        if self._soft_transition_active and "kv_cache_attention_bias" in merged_updates:
            logger.info(
                "Soft transition canceled: explicit kv_cache_attention_bias update received"
            )
            self._soft_transition_active = False
            self._soft_transition_chunks_remaining = 0
            self._soft_transition_temp_bias = None
            self._soft_transition_original_bias = None
            self._soft_transition_original_bias_was_set = False
            self._soft_transition_record_pending = False

        # Track if explicit prompts were set this chunk (for precedence)
        explicit_prompts_set = "prompts" in merged_updates

        # Handle world state update (full replace, thread-safe via model_validate)
        if "_rcp_world_state" in merged_updates:
            world_data = merged_updates.pop("_rcp_world_state")
            try:
                self.world_state = WorldState.model_validate(world_data)
                logger.debug(f"WorldState updated: action={self.world_state.action}")

                # Recompile if style active and no explicit prompts
                if self.style_manifest and not explicit_prompts_set:
                    compiled = self.prompt_compiler.compile(
                        self.world_state, self.style_manifest
                    )
                    self._compiled_prompt = compiled
                    # Inject compiled prompts into merged_updates for event processing
                    merged_updates["prompts"] = [p.to_dict() for p in compiled.prompts]
                    logger.debug(f"Auto-compiled prompt: {compiled.prompt[:80]}...")
                    # Note: LoRA NOT re-sent here (only on style change)
            except Exception as e:
                logger.warning(f"Failed to validate WorldState: {e}")

        # Handle style change
        if "_rcp_set_style" in merged_updates:
            style_name = merged_updates.pop("_rcp_set_style")
            new_style = self.style_registry.get(style_name)
            if new_style:
                style_changed = style_name != self._active_style_name

                self.style_manifest = new_style
                self._active_style_name = style_name
                logger.info(f"Active style set to: {style_name}")

                # Recreate compiler for the new style (may switch to LLM if available)
                if style_changed:
                    try:
                        self.prompt_compiler = create_compiler(new_style)
                    except Exception as e:
                        logger.warning(
                            f"Failed to create compiler for style {style_name}: {e}, "
                            "keeping current compiler"
                        )

                # Recompile with new style - but only if WorldState has content.
                # In performance mode (empty WorldState), preserve the current prompt.
                if self.world_state.is_empty():
                    logger.info(
                        "WorldState empty - preserving current prompt (performance mode)"
                    )
                else:
                    compiled = self.prompt_compiler.compile(
                        self.world_state, self.style_manifest
                    )
                    self._compiled_prompt = compiled

                    if not explicit_prompts_set:
                        merged_updates["prompts"] = [p.to_dict() for p in compiled.prompts]

                # LoRA only on style change (edge-trigger)
                if style_changed:
                    # Reset cache for clean transition, or skip for blend artifacts
                    if self.reset_cache_on_style_switch:
                        merged_updates["reset_cache"] = True
                    else:
                        logger.info(
                            "Style switch without cache reset (blend mode enabled)"
                        )

                    # Canonicalize paths and dedupe updates (styles may share the same LoRA).
                    lora_updates = self.style_registry.build_lora_scales_for_style(
                        style_name
                    )
                    if lora_updates:
                        merged_updates["lora_scales"] = lora_updates
                        # When blend mode is enabled, tell pipeline to skip cache reset on LoRA scale change
                        if not self.reset_cache_on_style_switch:
                            merged_updates["lora_scales_skip_cache_reset"] = True
                        logger.info(
                            "LoRA scales updated for style '%s' (%d paths)",
                            style_name,
                            len(lora_updates),
                        )
            else:
                logger.warning(f"Style not found in registry: {style_name}")

        step_requested = self._pending_steps > 0

        # ========================================================================
        # TRANSLATE: Convert dict updates to typed events for ordering
        # ========================================================================
        if merged_updates:
            # VACE control-map mode changes (e.g. raw video -> depth) should default
            # to a hard cut. Otherwise the KV cache can retain prior video-derived
            # appearance information and "leak" it after the switch.
            if "vace_control_map_mode" in merged_updates:
                prev_mode = self.parameters.get("vace_control_map_mode", "none")
                next_mode = merged_updates.get("vace_control_map_mode") or "none"
                if next_mode != prev_mode:
                    # Avoid stale previews across mode switches (e.g. canny -> depth).
                    with self.latest_control_frame_lock:
                        self.latest_control_frame_cpu = None
                        self._control_map_worker.clear_latest(lock_held=True)

                    # Allow callers to explicitly override by sending reset_cache.
                    if "reset_cache" not in merged_updates:
                        merged_updates["reset_cache"] = True
                        logger.info(
                            "VACE control-map mode change: %s -> %s (forcing reset_cache=True)",
                            prev_mode,
                            next_mode,
                        )

            if "vace_depth_temporal_mode" in merged_updates:
                prev_mode = (
                    (self.parameters.get("vace_depth_temporal_mode") or "stream")
                    .strip()
                    .lower()
                )
                next_mode = (
                    (merged_updates.get("vace_depth_temporal_mode") or "stream")
                    .strip()
                    .lower()
                )
                if next_mode not in ("stream", "stateless"):
                    logger.warning(
                        "Ignoring invalid vace_depth_temporal_mode=%r; expected 'stream' or 'stateless'",
                        next_mode,
                    )
                elif next_mode != prev_mode:
                    with self.latest_control_frame_lock:
                        self.latest_control_frame_cpu = None
                        self._control_map_worker.clear_latest(lock_held=True)
                    if "reset_cache" not in merged_updates:
                        merged_updates["reset_cache"] = True
                        logger.info(
                            "VACE depth temporal mode change: %s -> %s (forcing reset_cache=True)",
                            prev_mode,
                            next_mode,
                        )

            if "vace_control_buffer_enabled" in merged_updates:
                raw = merged_updates.get("vace_control_buffer_enabled")
                if isinstance(raw, str):
                    next_enabled = raw.strip().lower() in ("1", "true", "yes", "on")
                else:
                    next_enabled = bool(raw)
                prev_enabled = bool(self._control_buffer_enabled)
                if next_enabled != prev_enabled:
                    if "reset_cache" not in merged_updates:
                        merged_updates["reset_cache"] = True
                        logger.info(
                            "VACE control buffer enabled: %s -> %s (forcing reset_cache=True)",
                            prev_enabled,
                            next_enabled,
                        )
                self._control_buffer_enabled = next_enabled

            if "vace_control_map_worker_enabled" in merged_updates:
                raw = merged_updates.get("vace_control_map_worker_enabled")
                if isinstance(raw, str):
                    worker_enabled = raw.strip().lower() in ("1", "true", "yes", "on")
                else:
                    worker_enabled = bool(raw)
                if worker_enabled:
                    self._control_map_worker.start()
                else:
                    self._control_map_worker.stop()
                    with self.latest_control_frame_lock:
                        self.latest_control_frame_cpu = None
                        self._control_map_worker.clear_latest(lock_held=True)

            if "vace_control_map_worker_allow_heavy" in merged_updates:
                raw = merged_updates.get("vace_control_map_worker_allow_heavy")
                if isinstance(raw, str):
                    allow_heavy = raw.strip().lower() in ("1", "true", "yes", "on")
                else:
                    allow_heavy = bool(raw)
                self._control_map_worker.set_allow_heavy(allow_heavy)

            if "vace_control_map_worker_max_fps" in merged_updates:
                raw = merged_updates.get("vace_control_map_worker_max_fps")
                max_fps: float | None
                if raw is None:
                    max_fps = None
                else:
                    try:
                        max_fps = float(raw)
                    except (TypeError, ValueError):
                        logger.warning(
                            "Ignoring invalid vace_control_map_worker_max_fps=%r; expected float",
                            raw,
                        )
                        max_fps = None
                self._control_map_worker.set_max_fps(max_fps)

            # Handle pause/resume via events
            if "paused" in merged_updates:
                paused_val = merged_updates.pop("paused")
                if paused_val:
                    self.control_bus.enqueue(EventType.PAUSE)
                else:
                    self.control_bus.enqueue(EventType.RESUME)

            # Handle prompts/transition via events
            if "prompts" in merged_updates or "transition" in merged_updates:
                payload = {}
                if "prompts" in merged_updates:
                    payload["prompts"] = merged_updates.pop("prompts")
                if "transition" in merged_updates:
                    payload["transition"] = merged_updates.pop("transition")
                self.control_bus.enqueue(EventType.SET_PROMPT, payload=payload)

            # Handle lora_scales via events
            if "lora_scales" in merged_updates:
                lora_payload = {"lora_scales": merged_updates.pop("lora_scales")}
                if "lora_scales_skip_cache_reset" in merged_updates:
                    lora_payload["lora_scales_skip_cache_reset"] = merged_updates.pop(
                        "lora_scales_skip_cache_reset"
                    )
                self.control_bus.enqueue(
                    EventType.SET_LORA_SCALES,
                    payload=lora_payload,
                )

            # Handle base_seed via events
            if "base_seed" in merged_updates:
                self.control_bus.enqueue(
                    EventType.SET_SEED,
                    payload={"base_seed": merged_updates.pop("base_seed")},
                )

            # Handle denoising_step_list via events
            if "denoising_step_list" in merged_updates:
                self.control_bus.enqueue(
                    EventType.SET_DENOISE_STEPS,
                    payload={
                        "denoising_step_list": merged_updates.pop("denoising_step_list")
                    },
                )

            # Update video mode if input_mode parameter changes
            if "input_mode" in merged_updates:
                self._video_mode = merged_updates.get("input_mode") == "video"

            # Remaining keys merge directly into self.parameters (no event needed)
            if merged_updates:
                self.parameters = {**self.parameters, **merged_updates}

        # ========================================================================
        # ORDER + APPLY: Apply events in deterministic order
        # ========================================================================
        events = self.control_bus.drain_pending(
            is_paused=self.paused, chunk_index=self.chunk_index
        )

        applied_prompt_payload: dict[str, Any] | None = None
        for event in events:
            if event.type == EventType.PAUSE:
                self.paused = True
            elif event.type == EventType.RESUME:
                self.paused = False
            elif event.type == EventType.SET_PROMPT:
                # Clear stale transition when new prompts arrive without transition
                if (
                    "prompts" in event.payload
                    and "transition" not in event.payload
                    and "transition" in self.parameters
                ):
                    self.parameters.pop("transition", None)
                # Apply prompt/transition to parameters
                if "prompts" in event.payload:
                    self.parameters["prompts"] = event.payload["prompts"]
                if "transition" in event.payload:
                    self.parameters["transition"] = event.payload["transition"]
                applied_prompt_payload = event.payload
            elif event.type == EventType.SET_LORA_SCALES:
                self.parameters["lora_scales"] = event.payload["lora_scales"]
                if event.payload.get("lora_scales_skip_cache_reset"):
                    self.parameters["lora_scales_skip_cache_reset"] = True
            elif event.type == EventType.SET_SEED:
                new_seed = event.payload["base_seed"]
                self.parameters["base_seed"] = new_seed
                # Track seed history (keep last 50)
                if not hasattr(self, "_seed_history"):
                    self._seed_history = []
                self._seed_history.append(new_seed)
                if len(self._seed_history) > 50:
                    self._seed_history = self._seed_history[-50:]
                logger.info(f"Seed set: {new_seed}")
            elif event.type == EventType.SET_DENOISE_STEPS:
                denoising_step_list = event.payload.get("denoising_step_list")
                if (
                    not isinstance(denoising_step_list, list)
                    or not denoising_step_list
                    or not all(isinstance(step, int) for step in denoising_step_list)
                ):
                    logger.warning(
                        "Ignoring invalid denoising_step_list=%r",
                        denoising_step_list,
                    )
                else:
                    self.parameters["denoising_step_list"] = denoising_step_list
                    logger.info("Set denoising_step_list=%s", denoising_step_list)

        # Check if paused after applying events (step overrides pause)
        if self.paused and not step_requested:
            # Sleep briefly to avoid busy waiting
            self.shutdown_event.wait(SLEEP_TIME)
            return

        # Recorder prompt-edge detection: capture prompt changes applied while paused/video-waiting.
        fallback_prompt: str | None = None
        fallback_weight: float = 1.0
        if self.session_recorder.is_recording:
            prev_prompt = self.session_recorder.last_prompt
            cur_prompt, cur_weight = self._get_current_effective_prompt()
            if cur_prompt is not None and cur_prompt != prev_prompt:
                fallback_prompt = cur_prompt
                fallback_weight = float(cur_weight)

        # Get the current pipeline using sync wrapper
        pipeline = self.pipeline_manager.get_pipeline()

        external_hold_last = (
            getattr(pipeline, "vace_enabled", False) and self._ndi_external_hold_last_enabled()
        )
        if external_hold_last:
            # External control staleness policy:
            # - Hold-last keeps generation running through short gaps/jitter.
            # - If the newest control frame is too old, stall until fresh input arrives.
            stale_ms = self._get_vace_external_stale_ms()
            stale_now = False
            age_ms: float | None = None
            if stale_ms > 0 and self.ndi_last_frame_ts_s > 0:
                age_ms = (time.monotonic() - float(self.ndi_last_frame_ts_s)) * 1000.0
                stale_now = age_ms > stale_ms

            if stale_now:
                if not self.external_input_stale:
                    logger.warning(
                        "External control stale (age_ms=%.1f > stale_ms=%.1f); stalling generation",
                        age_ms,
                        stale_ms,
                    )
                self.external_input_stale = True
                self._external_resume_hard_cut_pending = True
                self.shutdown_event.wait(SLEEP_TIME)
                return

            if self.external_input_stale:
                self.external_input_stale = False
                resume_hard_cut = self._get_vace_external_resume_hard_cut_enabled()
                if (
                    resume_hard_cut
                    and self._external_resume_hard_cut_pending
                    and "reset_cache" not in self.parameters
                ):
                    self.parameters["reset_cache"] = True
                    logger.info(
                        "External control resumed (age_ms=%.1f <= stale_ms=%.1f); forcing reset_cache=True",
                        age_ms if age_ms is not None else -1.0,
                        stale_ms,
                    )
                elif self._external_resume_hard_cut_pending:
                    logger.info(
                        "External control resumed (age_ms=%.1f <= stale_ms=%.1f); resume hard cut disabled",
                        age_ms if age_ms is not None else -1.0,
                        stale_ms,
                    )
                self._external_resume_hard_cut_pending = False
        else:
            self.external_input_stale = False
            self._external_resume_hard_cut_pending = False

        # prepare() will handle any required preparation based on parameters internally
        reset_cache = self.parameters.get("reset_cache", None)
        lora_scales = self.parameters.get("lora_scales", None)
        lora_scales_skip_cache_reset = self.parameters.get(
            "lora_scales_skip_cache_reset", False
        )
        hard_cut_executed = False

        # Clear output buffer queue when reset_cache is requested to prevent old frames.
        # Keep reset_cache pending until it is actually applied (we might early-return
        # while waiting for video input).
        if reset_cache:
            if not self._hard_cut_flushed_pending:
                logger.info(
                    "HARD CUT: reset_cache=True received, will pass init_cache=True to pipeline"
                )
                self.flush_output_queue()

                # Phase 2.1b: clear generation control buffer + reset VDA streaming state.
                # Do this once per hard cut to avoid repeatedly draining buffers while we wait for video.
                try:
                    self._control_map_worker.request_hard_cut(
                        clear_queue=True, reason="reset_cache"
                    )
                except Exception:
                    logger.warning("ControlMapWorker hard_cut failed", exc_info=True)

                # Also reset local fallback generators so chunk-time fallback is clean.
                if self._depth_generator is not None:
                    self._depth_generator.reset_cache()
                self._prev_control_frames = None
                self._prev_control_map_mode = None

                self._hard_cut_flushed_pending = True
        else:
            self._hard_cut_flushed_pending = False

        requirements = None
        if hasattr(pipeline, "prepare"):
            prepare_params = dict(self.parameters.items())
            prepare_params.pop("reset_cache", None)
            prepare_params.pop("lora_scales", None)
            prepare_params.pop("lora_scales_skip_cache_reset", None)
            if self._video_mode:
                # Signal to prepare() that video input is expected.
                # This allows resolve_input_mode() to detect video mode correctly.
                prepare_params["video"] = True  # Placeholder, actual data passed later
            requirements = pipeline.prepare(
                **prepare_params,
            )

        video_input = None
        frame_ids: list[int] | None = None
        if requirements is not None:
            current_chunk_size = requirements.input_size
            hold_last = self._ndi_external_hold_last_enabled()
            if hold_last:
                # In external/passthrough mode, NDI frames are control maps. Avoid coupling
                # generator cadence to NDI arrival cadence by holding the last frame.
                with self.frame_buffer_lock:
                    buffer_len = len(self.frame_buffer)
                has_any_frame = buffer_len > 0 or self._ndi_hold_last_input_frame is not None
                if not has_any_frame:
                    # Sleep briefly to avoid busy waiting
                    self.shutdown_event.wait(SLEEP_TIME)
                    return
                video_input, frame_ids = self._prepare_chunk_hold_last(current_chunk_size)
            else:
                with self.frame_buffer_lock:
                    has_enough_frames = bool(self.frame_buffer) and (
                        len(self.frame_buffer) >= current_chunk_size
                    )
                if not has_enough_frames:
                    # Sleep briefly to avoid busy waiting
                    self.shutdown_event.wait(SLEEP_TIME)
                    return
                video_input, frame_ids = self.prepare_chunk(current_chunk_size)
            if len(video_input) < current_chunk_size:
                # Buffer state changed underneath us; retry next loop.
                self.shutdown_event.wait(SLEEP_TIME)
                return
        chunk_error: Exception | None = None
        try:
            # Pass parameters (excluding prepare-only parameters)
            call_params = dict(self.parameters.items())
            call_params.pop("reset_cache", None)
            call_params.pop("lora_scales", None)
            call_params.pop("lora_scales_skip_cache_reset", None)

            # Pass reset_cache as init_cache to pipeline
            call_params["init_cache"] = not self.is_prepared
            if reset_cache is not None:
                call_params["init_cache"] = reset_cache
                hard_cut_executed = bool(reset_cache)

            # Pass lora_scales only when present (one-time update)
            if lora_scales is not None:
                call_params["lora_scales"] = lora_scales
                # When blend mode is enabled, tell pipeline to skip its own cache reset
                if lora_scales_skip_cache_reset:
                    call_params["lora_scales_skip_cache_reset"] = True

            # Pass soft_transition_active to prevent cache reset during soft transitions
            if self._soft_transition_active:
                call_params["soft_transition_active"] = True

            # Route video input based on VACE status.
            #
            # Default behavior is mutually exclusive:
            # - VACE enabled: treat the incoming stream as conditioning-only (`vace_input_frames`)
            # - VACE disabled: treat the incoming stream as latent-init V2V (`video`)
            #
            # Experimental hybrid mode (opt-in via `vace_hybrid_video_init=True`):
            # - Provide BOTH `video` (latent init) and `vace_input_frames` (conditioning)
            if video_input is not None:
                vace_enabled = getattr(pipeline, "vace_enabled", False)
                if vace_enabled:
                    vace_hybrid_video_init = bool(
                        self.parameters.get("vace_hybrid_video_init", False)
                    )
                    # VACE V2V editing mode: route to vace_input_frames
                    # Apply control map transform if enabled
                    control_map_mode = self.parameters.get(
                        "vace_control_map_mode", "none"
                    )
                    if control_map_mode == "canny":
                        # Phase 2.1b: try buffer sampling first
                        control_frames = self._try_sample_control_frames(frame_ids)
                        if control_frames is None:
                            # Fallback: chunk-time compute
                            low = self.parameters.get("vace_canny_low_threshold", None)
                            high = self.parameters.get("vace_canny_high_threshold", None)
                            blur_kernel = self.parameters.get("vace_canny_blur_kernel", 5)
                            blur_sigma = self.parameters.get("vace_canny_blur_sigma", 1.4)
                            adaptive = self.parameters.get("vace_canny_adaptive", True)
                            dilate = self.parameters.get("vace_canny_dilate", False)
                            dilate_size = self.parameters.get("vace_canny_dilate_size", 2)
                            control_frames = apply_canny_edges(
                                video_input,
                                low_threshold=low,
                                high_threshold=high,
                                blur_kernel_size=blur_kernel,
                                blur_sigma=blur_sigma,
                                adaptive_thresholds=adaptive,
                                dilate_edges=dilate,
                                dilate_kernel_size=dilate_size,
                            )
                        call_params["vace_input_frames"] = control_frames
                        # Store latest control frame for preview streaming
                        with self.latest_control_frame_lock:
                            last = control_frames[-1].squeeze(0)
                            if last.dtype != torch.uint8:
                                last = last.clamp(0, 255).to(torch.uint8)
                            self.latest_control_frame_cpu = last.to(device="cpu")
                    elif control_map_mode == "pidinet":
                        # Phase 2.1b: try buffer sampling first
                        control_frames = self._try_sample_control_frames(frame_ids)
                        if control_frames is None:
                            # Fallback: chunk-time compute
                            if self._pidinet_generator is None:
                                self._pidinet_generator = PiDiNetEdgeGenerator()
                            safe_mode = self.parameters.get("vace_pidinet_safe", True)
                            apply_filter = self.parameters.get(
                                "vace_pidinet_filter", True
                            )
                            self._pidinet_generator.safe_mode = safe_mode
                            control_frames = self._pidinet_generator.process_frames(
                                video_input, apply_filter=apply_filter
                            )
                        call_params["vace_input_frames"] = control_frames
                        # Store latest control frame for preview streaming
                        with self.latest_control_frame_lock:
                            last = control_frames[-1].squeeze(0)
                            if last.dtype != torch.uint8:
                                last = last.clamp(0, 255).to(torch.uint8)
                            self.latest_control_frame_cpu = last.to(device="cpu")
                    elif control_map_mode == "depth":
                        hard_cut = reset_cache is not None and reset_cache
                        # Phase 2.1b: try buffer sampling first
                        control_frames = self._try_sample_control_frames(frame_ids)
                        if control_frames is None:
                            # Fallback: chunk-time compute
                            if self._depth_generator is None:
                                self._depth_generator = VDADepthControlMapGenerator()
                            depth_input_size = self.parameters.get("vace_depth_input_size")
                            depth_fp32 = self.parameters.get("vace_depth_fp32")
                            depth_temporal_mode = self.parameters.get(
                                "vace_depth_temporal_mode"
                            )
                            depth_contrast = self.parameters.get("vace_depth_contrast")
                            if depth_contrast is not None:
                                self._depth_generator.depth_contrast = depth_contrast

                            depth_output_device = os.getenv(
                                "SCOPE_VACE_DEPTH_CHUNK_OUTPUT_DEVICE", "cpu"
                            ).strip()
                            if depth_output_device.lower() not in ("", "cpu", "cuda") and not depth_output_device.lower().startswith(
                                "cuda:"
                            ):
                                logger.warning(
                                    "Invalid SCOPE_VACE_DEPTH_CHUNK_OUTPUT_DEVICE=%r; expected 'cpu' or 'cuda'",
                                    depth_output_device,
                                )
                                depth_output_device = "cpu"

                            control_frames = self._depth_generator.process_frames(
                                video_input,
                                hard_cut=hard_cut,
                                input_size=depth_input_size,
                                fp32=depth_fp32,
                                temporal_mode=depth_temporal_mode,
                                output_device=depth_output_device,
                            )
                        # Apply temporal EMA if enabled (after either path)
                        temporal_ema = self.parameters.get(
                            "vace_control_map_temporal_ema", 0.0
                        )
                        if temporal_ema > 0:
                            control_frames = self._apply_temporal_ema(
                                control_frames, "depth", temporal_ema, hard_cut
                            )
                        call_params["vace_input_frames"] = control_frames
                        # Store latest control frame for preview streaming
                        with self.latest_control_frame_lock:
                            last = control_frames[-1].squeeze(0)
                            if last.dtype != torch.uint8:
                                last = last.clamp(0, 255).to(torch.uint8)
                            self.latest_control_frame_cpu = last.to(device="cpu")
                    elif control_map_mode == "composite":
                        hard_cut = reset_cache is not None and reset_cache
                        # Phase 2.1b: try buffer sampling first
                        control_frames = self._try_sample_control_frames(frame_ids)
                        if control_frames is None:
                            # Fallback: chunk-time compute (depth + edges fused)
                            if self._depth_generator is None:
                                self._depth_generator = VDADepthControlMapGenerator()

                            edge_strength = self.parameters.get(
                                "composite_edge_strength", 0.6
                            )
                            edge_thickness = self.parameters.get(
                                "composite_edge_thickness", 8
                            )
                            sharpness = self.parameters.get("composite_sharpness", 10.0)
                            edge_source = self.parameters.get(
                                "composite_edge_source", "canny"
                            )

                            depth_input_size = self.parameters.get("vace_depth_input_size")
                            depth_fp32 = self.parameters.get("vace_depth_fp32")
                            depth_temporal_mode = self.parameters.get(
                                "vace_depth_temporal_mode"
                            )
                            depth_contrast = self.parameters.get("vace_depth_contrast")
                            if depth_contrast is not None:
                                self._depth_generator.depth_contrast = depth_contrast

                            depth_output_device = os.getenv(
                                "SCOPE_VACE_DEPTH_CHUNK_OUTPUT_DEVICE", "cpu"
                            ).strip()
                            if depth_output_device.lower() not in ("", "cpu", "cuda") and not depth_output_device.lower().startswith(
                                "cuda:"
                            ):
                                logger.warning(
                                    "Invalid SCOPE_VACE_DEPTH_CHUNK_OUTPUT_DEVICE=%r; expected 'cpu' or 'cuda'",
                                    depth_output_device,
                                )
                                depth_output_device = "cpu"

                            depth_frames = self._depth_generator.process_frames(
                                video_input,
                                hard_cut=hard_cut,
                                input_size=depth_input_size,
                                fp32=depth_fp32,
                                temporal_mode=depth_temporal_mode,
                                output_device=depth_output_device,
                            )

                            if edge_source == "pidinet":
                                if self._pidinet_generator is None:
                                    self._pidinet_generator = PiDiNetEdgeGenerator()
                                safe_mode = self.parameters.get(
                                    "vace_pidinet_safe", True
                                )
                                apply_filter = self.parameters.get(
                                    "vace_pidinet_filter", True
                                )
                                self._pidinet_generator.safe_mode = safe_mode
                                edge_frames = self._pidinet_generator.process_frames(
                                    video_input, apply_filter=apply_filter
                                )
                            else:
                                edge_frames = apply_canny_edges(
                                    video_input,
                                    adaptive_thresholds=True,
                                    dilate_edges=True,
                                    dilate_kernel_size=edge_thickness,
                                )

                            control_frames = []
                            for depth_f, edge_f in zip(depth_frames, edge_frames, strict=True):
                                if depth_f.device != edge_f.device:
                                    edge_f = edge_f.to(device=depth_f.device)
                                depth_norm = depth_f[:, :, :, 0] / 255.0
                                edge_norm = edge_f[:, :, :, 0] / 255.0

                                fused = soft_max_fusion(
                                    depth_norm,
                                    edge_norm,
                                    edge_strength=edge_strength,
                                    sharpness=sharpness,
                                )

                                fused_uint8 = (fused * 255.0).clamp(0, 255)
                                fused_rgb = fused_uint8.unsqueeze(-1).expand(-1, -1, -1, 3)
                                control_frames.append(fused_rgb)

                        # Apply temporal EMA if enabled (after either path)
                        temporal_ema = self.parameters.get(
                            "vace_control_map_temporal_ema", 0.0
                        )
                        if temporal_ema > 0:
                            control_frames = self._apply_temporal_ema(
                                control_frames, "composite", temporal_ema, hard_cut
                            )
                        call_params["vace_input_frames"] = control_frames
                        # Store latest control frame for preview streaming
                        with self.latest_control_frame_lock:
                            last = control_frames[-1].squeeze(0)
                            if last.dtype != torch.uint8:
                                last = last.clamp(0, 255).to(torch.uint8)
                            self.latest_control_frame_cpu = last.to(device="cpu")
                    elif control_map_mode == "external":
                        # External/passthrough mode: use video input directly as control signal
                        # No processing - frames are assumed to already be control maps (e.g., from OBS, TouchDesigner)
                        call_params["vace_input_frames"] = video_input
                        # Store latest frame for preview streaming (so user can see what VACE receives)
                        with self.latest_control_frame_lock:
                            last = video_input[-1].squeeze(0)
                            if last.dtype != torch.uint8:
                                last = last.clamp(0, 255).to(torch.uint8)
                            self.latest_control_frame_cpu = last.to(device="cpu")
                    else:
                        call_params["vace_input_frames"] = video_input
                    if vace_hybrid_video_init:
                        # Hybrid: also pass the raw video frames as latent-init base.
                        # Note: VACE encoding must avoid clobbering the VAE streaming cache when
                        # both `video` and `vace_input_frames` are present (handled in VACE blocks).
                        call_params["video"] = video_input
                else:
                    # Normal V2V mode: route to video
                    call_params["video"] = video_input

            transition_active_after_call = False
            with self.pipeline_manager.locked_pipeline() as locked_pipeline:
                output = locked_pipeline(**call_params)
                if hasattr(locked_pipeline, "state"):
                    transition_active_after_call = locked_pipeline.state.get(
                        "_transition_active", False
                    )

            # Consume one-shot updates only after they were passed to the pipeline.
            if lora_scales is not None:
                self.parameters.pop("lora_scales", None)
                self.parameters.pop("lora_scales_skip_cache_reset", None)
            if reset_cache is not None:
                self.parameters.pop("reset_cache", None)
                self._hard_cut_flushed_pending = False
                # Also reset control map worker cache (Phase 2.1a)
                self._control_map_worker.reset_cache()

            # Clear vace_ref_images from parameters after use to prevent sending them on subsequent chunks
            # vace_ref_images should only be sent when explicitly provided in parameter updates
            if (
                "vace_ref_images" in call_params
                and "vace_ref_images" in self.parameters
            ):
                self.parameters.pop("vace_ref_images", None)

            # Clear transition when complete (blocks signal completion via _transition_active)
            # Contract: Modular pipelines manage prompts internally; frame_processor manages lifecycle
            if "transition" in call_params and "transition" in self.parameters:
                transition_active = transition_active_after_call

                transition = call_params.get("transition")
                if not transition_active or transition is None:
                    target_prompts = None
                    if isinstance(transition, dict):
                        target_prompts = transition.get("target_prompts")
                    elif transition is not None and hasattr(
                        transition, "target_prompts"
                    ):
                        target_prompts = getattr(transition, "target_prompts", None)

                    if target_prompts is not None:
                        self.parameters["prompts"] = target_prompts
                    self.parameters.pop("transition", None)

            processing_time = time.time() - start_time
            num_frames = output.shape[0]
            logger.debug(
                f"Processed pipeline in {processing_time:.4f}s, {num_frames} frames"
            )

            # Normalize to [0, 255] and convert to uint8
            output = (
                (output * 255.0)
                .clamp(0, 255)
                .to(dtype=torch.uint8)
                .contiguous()
                .detach()
                .cpu()
            )

            # Store latest frame for non-destructive REST reads
            with self.latest_frame_lock:
                self.latest_frame_cpu = output[-1].clone()
                self.latest_frame_id += 1
            self._signal_latest_frame_available()

            # Resize output queue to meet target max size.
            #
            # Lock protects against race with flush_output_queue(). In low-latency output mode,
            # we also keep the queue size bounded (cap) and prefer dropping the oldest frames
            # over dropping newly-generated frames.
            with self.output_queue_lock:
                factor = OUTPUT_QUEUE_MAX_SIZE_FACTOR
                factor_env = os.getenv("SCOPE_OUTPUT_QUEUE_MAX_SIZE_FACTOR", "").strip()
                if factor_env:
                    try:
                        factor = max(1, int(factor_env))
                    except ValueError:
                        logger.warning(
                            "Invalid SCOPE_OUTPUT_QUEUE_MAX_SIZE_FACTOR=%r; expected int",
                            factor_env,
                        )
                elif self._low_latency_output_mode:
                    # Prefer a small default queue (≈ one chunk) when low-latency output
                    # is enabled. Users can increase via SCOPE_OUTPUT_QUEUE_MAX_SIZE_FACTOR.
                    factor = 1

                target_output_queue_max_size = num_frames * factor
                if self._output_queue_maxsize_cap is not None:
                    target_output_queue_max_size = min(
                        int(self._output_queue_maxsize_cap),
                        int(target_output_queue_max_size),
                    )

                if (
                    target_output_queue_max_size > 0
                    and self.output_queue.maxsize < target_output_queue_max_size
                ):
                    logger.info(
                        "Increasing output queue size to %s, current size %s, num_frames %s",
                        target_output_queue_max_size,
                        self.output_queue.maxsize,
                        num_frames,
                    )

                    # Transfer frames from old queue to new queue
                    old_queue = self.output_queue
                    self.output_queue = queue.Queue(maxsize=target_output_queue_max_size)
                    while not old_queue.empty():
                        try:
                            frame = old_queue.get_nowait()
                            self.output_queue.put_nowait(frame)
                        except queue.Empty:
                            break

                for frame in output:
                    if self._low_latency_output_mode:
                        # Drop oldest frames to make room for newest output.
                        while True:
                            try:
                                self.output_queue.put_nowait(frame)
                                break
                            except queue.Full:
                                try:
                                    self.output_queue.get_nowait()
                                    self.output_frames_dropped += 1
                                except queue.Empty:
                                    # Shouldn't happen, but avoid spinning.
                                    self.output_frames_dropped += 1
                                    break
                    else:
                        try:
                            self.output_queue.put_nowait(frame)
                        except queue.Full:
                            logger.warning("Output queue full, dropping processed frame")
                            self.output_frames_dropped += 1
                            continue

            # Update FPS calculation based on processing time and frame count
            self._calculate_pipeline_fps(start_time, num_frames)
        except Exception as e:
            chunk_error = e
            if self._is_recoverable(e):
                # Handle recoverable errors with full stack trace and continue processing
                logger.error(f"Error processing chunk: {e}", exc_info=True)
            else:
                raise e

        # SessionRecorder: record prompt/transition + hard/soft cuts for this chunk.
        # Record only after a successful pipeline call so paused/video-wait churn doesn't
        # create phantom segments.
        if self.session_recorder.is_recording and chunk_error is None:
            wall_time = time.monotonic()

            # Soft cut metadata is recorded ONCE per trigger, at the first generated chunk.
            soft_cut_bias = None
            soft_cut_chunks = None
            soft_restore_bias = None
            soft_restore_was_set = False
            if (
                self._soft_transition_record_pending
                and self._soft_transition_temp_bias is not None
            ):
                soft_cut_bias = float(self._soft_transition_temp_bias)
                soft_cut_chunks = int(self._soft_transition_chunks_remaining)
                soft_restore_was_set = bool(self._soft_transition_original_bias_was_set)
                soft_restore_bias = (
                    float(self._soft_transition_original_bias)
                    if self._soft_transition_original_bias is not None
                    and self._soft_transition_original_bias_was_set
                    else None
                )
                self._soft_transition_record_pending = False

            recorded_prompt_event = False
            if applied_prompt_payload is not None:
                prompt_text = None
                prompt_weight = 1.0

                tr = applied_prompt_payload.get("transition")
                if isinstance(tr, dict):
                    targets = tr.get("target_prompts")
                    if isinstance(targets, list) and targets:
                        first = targets[0]
                        if isinstance(first, dict):
                            prompt_text = first.get("text")
                            prompt_weight = float(first.get("weight", 1.0))

                if prompt_text is None:
                    prompts = applied_prompt_payload.get("prompts")
                    if isinstance(prompts, list) and prompts:
                        first = prompts[0]
                        if isinstance(first, dict):
                            prompt_text = first.get("text")
                            prompt_weight = float(first.get("weight", 1.0))

                transition_steps = None
                transition_method = None
                if isinstance(tr, dict):
                    transition_steps = tr.get("num_steps")
                    transition_method = tr.get("temporal_interpolation_method")

                if prompt_text is not None:
                    self.session_recorder.record_event(
                        chunk_index=self.chunk_index,
                        wall_time=wall_time,
                        prompt=prompt_text,
                        prompt_weight=prompt_weight,
                        transition_steps=transition_steps,
                        transition_method=transition_method,
                        hard_cut=hard_cut_executed,
                        soft_cut_bias=soft_cut_bias,
                        soft_cut_chunks=soft_cut_chunks,
                        soft_cut_restore_bias=soft_restore_bias,
                        soft_cut_restore_was_set=soft_restore_was_set,
                    )
                    recorded_prompt_event = True
                    hard_cut_executed = False
                    soft_cut_bias = None

            # Fallback: prompt changed since last recorded chunk (e.g. edits while paused)
            if not recorded_prompt_event and fallback_prompt is not None:
                self.session_recorder.record_event(
                    chunk_index=self.chunk_index,
                    wall_time=wall_time,
                    prompt=fallback_prompt,
                    prompt_weight=fallback_weight,
                    hard_cut=hard_cut_executed,
                    soft_cut_bias=soft_cut_bias,
                    soft_cut_chunks=soft_cut_chunks,
                    soft_cut_restore_bias=soft_restore_bias,
                    soft_cut_restore_was_set=soft_restore_was_set,
                )
                recorded_prompt_event = True
                hard_cut_executed = False
                soft_cut_bias = None

            # Cut-only event (no prompt change): recorder carries forward last prompt
            if (not recorded_prompt_event) and (
                hard_cut_executed or soft_cut_bias is not None
            ):
                self.session_recorder.record_event(
                    chunk_index=self.chunk_index,
                    wall_time=wall_time,
                    prompt=None,
                    hard_cut=hard_cut_executed,
                    soft_cut_bias=soft_cut_bias,
                    soft_cut_chunks=soft_cut_chunks,
                    soft_cut_restore_bias=soft_restore_bias,
                    soft_cut_restore_was_set=soft_restore_was_set,
                )

        self.is_prepared = True

        # Soft transition countdown and auto-restore at chunk boundary
        if self._soft_transition_active:
            self._soft_transition_chunks_remaining -= 1
            if self._soft_transition_chunks_remaining <= 0:
                if self._soft_transition_original_bias_was_set:
                    # Restore original bias value
                    if self._soft_transition_original_bias is not None:
                        self.parameters["kv_cache_attention_bias"] = (
                            self._soft_transition_original_bias
                        )
                        logger.info(
                            f"Soft transition complete: restored bias to "
                            f"{self._soft_transition_original_bias}"
                        )
                    else:
                        self.parameters.pop("kv_cache_attention_bias", None)
                        logger.info(
                            "Soft transition complete: restored kv_cache_attention_bias to <unset>"
                        )
                else:
                    # Restore to "unset" (pipeline/config default) if we didn't get overridden.
                    current_bias = self.parameters.get("kv_cache_attention_bias")
                    if (
                        self._soft_transition_temp_bias is None
                        or current_bias == self._soft_transition_temp_bias
                    ):
                        self.parameters.pop("kv_cache_attention_bias", None)
                        logger.info(
                            "Soft transition complete: restored kv_cache_attention_bias to <unset>"
                        )
                    else:
                        logger.info(
                            "Soft transition complete: keeping kv_cache_attention_bias override"
                        )

                self._soft_transition_active = False
                self._soft_transition_chunks_remaining = 0
                self._soft_transition_temp_bias = None
                self._soft_transition_original_bias = None
                self._soft_transition_original_bias_was_set = False
                self._soft_transition_record_pending = False

        self.chunk_index += 1

        # Send step response after completing a step-driven chunk generation.
        if self._pending_steps > 0:
            self._pending_steps = max(0, self._pending_steps - 1)

        if step_requested and self.snapshot_response_callback:
            self.snapshot_response_callback(
                {
                    "type": "step_response",
                    "chunk_index": self.chunk_index,
                    "success": chunk_error is None,
                    "error": str(chunk_error) if chunk_error is not None else None,
                }
            )

    def prepare_chunk(self, chunk_size: int) -> tuple[list[torch.Tensor], list[int]]:
        """
        Sample frames uniformly from the buffer, convert them to tensors, and remove processed frames.

        This function implements uniform sampling across the entire buffer to ensure
        temporal coverage of input frames. It samples frames at evenly distributed
        indices and removes all frames up to the last sampled frame to prevent
        buffer buildup.

        When low-latency mode is enabled (SCOPE_LOW_LATENCY_INPUT=1), the buffer is
        first trimmed to only keep the newest frames, reducing input lag at the cost
        of dropping older frames.

        Implementation note:
            We hold `frame_buffer_lock` only while sampling + popping frames, then
            convert frames to tensors outside the lock to avoid blocking concurrent
            input producers (WebRTC/Spout/NDI) on expensive `to_ndarray()` work.

        Example (normal mode):
            With buffer_len=8 and chunk_size=4:
            - step = 8/4 = 2.0
            - indices = [0, 2, 4, 6] (uniformly distributed)
            - Returns frames at positions 0, 2, 4, 6
            - Removes frames 0-6 from buffer (7 frames total)

        Example (low-latency mode, buffer_factor=2):
            With buffer_len=30 and chunk_size=4:
            - max_keep = 4 * 2 = 8
            - Drop 22 oldest frames first
            - Then sample uniformly from remaining 8 frames

        Returns:
            Tuple of (tensor_frames, frame_ids):
              - tensor_frames: List of (1, H, W, C) tensors for downstream preprocess_chunk
              - frame_ids: List of frame IDs for Phase 2.1b control buffer sampling
        """
        with self.frame_buffer_lock:
            if not self.frame_buffer or len(self.frame_buffer) < chunk_size:
                return [], []

            # Low-latency mode: trim buffer to reduce lag
            if self._low_latency_mode:
                max_keep = chunk_size * self._low_latency_buffer_factor
                if len(self.frame_buffer) > max_keep:
                    drop_count = len(self.frame_buffer) - max_keep
                    for _ in range(drop_count):
                        self.frame_buffer.popleft()
                    self.input_frames_dropped += drop_count

            # Calculate uniform sampling step
            step = len(self.frame_buffer) / chunk_size
            # Generate indices for uniform sampling
            indices = [round(i * step) for i in range(chunk_size)]
            # Extract VideoFrames at sampled indices
            video_frames = [self.frame_buffer[i] for i in indices]

            # Phase 2.1b: carry frame IDs through for control buffer lookup
            frame_ids = [int(getattr(f, "frame_id", -1)) for f in video_frames]

            # Drop all frames up to and including the last sampled frame
            last_idx = indices[-1]
            for _ in range(last_idx + 1):
                self.frame_buffer.popleft()

        # Convert VideoFrames to tensors
        mirror_input = self.parameters.get("mirror_input", True)  # Default: mirrored (selfie mode)
        tensor_frames = []
        for video_frame in video_frames:
            # Convert VideoFrame into (1, H, W, C) tensor on cpu
            # The T=1 dimension is expected by preprocess_chunk which rearranges T H W C -> T C H W
            tensor = (
                torch.from_numpy(video_frame.to_ndarray(format="rgb24"))
                .unsqueeze(0)
            )
            # Apply horizontal flip when mirror mode is DISABLED
            # mirror_input=True → selfie mode (natural, no flip - browser already sends mirrored)
            # mirror_input=False → raw camera mode (flip to show true orientation)
            if not mirror_input:
                tensor = torch.flip(tensor, dims=[2])  # Flip width dimension (1, H, W, C)
            tensor_frames.append(tensor)

        return tensor_frames, frame_ids

    def _ndi_external_hold_last_enabled(self) -> bool:
        if self.get_active_input_source() != "ndi":
            return False
        return (self.parameters.get("vace_control_map_mode") or "none") == "external"

    def _get_vace_external_stale_ms(self) -> float:
        raw = self.parameters.get("vace_external_stale_ms")
        if raw is None:
            raw = os.getenv("SCOPE_VACE_EXTERNAL_STALE_MS", "500")
        try:
            ms = float(raw)
        except (TypeError, ValueError):
            ms = 500.0
        return max(0.0, ms)

    def _get_vace_external_resume_hard_cut_enabled(self) -> bool:
        raw = self.parameters.get("vace_external_resume_hard_cut")
        if raw is None:
            raw = os.getenv("SCOPE_VACE_EXTERNAL_RESUME_HARD_CUT", "1")
        if isinstance(raw, str):
            return raw.strip().lower() in ("1", "true", "yes", "on")
        return bool(raw)

    def _get_output_pacing_fps(self) -> float | None:
        raw = self.parameters.get("output_pacing_fps")
        if raw is None:
            raw = os.getenv("SCOPE_OUTPUT_PACING_FPS", "").strip()

        if raw in (None, ""):
            return None

        try:
            fps = float(raw)
        except (TypeError, ValueError):
            return None

        if fps <= 0:
            return None

        return fps

    def _prepare_chunk_hold_last(self, chunk_size: int) -> tuple[list[torch.Tensor], list[int]]:
        """Prepare a chunk, repeating the latest input frame if we're underflowing.

        This is intended for NDI + `vace_control_map_mode="external"` where input
        frames are control maps and should not stall the generator.
        """
        # Fast path: preserve existing uniform-sampling semantics when enough frames exist.
        with self.frame_buffer_lock:
            buffer_len = len(self.frame_buffer)
        if buffer_len >= chunk_size:
            tensor_frames, frame_ids = self.prepare_chunk(chunk_size)
            if tensor_frames:
                self._ndi_hold_last_input_frame = tensor_frames[-1].detach().clone()
                self._ndi_hold_last_input_frame_id = int(frame_ids[-1]) if frame_ids else None
            return tensor_frames, frame_ids

        # Underflow: take whatever we have right now (possibly 0), then fill by repeating.
        with self.frame_buffer_lock:
            video_frames = list(self.frame_buffer)
            self.frame_buffer.clear()

        mirror_input = self.parameters.get("mirror_input", True)  # Default: mirrored (selfie mode)
        tensor_frames: list[torch.Tensor] = []
        frame_ids: list[int] = []
        for video_frame in video_frames:
            tensor = torch.from_numpy(video_frame.to_ndarray(format="rgb24")).unsqueeze(0)
            if not mirror_input:
                tensor = torch.flip(tensor, dims=[2])  # Flip width dimension (1, H, W, C)
            tensor_frames.append(tensor)
            frame_ids.append(int(getattr(video_frame, "frame_id", -1)))

        base_tensor: torch.Tensor | None = None
        base_frame_id: int = -1
        if tensor_frames:
            base_tensor = tensor_frames[-1]
            base_frame_id = frame_ids[-1]
            self._ndi_hold_last_input_frame = base_tensor.detach().clone()
            self._ndi_hold_last_input_frame_id = int(base_frame_id)
        elif self._ndi_hold_last_input_frame is not None:
            base_tensor = self._ndi_hold_last_input_frame
            if self._ndi_hold_last_input_frame_id is not None:
                base_frame_id = int(self._ndi_hold_last_input_frame_id)
        else:
            return [], []

        reused = 0
        while len(tensor_frames) < chunk_size:
            tensor_frames.append(base_tensor.clone())
            frame_ids.append(base_frame_id)
            reused += 1

        if reused:
            self.ndi_frames_reused += reused

        return tensor_frames, frame_ids

    def _create_snapshot(self) -> Snapshot:
        """Create a snapshot of current generation state.

        Captures:
        - Continuity state from pipeline.state (cloned tensors)
        - Control state (deep copy of parameters)
        - Metadata (chunk_index, timestamp, resolution)

        Returns:
            Snapshot object with unique ID
        """
        snapshot_id = str(uuid.uuid4())
        with self.pipeline_manager.locked_pipeline() as pipeline:
            # Capture continuity state from pipeline.state
            current_start_frame = 0
            first_context_frame = None
            context_frame_buffer = None
            decoded_frame_buffer = None
            context_frame_buffer_max_size = 0
            decoded_frame_buffer_max_size = 0

            if hasattr(pipeline, "state"):
                state = pipeline.state
                current_start_frame = state.get("current_start_frame", 0)
                context_frame_buffer_max_size = state.get(
                    "context_frame_buffer_max_size", 0
                )
                decoded_frame_buffer_max_size = state.get(
                    "decoded_frame_buffer_max_size", 0
                )

                # Clone tensors to avoid mutation
                fcf = state.get("first_context_frame")
                if fcf is not None and isinstance(fcf, torch.Tensor):
                    first_context_frame = fcf.detach().clone()

                cfb = state.get("context_frame_buffer")
                if cfb is not None and isinstance(cfb, torch.Tensor):
                    context_frame_buffer = cfb.detach().clone()

                dfb = state.get("decoded_frame_buffer")
                if dfb is not None and isinstance(dfb, torch.Tensor):
                    decoded_frame_buffer = dfb.detach().clone()

        # Get resolution from pipeline manager
        resolution = self._get_pipeline_dimensions()

        # Get pipeline_id if available
        pipeline_id = None
        try:
            status_info = self.pipeline_manager.get_status_info()
            pipeline_id = status_info.get("pipeline_id")
        except Exception:
            pass

        snapshot = Snapshot(
            snapshot_id=snapshot_id,
            chunk_index=self.chunk_index,
            created_at=time.time(),
            current_start_frame=current_start_frame,
            first_context_frame=first_context_frame,
            context_frame_buffer=context_frame_buffer,
            decoded_frame_buffer=decoded_frame_buffer,
            context_frame_buffer_max_size=context_frame_buffer_max_size,
            decoded_frame_buffer_max_size=decoded_frame_buffer_max_size,
            parameters=copy.deepcopy(self.parameters),
            paused=self.paused,
            video_mode=self._video_mode,
            # Style layer state
            world_state_json=self.world_state.model_dump_json(),
            active_style_name=self.style_manifest.name if self.style_manifest else None,
            compiled_prompt_text=(
                self._compiled_prompt.prompt if self._compiled_prompt else None
            ),
            pipeline_id=pipeline_id,
            resolution=resolution,
        )

        # Store snapshot with LRU eviction
        self.snapshots[snapshot_id] = snapshot
        self.snapshot_order.append(snapshot_id)

        # Evict oldest if over limit
        while len(self.snapshots) > MAX_SNAPSHOTS:
            oldest_id = self.snapshot_order.pop(0)
            old_snapshot = self.snapshots.pop(oldest_id, None)
            if old_snapshot:
                # Release tensor memory explicitly
                old_snapshot.first_context_frame = None
                old_snapshot.context_frame_buffer = None
                old_snapshot.decoded_frame_buffer = None
                logger.debug(f"Evicted snapshot {oldest_id} (LRU)")

        logger.info(
            f"Created snapshot {snapshot_id} at chunk {self.chunk_index}, "
            f"total snapshots: {len(self.snapshots)}"
        )

        return snapshot

    def _restore_snapshot(self, snapshot_id: str) -> bool:
        """Restore generation state from a snapshot.

        Restores:
        - Continuity state to pipeline.state
        - Control state to self.parameters
        - Clears output_queue to prevent stale frames
        - Sets is_prepared=True to avoid accidental cache reset

        Args:
            snapshot_id: ID of snapshot to restore

        Returns:
            True if restore succeeded, False if snapshot not found
        """
        snapshot = self.snapshots.get(snapshot_id)
        if snapshot is None:
            logger.warning(f"Snapshot {snapshot_id} not found")
            return False

        # LRU: move restored snapshot to end of order (most recently used)
        if snapshot_id in self.snapshot_order:
            self.snapshot_order.remove(snapshot_id)
            self.snapshot_order.append(snapshot_id)

        with self.pipeline_manager.locked_pipeline() as pipeline:
            # Restore continuity state to pipeline.state
            if hasattr(pipeline, "state"):
                state = pipeline.state
                state.set("current_start_frame", snapshot.current_start_frame)
                state.set(
                    "context_frame_buffer_max_size",
                    snapshot.context_frame_buffer_max_size,
                )
                state.set(
                    "decoded_frame_buffer_max_size",
                    snapshot.decoded_frame_buffer_max_size,
                )

                # Restore tensors back to pipeline.state (or clear when None).
                state.set(
                    "first_context_frame",
                    snapshot.first_context_frame.detach().clone()
                    if snapshot.first_context_frame is not None
                    else None,
                )
                state.set(
                    "context_frame_buffer",
                    snapshot.context_frame_buffer.detach().clone()
                    if snapshot.context_frame_buffer is not None
                    else None,
                )
                state.set(
                    "decoded_frame_buffer",
                    snapshot.decoded_frame_buffer.detach().clone()
                    if snapshot.decoded_frame_buffer is not None
                    else None,
                )

        # Restore control state
        self.parameters = copy.deepcopy(snapshot.parameters)
        self.paused = snapshot.paused
        self._video_mode = snapshot.video_mode
        self.chunk_index = snapshot.chunk_index

        # Restore style layer state (thread-safe via model_validate_json)
        if snapshot.world_state_json:
            self.world_state = WorldState.model_validate_json(snapshot.world_state_json)
        if snapshot.active_style_name:
            self.style_manifest = self.style_registry.get(snapshot.active_style_name)
            self._active_style_name = snapshot.active_style_name
        # Recompile after restore
        if self.world_state and self.style_manifest:
            self._compiled_prompt = self.prompt_compiler.compile(
                self.world_state, self.style_manifest
            )

        # Clear output_queue to prevent stale pre-restore frames
        self.flush_output_queue()

        # Clear frame_buffer in V2V mode to prevent stale input frames
        if self._video_mode:
            with self.frame_buffer_lock:
                self.frame_buffer.clear()

        # Set is_prepared=True to avoid accidental cache reset on next chunk
        self.is_prepared = True

        logger.info(
            f"Restored snapshot {snapshot_id} to chunk {snapshot.chunk_index}"
        )

        return True

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    @staticmethod
    def _is_recoverable(error: Exception) -> bool:
        """
        Check if an error is recoverable (i.e., processing can continue).
        Non-recoverable errors will cause the stream to stop.
        """
        if isinstance(error, torch.cuda.OutOfMemoryError):
            return False
        # Add more non-recoverable error types here as needed
        return True
