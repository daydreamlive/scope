import logging

import torch
from einops import rearrange

logger = logging.getLogger(__name__)


def normalize_frame_sizes(frames: list[torch.Tensor]) -> list[torch.Tensor]:
    """Normalize all frames to match the first frame's dimensions.

    Frames may have different sizes (e.g., from switching video sources or
    resolution changes). This function resizes all frames to match the first
    frame's height and width to ensure they can be stacked.

    Args:
        frames: List of tensors in THWC format

    Returns:
        List of tensors all with the same H and W dimensions
    """
    if not frames:
        return frames

    # Use first frame's shape as target
    target_h, target_w = frames[0].shape[1], frames[0].shape[2]
    normalized = []

    for i, frame in enumerate(frames):
        h, w = frame.shape[1], frame.shape[2]
        if h == target_h and w == target_w:
            normalized.append(frame)
            continue

        # Resize frame: THWC -> TCHW for interpolate, then back to THWC
        frame_tchw = frame.permute(0, 3, 1, 2).float()
        frame_resized = torch.nn.functional.interpolate(
            frame_tchw,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )
        frame_thwc = frame_resized.permute(0, 2, 3, 1).to(frame.dtype)
        logger.debug(f"Resized frame {i} from {w}x{h} to {target_w}x{target_h}")
        normalized.append(frame_thwc)

    return normalized


def preprocess_chunk(
    chunk: list[torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
    height: int | None = None,
    width: int | None = None,
) -> torch.Tensor:
    frames = []

    for frame in chunk:
        # Move to pipeline device first (likely as uint8), then convert dtype on device
        frame = frame.to(device=device).to(dtype=dtype)
        frame = rearrange(frame, "T H W C -> T C H W")

        _, _, H, W = frame.shape

        # If no height and width requested no resizing needed
        if height is None or width is None:
            frames.append(frame)
            continue

        # If we have a height and width match no resizing needed
        if H == height and W == width:
            frames.append(frame)
            continue

        frame_resized = torch.nn.functional.interpolate(
            frame,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )

        logger.debug(f"Resized frame from {W}x{H} to {width}x{height}")

        frames.append(frame_resized)

    # stack and rearrange to get a BCTHW tensor
    chunk = rearrange(torch.stack(frames, dim=1), "B T C H W -> B C T H W")
    # Normalize to [-1, 1] range
    return chunk / 255.0 * 2.0 - 1.0


def postprocess_chunk(chunk: torch.Tensor) -> torch.Tensor:
    # chunk is a BTCHW tensor
    # Drop the batch dim
    chunk = rearrange(chunk.squeeze(0), "T C H W -> T H W C")
    # Normalize to [0, 1]
    return (chunk / 2 + 0.5).clamp(0, 1).float()
