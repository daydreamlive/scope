import logging

import torch
from einops import rearrange

logger = logging.getLogger(__name__)


def _resize_tchw(frame: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """Resize a TCHW tensor using bilinear interpolation."""
    return torch.nn.functional.interpolate(
        frame,
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=False,
    )


def _needs_resize(h: int, w: int, target_h: int, target_w: int) -> bool:
    """Check if dimensions differ from target."""
    return h != target_h or w != target_w


def _resize_thwc(
    frame: torch.Tensor, target_h: int, target_w: int, output_dtype: torch.dtype
) -> torch.Tensor:
    """Resize a THWC tensor, returning result in the same format."""
    frame_tchw = frame.permute(0, 3, 1, 2).float()
    frame_resized = _resize_tchw(frame_tchw, target_h, target_w)
    return frame_resized.permute(0, 2, 3, 1).to(output_dtype)


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

    target_h, target_w = frames[0].shape[1], frames[0].shape[2]

    normalized = []
    for i, frame in enumerate(frames):
        h, w = frame.shape[1], frame.shape[2]
        if not _needs_resize(h, w, target_h, target_w):
            normalized.append(frame)
        else:
            logger.debug(f"Resized frame {i} from {w}x{h} to {target_w}x{target_h}")
            normalized.append(_resize_thwc(frame, target_h, target_w, frame.dtype))

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

        if height is not None and width is not None:
            _, _, h, w = frame.shape
            if _needs_resize(h, w, height, width):
                logger.debug(f"Resized frame from {w}x{h} to {width}x{height}")
                frame = _resize_tchw(frame, height, width)

        frames.append(frame)

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
