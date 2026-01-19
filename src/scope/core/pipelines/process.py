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


def normalize_frame_sizes(
    frames: list[torch.Tensor],
    target_height: int | None = None,
    target_width: int | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> list[torch.Tensor]:
    """Normalize all frames to match target dimensions.

    Frames may have different sizes (e.g., from switching video sources or
    resolution changes). This function resizes all frames to the target
    height and width to ensure they can be stacked.

    Args:
        frames: List of tensors in THWC format
        target_height: Target height for all frames. If None, uses first frame's height.
        target_width: Target width for all frames. If None, uses first frame's width.
        device: Target device to move frames to before resizing. If None, frames remain on their current device.
        dtype: Target dtype to convert frames to. If None, frames keep their original dtype.

    Returns:
        List of tensors all with the same H and W dimensions
    """
    if not frames:
        return frames

    target_h = target_height if target_height is not None else frames[0].shape[1]
    target_w = target_width if target_width is not None else frames[0].shape[2]
    output_dtype = dtype if dtype is not None else frames[0].dtype

    normalized = []
    for i, frame in enumerate(frames):
        # Move to target device if specified (before resizing for efficiency)
        if device is not None:
            frame = frame.to(device=device)
        frame = frame.to(dtype=output_dtype)

        h, w = frame.shape[1], frame.shape[2]
        if not _needs_resize(h, w, target_h, target_w):
            normalized.append(frame)
        else:
            logger.debug(f"Resized frame {i} from {w}x{h} to {target_w}x{target_h}")
            normalized.append(_resize_thwc(frame, target_h, target_w, output_dtype))

    return normalized


def preprocess_chunk(
    chunk: list[torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
    height: int | None = None,
    width: int | None = None,
) -> torch.Tensor:
    # Normalize frame sizes while still in THWC format, moving to device and converting dtype
    chunk = normalize_frame_sizes(
        chunk, target_height=height, target_width=width, device=device, dtype=dtype
    )

    # Stack frames (in THWC format) and rearrange once to get BCTHW tensor
    chunk = rearrange(torch.stack(chunk, dim=1), "B T H W C -> B C T H W")
    # Normalize to [-1, 1] range
    return chunk / 255.0 * 2.0 - 1.0


def postprocess_chunk(chunk: torch.Tensor) -> torch.Tensor:
    # chunk is a BTCHW tensor
    # Drop the batch dim
    chunk = rearrange(chunk.squeeze(0), "T C H W -> T H W C")
    # Normalize to [0, 1]
    return (chunk / 2 + 0.5).clamp(0, 1).float()
