import logging

import torch
from einops import rearrange

logger = logging.getLogger(__name__)


def preprocess_chunk(
    chunk: list[torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
    height: int | None = None,
    width: int | None = None,
) -> torch.Tensor:
    frames = []

    for frame in chunk:
        # Move to pipeline device
        frame = frame.to(device=device, dtype=dtype)
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

        logger.debug(f"Resized frame from {H}x{W} to {height}x{width}")

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
