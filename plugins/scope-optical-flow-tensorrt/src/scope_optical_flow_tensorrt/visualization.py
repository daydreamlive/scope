"""Optical flow visualization utilities.

Converts raw optical flow vectors to RGB visualization images
suitable for VACE conditioning.
"""

import torch
from torchvision.utils import flow_to_image


def flow_to_rgb(flow: torch.Tensor) -> torch.Tensor:
    """Convert optical flow tensor to RGB visualization.

    Uses torchvision's flow_to_image which encodes flow vectors as RGB colors:
    - Hue encodes flow direction
    - Saturation encodes flow magnitude

    Args:
        flow: Optical flow tensor of shape [B, 2, H, W] or [2, H, W]
              Channel 0: horizontal displacement (u)
              Channel 1: vertical displacement (v)

    Returns:
        RGB visualization tensor of shape [B, 3, H, W] or [3, H, W]
        Values in [0, 255] uint8 range
    """
    # Handle unbatched input
    squeeze_batch = False
    if flow.dim() == 3:
        flow = flow.unsqueeze(0)
        squeeze_batch = True

    # flow_to_image expects [B, 2, H, W] and returns [B, 3, H, W] uint8
    rgb = flow_to_image(flow)

    if squeeze_batch:
        rgb = rgb.squeeze(0)

    return rgb


def create_zero_flow_rgb(height: int, width: int, device: torch.device) -> torch.Tensor:
    """Create RGB visualization for zero flow (no motion).

    Zero flow produces a specific color (typically dark/gray) in the
    flow visualization colormap.

    Args:
        height: Output height
        width: Output width
        device: Target device

    Returns:
        RGB tensor [3, H, W] in [0, 255] uint8 range
    """
    zero_flow = torch.zeros(1, 2, height, width, device=device)
    rgb = flow_to_image(zero_flow)
    return rgb.squeeze(0)
