"""RAFT model loading utilities for optical flow pipeline."""

import logging

logger = logging.getLogger(__name__)

# Default resolution for optical flow computation
DEFAULT_HEIGHT = 512
DEFAULT_WIDTH = 512


def load_raft_model(use_large_model: bool, device: str = "cuda"):
    """Load RAFT model from torchvision.

    Args:
        use_large_model: If True, load RAFT Large. Otherwise load RAFT Small.
        device: Device to load model on.

    Returns:
        Tuple of (model, weights) where weights can be used for transforms.
    """
    from torchvision.models.optical_flow import (
        Raft_Large_Weights,
        Raft_Small_Weights,
        raft_large,
        raft_small,
    )

    if use_large_model:
        weights = Raft_Large_Weights.DEFAULT
        model = raft_large(weights=weights, progress=True)
    else:
        weights = Raft_Small_Weights.DEFAULT
        model = raft_small(weights=weights, progress=True)

    model = model.to(device=device)
    model.eval()
    return model, weights
