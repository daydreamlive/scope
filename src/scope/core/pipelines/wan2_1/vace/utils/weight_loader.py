"""
Utility for selectively loading VACE-specific weights into LongLive model.

This module extracts only VACE components (vace_blocks, vace_patch_embedding)
from a VACE checkpoint and loads them into a CausalVaceWanModel that's already
been initialized with LongLive weights.
"""

import logging

from scope.core.pipelines.utils import load_state_dict

logger = logging.getLogger(__name__)


def load_vace_weights_only(model, vace_checkpoint_path: str) -> None:
    """
    load_vace_weights_only: Load only VACE-specific weights from checkpoint into model.

    Extracts and loads:
    - vace_blocks.* (VACE attention blocks for hint generation)
    - vace_patch_embedding.* (Conv3D for encoding reference images)

    Skips all base model weights since those come from LongLive.

    Args:
        model: CausalVaceWanModel instance (already has LongLive base weights)
                May be wrapped by PEFT, in which case we unwrap to access base_model.
                Typically called before PEFT wrapping to avoid unwrapping issues.
        vace_checkpoint_path: Path to VACE safetensors checkpoint

    Returns:
        None (modifies model in-place)
    """
    logger.info(
        f"load_vace_weights_only: Loading VACE-specific weights from {vace_checkpoint_path}"
    )

    # Check if model is PEFT-wrapped and unwrap if needed
    actual_model = model
    is_peft_wrapped = hasattr(model, "peft_config") or hasattr(model, "base_model")
    if is_peft_wrapped:
        logger.info(
            "load_vace_weights_only: Detected PEFT-wrapped model, accessing base_model"
        )
        logger.info(
            f"load_vace_weights_only: Original model type: {type(model).__name__}"
        )
        logger.info(
            f"load_vace_weights_only: Has base_model: {hasattr(model, 'base_model')}"
        )
        logger.info(
            f"load_vace_weights_only: Has base_model.model: {hasattr(model, 'base_model') and hasattr(model.base_model, 'model')}"
        )

        if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
            actual_model = model.base_model.model
        elif hasattr(model, "base_model"):
            actual_model = model.base_model
        logger.info(
            f"load_vace_weights_only: Unwrapped to {type(actual_model).__name__}"
        )
        logger.info(
            f"load_vace_weights_only: Unwrapped model has vace_blocks: {hasattr(actual_model, 'vace_blocks')}"
        )
        logger.info(
            f"load_vace_weights_only: Unwrapped model has vace_patch_embedding: {hasattr(actual_model, 'vace_patch_embedding')}"
        )
    else:
        logger.info(
            f"load_vace_weights_only: Model not PEFT-wrapped, type: {type(model).__name__}"
        )
        logger.info(
            f"load_vace_weights_only: Model has vace_blocks: {hasattr(model, 'vace_blocks')}"
        )
        logger.info(
            f"load_vace_weights_only: Model has vace_patch_embedding: {hasattr(model, 'vace_patch_embedding')}"
        )

    # Load full VACE checkpoint
    state_dict = load_state_dict(vace_checkpoint_path)

    # Filter to only VACE-specific keys
    vace_keys = [
        "vace_blocks.",
        "vace_patch_embedding.",
    ]

    vace_state_dict = {}
    for key, value in state_dict.items():
        if any(key.startswith(prefix) for prefix in vace_keys):
            vace_state_dict[key] = value

    if not vace_state_dict:
        raise ValueError(
            f"load_vace_weights_only: No VACE-specific weights found in checkpoint. "
            f"Expected keys starting with: {vace_keys}"
        )

    logger.info(
        f"load_vace_weights_only: Found {len(vace_state_dict)} VACE-specific parameters"
    )
    logger.info(
        f"load_vace_weights_only: VACE components: vace_blocks ({sum(1 for k in vace_state_dict if 'vace_blocks' in k)} params), "
        f"vace_patch_embedding ({sum(1 for k in vace_state_dict if 'vace_patch_embedding' in k)} params)"
    )

    # Debug: Check shapes before loading
    logger.info(
        "load_vace_weights_only: Checking vace_patch_embedding compatibility..."
    )
    if "vace_patch_embedding.weight" in vace_state_dict:
        ckpt_shape = vace_state_dict["vace_patch_embedding.weight"].shape
        model_shape = actual_model.vace_patch_embedding.weight.shape
        logger.info(
            f"load_vace_weights_only: Checkpoint vace_patch_embedding.weight shape: {ckpt_shape}"
        )
        logger.info(
            f"load_vace_weights_only: Model vace_patch_embedding.weight shape: {model_shape}"
        )
        if ckpt_shape != model_shape:
            logger.error(
                "load_vace_weights_only: SHAPE MISMATCH! Cannot load vace_patch_embedding weights"
            )

    # Debug: List some sample keys from vace_state_dict
    sample_vace_keys = list(vace_state_dict.keys())[:10]
    logger.info(
        f"load_vace_weights_only: Sample keys from checkpoint: {sample_vace_keys}"
    )

    # Debug: List some sample keys from actual_model
    model_vace_keys = [
        k
        for k in actual_model.state_dict().keys()
        if any(k.startswith(prefix) for prefix in vace_keys)
    ][:10]
    logger.info(f"load_vace_weights_only: Sample VACE keys in model: {model_vace_keys}")

    # Load into actual model (not PEFT wrapper)
    missing_keys, unexpected_keys = actual_model.load_state_dict(
        vace_state_dict, strict=False
    )

    # Filter out expected missing keys (all the base model weights)
    actual_missing = [
        k for k in missing_keys if any(k.startswith(prefix) for prefix in vace_keys)
    ]

    if actual_missing:
        logger.warning(
            f"load_vace_weights_only: Missing VACE keys (first 20): {actual_missing[:20]}"
        )

    if unexpected_keys:
        logger.warning(
            f"load_vace_weights_only: Unexpected keys (first 20): {unexpected_keys[:20]}"
        )
        logger.error(
            "load_vace_weights_only: CRITICAL - VACE weights marked as unexpected! Model structure mismatch!"
        )
        logger.error(
            "load_vace_weights_only: This likely means the model doesn't have the VACE structure (vace_blocks, vace_patch_embedding)"
        )
        logger.error(
            "load_vace_weights_only: Check that CausalVaceWanModel was used, not CausalWanModel"
        )

    # Debug: Verify vace_patch_embedding was actually loaded
    patch_weight = actual_model.vace_patch_embedding.weight
    pw_min, pw_max = patch_weight.min().item(), patch_weight.max().item()
    pw_mean = patch_weight.mean().item()
    pw_std = patch_weight.std().item()
    pw_checksum = patch_weight.abs().sum().item()
    logger.info(
        f"load_vace_weights_only[COMPOSITION]: After loading, vace_patch_embedding.weight stats: min={pw_min:.6f}, max={pw_max:.6f}, mean={pw_mean:.6f}, std={pw_std:.6f}, checksum={pw_checksum:.6f}"
    )
    if pw_min == 0.0 and pw_max == 0.0:
        logger.error(
            "load_vace_weights_only: CRITICAL - vace_patch_embedding weights are all zeros! Loading failed!"
        )
        raise RuntimeError(
            "VACE weight loading failed - vace_patch_embedding weights are all zeros"
        )

    # Verify vace_blocks weights were loaded
    for i in range(min(3, len(actual_model.vace_blocks))):
        block_checksum = sum(
            p.abs().sum().item() for p in actual_model.vace_blocks[i].parameters()
        )
        logger.info(
            f"load_vace_weights_only[COMPOSITION]: vace_blocks[{i}] weight checksum={block_checksum:.6f}"
        )

    logger.info("load_vace_weights_only: Successfully loaded VACE weights")
