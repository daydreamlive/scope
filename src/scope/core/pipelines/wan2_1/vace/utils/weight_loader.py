"""
Utility for loading VACE-specific weights into a Wan2.1 based causal model.

This module loads VACE components (vace_blocks, vace_patch_embedding) from either:
- A VACE-only module file (e.g., Wan2_1-VACE_module_1_3B_bf16.safetensors) - loaded directly
- A full VACE checkpoint (e.g., Wan2.1-VACE-1.3B) - filtered to extract only VACE weights

The function auto-detects the checkpoint type and handles both cases.
"""

import logging

from scope.core.pipelines.utils import load_state_dict

logger = logging.getLogger(__name__)


def load_vace_weights_only(model, vace_checkpoint_path: str) -> None:
    """
    Load VACE-specific weights from checkpoint into model.

    Loads:
    - vace_blocks.* (VACE attention blocks for hint generation)
    - vace_patch_embedding.* (Conv3D for encoding reference images)

    Auto-detects checkpoint type:
    - VACE-only module: All keys are VACE keys, loaded directly
    - Full checkpoint: Contains base model weights, filtered to VACE keys only

    Args:
        model: CausalVaceWanModel instance (already has causal model base weights)
                May be wrapped by PEFT, in which case we unwrap to access base_model.
                Typically called before PEFT wrapping to avoid unwrapping issues.
        vace_checkpoint_path: Path to VACE safetensors checkpoint (module or full)

    Returns:
        None (modifies model in-place)
    """
    logger.info(
        f"load_vace_weights_only: Loading VACE-specific weights from {vace_checkpoint_path}"
    )

    # Determine the actual model to load weights into
    # If model is CausalVaceWanModel, use it directly (VACE attributes are at this level)
    # Otherwise, unwrap PEFT if present
    actual_model = model
    model_type_name = type(model).__name__

    logger.debug(f"load_vace_weights_only: Input model type: {model_type_name}")

    # Check if this is a CausalVaceWanModel (directly or indirectly)
    is_vace_model = model_type_name == "CausalVaceWanModel" or (
        hasattr(model, "causal_wan_model")
        and hasattr(model, "vace_patch_embedding")
        and hasattr(model, "vace_blocks")
    )

    if is_vace_model:
        # CausalVaceWanModel: VACE attributes are direct attributes of this wrapper
        # Keep any inner PEFT wrappers intact (they're inside causal_wan_model)
        logger.debug(
            "load_vace_weights_only: Detected CausalVaceWanModel, using directly"
        )
        logger.debug(
            f"load_vace_weights_only: Model has vace_blocks: {hasattr(actual_model, 'vace_blocks')}"
        )
        logger.debug(
            f"load_vace_weights_only: Model has vace_patch_embedding: {hasattr(actual_model, 'vace_patch_embedding')}"
        )
    else:
        # Not a CausalVaceWanModel, check if PEFT-wrapped and unwrap if needed
        is_peft_wrapped = hasattr(model, "peft_config") or hasattr(model, "base_model")
        if is_peft_wrapped:
            logger.debug(
                "load_vace_weights_only: Detected PEFT-wrapped model (non-VACE), accessing base_model"
            )

            if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
                actual_model = model.base_model.model
            elif hasattr(model, "base_model"):
                actual_model = model.base_model

            logger.debug(
                f"load_vace_weights_only: Unwrapped to {type(actual_model).__name__}"
            )
            logger.debug(
                f"load_vace_weights_only: Unwrapped model has vace_blocks: {hasattr(actual_model, 'vace_blocks')}"
            )
            logger.debug(
                f"load_vace_weights_only: Unwrapped model has vace_patch_embedding: {hasattr(actual_model, 'vace_patch_embedding')}"
            )
        else:
            logger.debug(
                f"load_vace_weights_only: Model not PEFT-wrapped, type: {model_type_name}"
            )
            logger.debug(
                f"load_vace_weights_only: Model has vace_blocks: {hasattr(model, 'vace_blocks')}"
            )
            logger.debug(
                f"load_vace_weights_only: Model has vace_patch_embedding: {hasattr(model, 'vace_patch_embedding')}"
            )

    # Load VACE checkpoint
    state_dict = load_state_dict(vace_checkpoint_path)

    # VACE-specific key prefixes
    vace_prefixes = [
        "vace_blocks.",
        "vace_patch_embedding.",
    ]

    # Check if this is a VACE-only module (all keys are VACE keys) or a full checkpoint
    all_keys_are_vace = all(
        any(key.startswith(prefix) for prefix in vace_prefixes)
        for key in state_dict.keys()
    )

    if all_keys_are_vace:
        # VACE-only module file - use directly without filtering
        logger.debug(
            f"load_vace_weights_only: Detected VACE-only module with {len(state_dict)} parameters, loading directly"
        )
        vace_state_dict = state_dict
    else:
        # Full checkpoint - filter to only VACE-specific keys
        logger.debug(
            "load_vace_weights_only: Detected full checkpoint, filtering for VACE keys"
        )
        vace_state_dict = {}
        for key, value in state_dict.items():
            if any(key.startswith(prefix) for prefix in vace_prefixes):
                vace_state_dict[key] = value

        if not vace_state_dict:
            raise ValueError(
                f"load_vace_weights_only: No VACE-specific weights found in checkpoint. "
                f"Expected keys starting with: {vace_prefixes}"
            )

        logger.debug(
            f"load_vace_weights_only: Filtered to {len(vace_state_dict)} VACE-specific parameters"
        )

    # Check shapes before loading
    if "vace_patch_embedding.weight" in vace_state_dict:
        ckpt_shape = vace_state_dict["vace_patch_embedding.weight"].shape
        model_shape = actual_model.vace_patch_embedding.weight.shape
        logger.debug(
            f"load_vace_weights_only: Checkpoint shape: {ckpt_shape}, Model shape: {model_shape}"
        )
        if ckpt_shape != model_shape:
            error_msg = (
                f"load_vace_weights_only: Shape mismatch for vace_patch_embedding.weight! "
                f"Checkpoint shape {ckpt_shape} != Model shape {model_shape}. "
                f"Cannot load incompatible VACE weights."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

    # Load into actual model (not PEFT wrapper)
    # Use assign=False to preserve existing tensor device/dtype
    missing_keys, unexpected_keys = actual_model.load_state_dict(
        vace_state_dict, strict=False, assign=False
    )

    # Filter out expected missing keys (all the base model weights)
    actual_missing = [
        k for k in missing_keys if any(k.startswith(prefix) for prefix in vace_prefixes)
    ]

    if actual_missing:
        error_msg = (
            f"load_vace_weights_only: Missing expected VACE keys in model. "
            f"This indicates the model structure doesn't match the checkpoint. "
            f"Missing keys (first 20): {actual_missing[:20]}"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

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

    # Verify vace_patch_embedding was actually loaded
    patch_weight = actual_model.vace_patch_embedding.weight
    pw_min, pw_max = patch_weight.min().item(), patch_weight.max().item()
    if pw_min == 0.0 and pw_max == 0.0:
        logger.error(
            "load_vace_weights_only: CRITICAL - vace_patch_embedding weights are all zeros! Loading failed!"
        )
        raise RuntimeError(
            "VACE weight loading failed - vace_patch_embedding weights are all zeros"
        )

    logger.debug("load_vace_weights_only: Successfully loaded VACE weights")
