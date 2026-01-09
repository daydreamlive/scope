"""
VACE encoding utilities for reference image conditioning.

Adapted from https://github.com/ali-vilab/VACE/blob/48eb44f1c4be87cc65a98bff985a26976841e9f3/vace/models/wan/wan_vace.py

For conditioning modes (following original VACE architecture):
- Conditioning maps (depth, flow, pose, etc.) are 3-channel RGB from annotators
- Treated as vace_input_frames through standard encoding path
- vace_input_masks defaults to ones (all white) when not provided - goes through masking path
- vace_ref_images = optional (can be combined with conditioning for style + structure)
- Standard path: vace_encode_frames -> vace_encode_masks -> vace_latent
"""

import os

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image


_ZERO_INACTIVE_LATENT_CACHE: dict[tuple, torch.Tensor] = {}
_FULL_MASK_ENCODED_MASK_CACHE: dict[tuple, torch.Tensor] = {}


def _get_zero_inactive_latent(
    vae,
    *,
    channels: int,
    frames: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    key = (id(vae), channels, frames, height, width, str(device), str(dtype))
    cached = _ZERO_INACTIVE_LATENT_CACHE.get(key)
    if cached is not None:
        return cached

    zeros_pixel = torch.zeros(
        (1, channels, frames, height, width),
        device=device,
        dtype=dtype,
    )
    zeros_latent_out = vae.encode_to_latent(zeros_pixel, use_cache=False)
    zeros_latent = zeros_latent_out[0].permute(1, 0, 2, 3).contiguous()

    _ZERO_INACTIVE_LATENT_CACHE[key] = zeros_latent
    return zeros_latent


def vace_encode_frames(
    vae,
    frames,
    ref_images,
    masks=None,
    pad_to_96=True,
    use_cache=True,
    *,
    full_mask: bool = False,
):
    """
    Encode frames and reference images via VAE for VACE conditioning.

    Args:
        vae: VAE model wrapper
        frames: List of video frames [B, C, F, H, W] or single frame [C, F, H, W]
        ref_images: List of reference images, one list per batch element
                   Each element is a list of reference images [C, 1, H, W]
        masks: Optional list of masks [B, 1, F, H, W] for masked video generation
        pad_to_96: Whether to pad to 96 channels (default True). Set False when masks will be added later.
        use_cache: Whether to use streaming encode cache for frames (default True).
                   Set False for one-off encoding (e.g., reference images only mode).
        full_mask: If True, masks represent an all-ones full-frame mask (white=generate).
                   Enables optional fastpaths for VACE when `SCOPE_VACE_FULL_MASK_FASTPATH=1`.

    Returns:
        List of concatenated latents [ref_latents + frame_latents]
    """
    use_full_mask_fastpath = full_mask and os.getenv("SCOPE_VACE_FULL_MASK_FASTPATH", "0") == "1"
    uses_mask_path = masks is not None or use_full_mask_fastpath

    frames_stacked: torch.Tensor | None = None
    if isinstance(frames, torch.Tensor):
        frames_stacked = frames
        batch_size = int(frames_stacked.shape[0])
    else:
        batch_size = len(frames)

    if ref_images is None:
        ref_images = [None] * batch_size
    else:
        assert batch_size == len(ref_images)

    # Get VAE dtype for consistent encoding
    vae_dtype = next(vae.parameters()).dtype

    # Encode frames (with optional masking)
    # Note: WanVAEWrapper expects [B, C, F, H, W] and returns [B, F, C, H, W]
    if masks is None:
        if use_full_mask_fastpath:
            # Full-mask (all ones) means the inactive branch is identically the encoding of
            # all-zero pixels. Avoid materializing full-resolution masks and avoid re-encoding
            # the inactive branch every chunk.
            if frames_stacked is None:
                frames_stacked = torch.stack(frames, dim=0)
            frames_stacked = frames_stacked.to(dtype=vae_dtype)

            reactive_out = vae.encode_to_latent(frames_stacked, use_cache=use_cache)
            reactive_transposed = [lat.permute(1, 0, 2, 3) for lat in reactive_out]

            zero_latent = _get_zero_inactive_latent(
                vae,
                channels=frames_stacked.shape[1],
                frames=frames_stacked.shape[2],
                height=frames_stacked.shape[3],
                width=frames_stacked.shape[4],
                device=frames_stacked.device,
                dtype=vae_dtype,
            )
            latents = [torch.cat((zero_latent, c), dim=0) for c in reactive_transposed]
        else:
            # Stack list of [C, F, H, W] -> [B, C, F, H, W]
            if frames_stacked is None:
                frames_stacked = torch.stack(frames, dim=0)
            frames_stacked = frames_stacked.to(dtype=vae_dtype)
            # Use provided cache setting (use_cache=False for reference-only mode with dummy frames)
            latents_out = vae.encode_to_latent(frames_stacked, use_cache=use_cache)
            # Convert [B, F, C, H, W] -> list of [C, F, H, W] (transpose to channel-first)
            latents = [lat.permute(1, 0, 2, 3) for lat in latents_out]
    else:
        if use_full_mask_fastpath:
            # For the common "mask omitted" case we currently default to an all-ones
            # mask, which makes the inactive branch identically zero. We can avoid
            # re-encoding that inactive branch every chunk by caching a single
            # zero-latent (per shape/device/dtype) and only encoding the reactive
            # (full) frames.
            if frames_stacked is None:
                frames_stacked = torch.stack(frames, dim=0)
            frames_stacked = frames_stacked.to(dtype=vae_dtype)
            reactive_out = vae.encode_to_latent(frames_stacked, use_cache=use_cache)
            reactive_transposed = [lat.permute(1, 0, 2, 3) for lat in reactive_out]

            zero_latent = _get_zero_inactive_latent(
                vae,
                channels=frames_stacked.shape[1],
                frames=frames_stacked.shape[2],
                height=frames_stacked.shape[3],
                width=frames_stacked.shape[4],
                device=frames_stacked.device,
                dtype=vae_dtype,
            )
            latents = [torch.cat((zero_latent, c), dim=0) for c in reactive_transposed]
        else:
            masks = [torch.where(m > 0.5, 1.0, 0.0) for m in masks]
            inactive = [
                i * (1 - m) + 0 * m for i, m in zip(frames, masks, strict=False)
            ]
            reactive = [i * m + 0 * (1 - m) for i, m in zip(frames, masks, strict=False)]
            inactive_stacked = torch.stack(inactive, dim=0).to(dtype=vae_dtype)
            reactive_stacked = torch.stack(reactive, dim=0).to(dtype=vae_dtype)
            # Default to cache=True for streaming consistency, but allow stateless
            # encoding (use_cache=False) for hybrid modes that also encode `video`.
            inactive_out = vae.encode_to_latent(inactive_stacked, use_cache=use_cache)
            reactive_out = vae.encode_to_latent(reactive_stacked, use_cache=use_cache)
            # Transpose [B, F, C, H, W] -> [B, C, F, H, W] and concatenate along channel dim
            inactive_transposed = [lat.permute(1, 0, 2, 3) for lat in inactive_out]
            reactive_transposed = [lat.permute(1, 0, 2, 3) for lat in reactive_out]
            latents = [
                torch.cat((u, c), dim=0)
                for u, c in zip(inactive_transposed, reactive_transposed, strict=False)
            ]

    # Concatenate reference images if provided
    cat_latents = []
    for latent, refs in zip(latents, ref_images, strict=False):
        if refs is not None:
            # Stack refs: list of [C, 1, H, W] -> [1, C, num_refs, H, W]
            refs_stacked = torch.stack(refs, dim=0)
            # Convert to VAE dtype (e.g., bfloat16)
            refs_stacked = refs_stacked.to(dtype=vae_dtype)
            # Encode: [1, C, num_refs, H, W] -> [1, num_refs, C, H, W]
            # Reference images are static, so cache doesn't matter, but use False to avoid affecting video cache
            ref_latent_out = vae.encode_to_latent(refs_stacked, use_cache=False)
            # Get first batch element and transpose: [num_refs, C, H, W] -> [C, num_refs, H, W]
            ref_latent_batch = ref_latent_out[0].permute(1, 0, 2, 3)

            if uses_mask_path:
                # Pad reference latents with zeros for mask channel
                zeros = torch.zeros_like(ref_latent_batch)
                ref_latent_batch = torch.cat((ref_latent_batch, zeros), dim=0)

            # Concatenate: [ref_frames, video_frames] along frame dim (dim=1)
            # ref_latent_batch: [C, num_refs, H, W], latent: [C, F, H, W]
            latent = torch.cat([ref_latent_batch, latent], dim=1)

        # Pad latents to 96 channels for VACE compatibility (if requested)
        # VACE was trained with 96 channels (16 base * 6 for masked video generation)
        # For R2V mode without masks, we pad with zeros
        # For depth mode, padding happens after mask concatenation (pad_to_96=False)
        current_channels = latent.shape[0]
        if pad_to_96 and current_channels < 96:
            pad_channels = 96 - current_channels
            padding = torch.zeros(
                (pad_channels, latent.shape[1], latent.shape[2], latent.shape[3]),
                dtype=latent.dtype,
                device=latent.device,
            )
            latent = torch.cat([latent, padding], dim=0)

        cat_latents.append(latent)

    return cat_latents


def _get_full_mask_encoded_mask(
    *,
    num_frames: int,
    height: int,
    width: int,
    ref_length: int,
    device: torch.device,
    dtype: torch.dtype,
    vae_stride: tuple[int, int, int],
) -> torch.Tensor:
    key = (
        num_frames,
        height,
        width,
        ref_length,
        str(device),
        str(dtype),
        tuple(int(v) for v in vae_stride),
    )
    cached = _FULL_MASK_ENCODED_MASK_CACHE.get(key)
    if cached is not None:
        return cached

    new_depth = int((num_frames + (vae_stride[0] - 1)) // vae_stride[0])
    latent_height = 2 * (int(height) // (vae_stride[1] * 2))
    latent_width = 2 * (int(width) // (vae_stride[2] * 2))
    mask_channels = int(vae_stride[1]) * int(vae_stride[2])

    base = torch.ones(
        (mask_channels, new_depth, latent_height, latent_width),
        device=device,
        dtype=dtype,
    )
    if ref_length > 0:
        pad = torch.zeros(
            (mask_channels, int(ref_length), latent_height, latent_width),
            device=device,
            dtype=dtype,
        )
        base = torch.cat((pad, base), dim=1)

    _FULL_MASK_ENCODED_MASK_CACHE[key] = base
    return base


def vace_encode_masks(
    masks,
    ref_images=None,
    vae_stride=(4, 8, 8),
    *,
    full_mask: bool = False,
    batch_size: int | None = None,
    num_frames: int | None = None,
    height: int | None = None,
    width: int | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
):
    """
    Encode masks for VACE context at VAE latent resolution.

    Args:
        masks: List of masks [B, 1, F, H, W]
        ref_images: List of reference images (to determine padding)
        vae_stride: VAE downsampling stride (default: (4, 8, 8))

    Returns:
        List of encoded masks at latent resolution
    """
    use_full_mask_fastpath = full_mask and os.getenv("SCOPE_VACE_FULL_MASK_FASTPATH", "0") == "1"

    if use_full_mask_fastpath:
        inferred_batch = batch_size
        if inferred_batch is None:
            if masks is not None:
                inferred_batch = len(masks)
            elif ref_images is not None:
                inferred_batch = len(ref_images)
        if inferred_batch is None:
            raise ValueError(
                "vace_encode_masks: batch_size is required when masks=None and ref_images=None"
            )

        inferred_num_frames = num_frames
        inferred_height = height
        inferred_width = width
        inferred_device = device
        inferred_dtype = dtype
        if masks is not None:
            sample = masks[0]
            if isinstance(sample, torch.Tensor) and sample.ndim == 4:
                _, inferred_num_frames, inferred_height, inferred_width = sample.shape
                inferred_device = sample.device
                inferred_dtype = sample.dtype

        if (
            inferred_num_frames is None
            or inferred_height is None
            or inferred_width is None
            or inferred_device is None
            or inferred_dtype is None
        ):
            raise ValueError(
                "vace_encode_masks: num_frames/height/width/device/dtype are required for full_mask fastpath"
            )

        if ref_images is None:
            ref_images = [None] * int(inferred_batch)
        else:
            assert int(inferred_batch) == len(ref_images)

        result_masks = []
        for refs in ref_images:
            ref_len = len(refs) if refs is not None else 0
            result_masks.append(
                _get_full_mask_encoded_mask(
                    num_frames=int(inferred_num_frames),
                    height=int(inferred_height),
                    width=int(inferred_width),
                    ref_length=int(ref_len),
                    device=inferred_device,
                    dtype=inferred_dtype,
                    vae_stride=tuple(int(v) for v in vae_stride),
                )
            )
        return result_masks

    if ref_images is None:
        ref_images = [None] * len(masks)
    else:
        assert len(masks) == len(ref_images)

    result_masks = []
    for mask, refs in zip(masks, ref_images, strict=False):
        c, depth, height, width = mask.shape
        new_depth = int((depth + 3) // vae_stride[0])
        height = 2 * (int(height) // (vae_stride[1] * 2))
        width = 2 * (int(width) // (vae_stride[2] * 2))

        # Reshape mask
        mask = mask[0, :, :, :]
        mask = mask.view(depth, height, vae_stride[1], width, vae_stride[2])
        mask = mask.permute(2, 4, 0, 1, 3)
        mask = mask.reshape(vae_stride[1] * vae_stride[2], depth, height, width)

        # Interpolate to latent resolution
        mask = F.interpolate(
            mask.unsqueeze(0), size=(new_depth, height, width), mode="nearest-exact"
        ).squeeze(0)

        # Add padding for reference images
        if refs is not None:
            length = len(refs)
            mask_pad = torch.zeros_like(mask[:, :length, :, :])
            mask = torch.cat((mask_pad, mask), dim=1)

        result_masks.append(mask)

    return result_masks


def vace_latent(z, m):
    """
    Concatenate latents with masks for VACE context.

    Args:
        z: Latent encodings
        m: Encoded masks

    Returns:
        List of concatenated [latent, mask] tensors
    """
    result = [torch.cat([zz, mm], dim=0) for zz, mm in zip(z, m, strict=False)]
    return result


def load_and_prepare_reference_images(
    ref_image_paths, target_height, target_width, device
):
    """
    Load and prepare reference images for VACE conditioning.

    Args:
        ref_image_paths: List of paths to reference images
        target_height: Target frame height
        target_width: Target frame width
        device: Target device

    Returns:
        List of prepared reference image tensors [C, 1, H, W]
    """
    prepared_refs = []

    for ref_path in ref_image_paths:
        # Load image
        ref_img = Image.open(ref_path).convert("RGB")

        # Convert to tensor and normalize to [-1, 1]
        ref_img = TF.to_tensor(ref_img).sub_(0.5).div_(0.5).unsqueeze(1)

        # Resize/pad to target size
        if ref_img.shape[-2:] != (target_height, target_width):
            ref_height, ref_width = ref_img.shape[-2:]

            # Create white canvas
            white_canvas = torch.ones(
                (3, 1, target_height, target_width), device=device
            )

            # Calculate scale to fit
            scale = min(target_height / ref_height, target_width / ref_width)
            new_height = int(ref_height * scale)
            new_width = int(ref_width * scale)

            # Resize
            resized_image = (
                F.interpolate(
                    ref_img.squeeze(1).unsqueeze(0),
                    size=(new_height, new_width),
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze(0)
                .unsqueeze(1)
            )

            # Center on canvas
            top = (target_height - new_height) // 2
            left = (target_width - new_width) // 2
            white_canvas[:, :, top : top + new_height, left : left + new_width] = (
                resized_image
            )
            ref_img = white_canvas
        else:
            ref_img = ref_img.to(device)

        prepared_refs.append(ref_img)

    return prepared_refs
