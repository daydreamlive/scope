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

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image


def vace_encode_frames(
    vae,
    frames,
    ref_images,
    masks=None,
    pad_to_96=True,
    use_cache=True,
    inactive_cache=None,
    reactive_cache=None,
):
    """
    Encode frames and reference images via VAE for VACE conditioning.

    Args:
        vae: VAE model wrapper (TAEWrapper or WanVAEWrapper)
        frames: List of video frames [B, C, F, H, W] or single frame [C, F, H, W]
        ref_images: List of reference images, one list per batch element
                   Each element is a list of reference images [C, 1, H, W]
        masks: Optional list of masks [B, 1, F, H, W] for masked video generation
        pad_to_96: Whether to pad to 96 channels (default True). Set False when masks will be added later.
        use_cache: Whether to use streaming encode cache for frames (default True).
                   Set False for one-off encoding (e.g., reference images only mode).
                   When masks are provided, caching is handled automatically based on
                   mask content: conditioning mode (all-1s masks) uses cache for both
                   streams, while extension/inpainting mode (mixed masks) skips cache
                   for reactive to weaken temporal blending.
        inactive_cache: Explicit encoder cache for inactive stream (TAE only).
                       Create via vae.create_encoder_cache(). Reuse across chunks
                       for temporal continuity. If None, uses VAE's default cache.
        reactive_cache: Explicit encoder cache for reactive stream (TAE only).
                       Must be separate from inactive_cache to prevent memory pollution.

    Returns:
        List of concatenated latents [ref_latents + frame_latents]

    Note:
        For TAE with masked encoding (depth/flow/pose/inpainting), you MUST provide
        separate inactive_cache and reactive_cache to prevent MemBlock memory pollution.
        WanVAE ignores these caches as its CausalConv3d doesn't have this issue.
    """
    if ref_images is None:
        ref_images = [None] * len(frames)
    else:
        assert len(frames) == len(ref_images)

    # Get VAE dtype for consistent encoding
    vae_dtype = next(vae.parameters()).dtype

    # Encode frames (with optional masking)
    # Note: WanVAEWrapper expects [B, C, F, H, W] and returns [B, F, C, H, W]
    if masks is None:
        # Single encode path - no masks, just encode frames directly
        # Stack list of [C, F, H, W] -> [B, C, F, H, W]
        frames_stacked = torch.stack(frames, dim=0)
        frames_stacked = frames_stacked.to(dtype=vae_dtype)
        # Use provided cache setting (use_cache=False for reference-only mode with dummy frames)
        latents_out = vae.encode_to_latent(frames_stacked, use_cache=use_cache)
        # Convert [B, F, C, H, W] -> list of [C, F, H, W] (transpose to channel-first)
        latents = [lat.permute(1, 0, 2, 3) for lat in latents_out]
    else:
        # Dual encode path for masked video generation
        # Each stream needs its own cache to prevent TAE's MemBlock memory pollution
        masks = [torch.where(m > 0.5, 1.0, 0.0) for m in masks]
        inactive = [i * (1 - m) + 0 * m for i, m in zip(frames, masks, strict=False)]
        reactive = [i * m + 0 * (1 - m) for i, m in zip(frames, masks, strict=False)]

        inactive_stacked = torch.stack(inactive, dim=0).to(dtype=vae_dtype)
        reactive_stacked = torch.stack(reactive, dim=0).to(dtype=vae_dtype)

        # Auto-detect mode based on mask content and handle caching appropriately:
        # - Conditioning mode (mask all 1s): inactive=zeros, reactive=content → both use cache
        # - Extension/inpainting mode (mixed mask): reactive skips cache to weaken temporal blending
        is_conditioning_mode = all((m > 0.5).all() for m in masks)

        # Encode with separate caches for temporal continuity without cross-contamination
        inactive_out = vae.encode_to_latent(
            inactive_stacked, use_cache=True, encoder_cache=inactive_cache
        )
        reactive_out = vae.encode_to_latent(
            reactive_stacked,
            use_cache=is_conditioning_mode,
            encoder_cache=reactive_cache,
        )

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

            if masks is not None:
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


def vace_encode_masks(masks, ref_images=None, vae_stride=(4, 8, 8)):
    """
    Encode masks for VACE context at VAE latent resolution.

    Args:
        masks: List of masks [B, 1, F, H, W]
        ref_images: List of reference images (to determine padding)
        vae_stride: VAE downsampling stride (default: (4, 8, 8))

    Returns:
        List of encoded masks at latent resolution
    """
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
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Load and prepare reference images for VACE conditioning.

    Uses crop-to-fill strategy: scales image to cover target dimensions, then
    center-crops to exact size. This avoids padding artifacts at the cost of
    losing a small amount of edge content when aspect ratios differ.

    Args:
        ref_image_paths: List of paths to reference images
        target_height: Target frame height
        target_width: Target frame width
        device: Target device

    Returns:
        Tuple of (prepared_images, spatial_masks) where:
        - prepared_images: List of image tensors [C, 1, H, W] normalized to [-1, 1]
        - spatial_masks: List of mask tensors [1, 1, H, W] indicating padding regions
          (0=image region to preserve, 1=padding region to generate). Always all-zeros
          with crop-to-fill since entire frame is image content.
    """
    prepared_refs = []
    spatial_masks = []

    for ref_path in ref_image_paths:
        # Load image
        ref_img = Image.open(ref_path).convert("RGB")

        # Convert to tensor and normalize to [-1, 1]
        ref_img = TF.to_tensor(ref_img).sub_(0.5).div_(0.5).unsqueeze(1)

        # Resize/pad to target size
        if ref_img.shape[-2:] != (target_height, target_width):
            ref_height, ref_width = ref_img.shape[-2:]

            # Scale to fill (crop-to-fit) - use max to ensure image covers target
            scale = max(target_height / ref_height, target_width / ref_width)
            new_height = int(ref_height * scale)
            new_width = int(ref_width * scale)

            # Resize to cover target dimensions
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

            # Center-crop to target size
            top = (new_height - target_height) // 2
            left = (new_width - target_width) // 2
            ref_img = resized_image[
                :, :, top : top + target_height, left : left + target_width
            ].to(device)

            # No padding, so spatial mask is all zeros (entire frame is image)
            spatial_mask = torch.zeros(
                (1, 1, target_height, target_width),
                device=device,
                dtype=torch.float32,
            )
            spatial_masks.append(spatial_mask)
        else:
            ref_img = ref_img.to(device)

            # No padding needed, so mask is all zeros (all image, no padding)
            spatial_mask = torch.zeros(
                (1, 1, target_height, target_width),
                device=device,
                dtype=torch.float32,
            )
            spatial_masks.append(spatial_mask)

        prepared_refs.append(ref_img)

    return prepared_refs, spatial_masks
