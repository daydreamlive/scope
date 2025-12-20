"""
VACE encoding utilities for reference image conditioning.

Adapted from notes/VACE/vace/models/wan/wan_vace.py

For conditioning modes (following original VACE architecture):
- Conditioning maps (depth, flow, pose, etc.) are 3-channel RGB from annotators
- Treated as input_frames through standard encoding path
- input_masks defaults to ones (all white) when not provided - goes through masking path
- ref_images = optional (can be combined with conditioning for style + structure)
- Standard path: vace_encode_frames -> vace_encode_masks -> vace_latent
"""

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image


def vace_encode_frames(
    vae, frames, ref_images, masks=None, pad_to_96=True, use_cache=True
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
        use_cache: Whether to use VAE streaming cache (default True). Set False for per-chunk encoding.

    Returns:
        List of concatenated latents [ref_latents + frame_latents]
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
        # Stack list of [C, F, H, W] -> [B, C, F, H, W]
        frames_stacked = torch.stack(frames, dim=0)
        frames_stacked = frames_stacked.to(dtype=vae_dtype)
        # Use cache parameter to control temporal state sharing
        latents_out = vae.encode_to_latent(frames_stacked, use_cache=use_cache)
        # Convert [B, F, C, H, W] -> list of [C, F, H, W] (transpose to channel-first)
        latents = [lat.permute(1, 0, 2, 3) for lat in latents_out]
    else:
        masks = [torch.where(m > 0.5, 1.0, 0.0) for m in masks]
        inactive = [i * (1 - m) + 0 * m for i, m in zip(frames, masks, strict=False)]
        reactive = [i * m + 0 * (1 - m) for i, m in zip(frames, masks, strict=False)]
        inactive_stacked = torch.stack(inactive, dim=0).to(dtype=vae_dtype)
        reactive_stacked = torch.stack(reactive, dim=0).to(dtype=vae_dtype)
        # Use cache parameter for inactive portion
        # When use_cache=True: shares temporal state with video encoding for consistency
        # When use_cache=False: independent per-chunk encoding (e.g., extension mode)
        inactive_out = vae.encode_to_latent(inactive_stacked, use_cache=use_cache)
        # Reactive portion always uses cache=False since it's masked (will be inpainted)
        reactive_out = vae.encode_to_latent(reactive_stacked, use_cache=False)
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


def decode_vace_latent(vae, zs, ref_images=None):
    """
    Decode VAE latents, removing reference image frames.

    Args:
        vae: VAE model wrapper
        zs: List of latent tensors (may include reference frames)
        ref_images: List of reference image lists (to determine trim length)

    Returns:
        Decoded video frames
    """
    if ref_images is None:
        ref_images = [None] * len(zs)
    else:
        assert len(zs) == len(ref_images)

    # Trim reference frames
    trimmed_zs = []
    for z, refs in zip(zs, ref_images, strict=False):
        if refs is not None:
            # Remove reference frames from latent
            z = z[:, len(refs) :, :, :]
        trimmed_zs.append(z)

    return vae.decode(trimmed_zs)


def preprocess_depth_frames(depth_frames, target_height, target_width, device):
    """
    Preprocess depth frames for VACE depth encoding.

    Following original VACE architecture (notes/VACE/vace/annotators/depth.py line 37):
    - Depth maps must be 3-channel RGB format for VAE encoding
    - This matches how depth annotators output depth maps

    Args:
        depth_frames: Depth frames tensor [F, H, W] or [F, C, H, W]
        target_height: Target video height
        target_width: Target video width
        device: Target device

    Returns:
        Preprocessed depth frames [1, 3, F, H, W] (3-channel RGB, batch dim added)
    """
    import torch.nn.functional as F

    # Add channel dimension if needed
    if depth_frames.dim() == 3:
        # [F, H, W] -> [F, 1, H, W]
        depth_frames = depth_frames.unsqueeze(1)

    # Ensure single channel first (take first channel if already RGB)
    if depth_frames.shape[1] > 1:
        depth_frames = depth_frames[:, :1, :, :]

    # Resize if needed
    _, _, curr_h, curr_w = depth_frames.shape
    if curr_h != target_height or curr_w != target_width:
        depth_frames = F.interpolate(
            depth_frames,
            size=(target_height, target_width),
            mode="bilinear",
            align_corners=False,
        )

    # Normalize to [-1, 1] if not already
    if depth_frames.min() >= 0:
        depth_frames = depth_frames * 2.0 - 1.0

    # Convert grayscale to 3-channel RGB (matching original VACE depth annotator output)
    # [F, 1, H, W] -> [F, 3, H, W]
    depth_frames = depth_frames.repeat(1, 3, 1, 1)

    # Move to device and add batch dimension: [F, 3, H, W] -> [1, 3, F, H, W]
    depth_frames = depth_frames.to(device)
    depth_frames = depth_frames.unsqueeze(0).permute(0, 2, 1, 3, 4)

    return depth_frames


def extract_depth_chunk(depth_video, chunk_index, num_frames_per_chunk):
    """
    Extract a chunk of depth frames from full depth video.

    Args:
        depth_video: Full depth video [1, C, F, H, W]
        chunk_index: Chunk index (0-based)
        num_frames_per_chunk: Number of frames per chunk (typically 12)

    Returns:
        Depth chunk [1, C, num_frames_per_chunk, H, W]
    """
    start_frame = chunk_index * num_frames_per_chunk
    end_frame = start_frame + num_frames_per_chunk

    batch_size, channels, num_frames, height, width = depth_video.shape

    if end_frame > num_frames:
        raise ValueError(
            f"extract_depth_chunk: Chunk {chunk_index} exceeds video length. "
            f"Requested frames {start_frame}-{end_frame}, but video has only {num_frames} frames."
        )

    return depth_video[:, :, start_frame:end_frame, :, :]
