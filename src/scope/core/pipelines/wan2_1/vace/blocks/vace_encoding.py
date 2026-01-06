"""
VACE Encoding Block for Conditioning Video Generation.

This block handles encoding of VACE conditioning inputs and prepares vace_context for
the denoising block. Supports flexible combinations:
- Reference images only (R2V): Static reference images for style/character consistency
- Conditioning input only: Per-chunk guidance (depth, flow, pose, scribble, etc.)
- Both combined: Reference images + conditioning input for style + structural guidance

The mode is implicit based on what inputs are provided - no explicit mode parameter needed.

For conditioning inputs (depth, flow, etc.), follows original VACE architecture:
- vace_input_frames = source RGB video frames OR conditioning maps (3-channel RGB from annotators)
- vace_input_masks = spatial control masks (ones for full-frame, regional for masked areas)
- Standard path: vace_encode_frames -> vace_encode_masks -> vace_latent
"""

import logging
from typing import Any

import torch
from diffusers.modular_pipelines import (
    ModularPipelineBlocks,
    PipelineState,
)
from diffusers.modular_pipelines.modular_pipeline_utils import (
    ComponentSpec,
    ConfigSpec,
    InputParam,
    OutputParam,
)

from ..utils.encoding import load_and_prepare_reference_images

logger = logging.getLogger(__name__)


class VaceEncodingBlock(ModularPipelineBlocks):
    """
    VACE encoding block with internal routing logic.

    Architectural Note: This block does NOT use AutoPipelineBlock pattern despite having
    multiple execution paths. Rationale:

    1. Single Operation: All paths perform the same conceptual operation (VACE encoding)
       with shared logic (VAE selection, reference image loading, validation).

    2. OR Condition: Block should run if EITHER vace_ref_images OR vace_input_frames is provided.
       AutoPipelineBlock uses sequential first-match logic, making OR conditions awkward
       (would require duplicate block instances with different triggers).

    3. Simple Skip Logic: The skip condition is trivial (3 lines in __call__) and
       self-contained. Using AutoPipelineBlock would add complexity without benefit.

    Compare to AutoPrepareLatentsBlock: That routes between fundamentally different
    operations (prepare fresh latents vs. encode video) with a single trigger (video).
    VACE has one operation with multiple potential triggers, better handled internally.
    """

    def __init__(self):
        super().__init__()
        # Explicit encoder caches for TAE VACE dual-encode (inactive + reactive)
        # These persist across chunks to maintain temporal continuity within each stream
        # while preventing MemBlock memory pollution between streams.
        # Lazily initialized on first use since we need the VAE to create caches.
        self._inactive_cache = None
        self._reactive_cache = None
        self._caches_initialized = False

    def clear_encoder_caches(self):
        """Clear encoder caches for a new video sequence."""
        self._inactive_cache = None
        self._reactive_cache = None
        self._caches_initialized = False

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("vae", torch.nn.Module),
        ]

    @property
    def expected_configs(self) -> list[ConfigSpec]:
        return [
            ConfigSpec("num_frame_per_block", 3),
            ConfigSpec("vae_temporal_downsample_factor", 4),
            ConfigSpec("device", torch.device("cuda")),
        ]

    @property
    def description(self) -> str:
        return "VaceEncodingBlock: Encode VACE context for conditioning (ref images, depth, flow, pose, etc.)"

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "vace_ref_images",
                default=None,
                type_hint=list[str] | None,
                description="List of reference image paths for style/character consistency (can be combined with vace_input_frames)",
            ),
            InputParam(
                "vace_input_frames",
                default=None,
                type_hint=torch.Tensor | None,
                description="VACE conditioning input frames [B, C, F, H, W]: source RGB video frames OR conditioning maps (depth, flow, pose, scribble, etc.). 12 frames per chunk, can be combined with vace_ref_images",
            ),
            InputParam(
                "vace_input_masks",
                default=None,
                type_hint=torch.Tensor | None,
                description="Spatial control masks [B, 1, F, H, W]: defines WHERE to apply conditioning (white=generate, black=preserve). Defaults to ones (all white) when None. Works with any vace_input_frames type.",
            ),
            InputParam(
                "height",
                type_hint=int,
                description="Target video height",
            ),
            InputParam(
                "width",
                type_hint=int,
                description="Target video width",
            ),
            InputParam(
                "current_start_frame",
                required=True,
                type_hint=int,
                description="Current starting frame index",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "vace_context",
                type_hint=list | None,
                description="Encoded VACE context for denoising block",
            ),
            OutputParam(
                "vace_ref_images",
                type_hint=list | None,
                description="Prepared reference images (for decode block)",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> tuple[Any, PipelineState]:
        block_state = self.get_block_state(state)

        vace_ref_images = block_state.vace_ref_images
        vace_input_frames = block_state.vace_input_frames
        current_start = block_state.current_start_frame

        # If neither input is provided, skip VACE conditioning
        if (
            vace_ref_images is None or len(vace_ref_images) == 0
        ) and vace_input_frames is None:
            block_state.vace_context = None
            block_state.vace_ref_images = None
            self.set_block_state(state, block_state)
            return components, state

        # Determine encoding path based on what's provided (implicit mode detection)
        has_ref_images = vace_ref_images is not None and len(vace_ref_images) > 0
        has_input_frames = vace_input_frames is not None

        if has_input_frames:
            # Standard VACE path: conditioning input (depth, flow, pose, etc.)
            # with optional reference images
            block_state.vace_context, block_state.vace_ref_images = (
                self._encode_with_conditioning(components, block_state, current_start)
            )
        elif has_ref_images:
            # Reference images only mode (R2V)
            # Encode reference images whenever they are provided
            block_state.vace_context, block_state.vace_ref_images = (
                self._encode_reference_only(components, block_state, current_start)
            )

        self.set_block_state(state, block_state)
        return components, state

    def _encode_reference_only(self, components, block_state, current_start):
        """
        Encode reference images only (R2V mode) using direct encoding.

        This implementation encodes reference images directly without dummy frames.
        See design.md for rationale on design differences from original VACE.

        R2V characteristics:
        - Static reference images (1-3 frames)
        - Encoded fresh each chunk (application layer manages reuse)
        - Direct encoding via VAE with use_cache=False
        - Padded to 96 channels for VACE compatibility
        - No masking path (masks=None)
        """
        ref_image_paths = block_state.vace_ref_images
        if ref_image_paths is None or len(ref_image_paths) == 0:
            logger.warning(
                "VaceEncodingBlock._encode_reference_only: No vace_ref_images provided"
            )
            return None, None

        # Load and prepare reference images
        prepared_refs = load_and_prepare_reference_images(
            ref_image_paths,
            block_state.height,
            block_state.width,
            components.config.device,
        )

        # Use main VAE with use_cache=False for encoding reference images.
        # use_cache=False creates a temporary one-off cache for normal encoding without
        # invoking the streaming encode implementation, avoiding temporal dimension issues
        # with small numbers of reference frames (1-2 frames).
        vae = components.vae

        # Encode only reference images (no dummy frames)
        # Stack refs: list of [C, 1, H, W] -> [1, C, num_refs, H, W]
        prepared_refs_stacked = torch.cat(prepared_refs, dim=1).unsqueeze(0)
        # Convert to VAE's dtype (typically bfloat16)
        vae_dtype = next(vae.parameters()).dtype
        prepared_refs_stacked = prepared_refs_stacked.to(dtype=vae_dtype)
        # Reference images are static, so use_cache=False to avoid affecting video cache
        ref_latents_out = vae.encode_to_latent(prepared_refs_stacked, use_cache=False)

        # Convert [1, num_refs, C, H, W] -> [C, num_refs, H, W] (transpose to channel-first)
        ref_latent_batch = ref_latents_out[0].permute(1, 0, 2, 3)

        # Pad to 96 channels for VACE compatibility
        # VACE was trained with 96 channels (16 base * 6 for masked video generation)
        # For R2V mode without masks, we pad with zeros
        current_channels = ref_latent_batch.shape[0]
        if current_channels < 96:
            pad_channels = 96 - current_channels
            padding = torch.zeros(
                (
                    pad_channels,
                    ref_latent_batch.shape[1],
                    ref_latent_batch.shape[2],
                    ref_latent_batch.shape[3],
                ),
                dtype=ref_latent_batch.dtype,
                device=ref_latent_batch.device,
            )
            ref_latent_batch = torch.cat([ref_latent_batch, padding], dim=0)

        # VACE context is just the reference images (no dummy frames for R2V)
        vace_context = [ref_latent_batch]

        # Return original paths, not tensors, so they can be reused in subsequent chunks
        return vace_context, ref_image_paths

    def _encode_with_conditioning(self, components, block_state, current_start):
        """
        Encode VACE input using the standard VACE path, with optional reference images.

        Supports any type of conditioning input (depth, flow, pose, scribble, etc.) following
        original VACE approach from https://github.com/ali-vilab/VACE/blob/48eb44f1c4be87cc65a98bff985a26976841e9f3/vace/models/wan/wan_vace.py:
        - vace_input_frames = conditioning maps (3-channel RGB from annotators)
        - vace_input_masks = spatial control masks (defaults to ones if None)
        - vace_ref_images = optional (for combined style + structural guidance)
        - Standard encoding: z0 = vace_encode_frames(vace_input_frames, ref_images, masks=vace_input_masks)
                           m0 = vace_encode_masks(vace_input_masks, ref_images)
                           z = vace_latent(z0, m0)

        Characteristics:
        - Per-chunk conditioning (12 frames matching output chunk)
        - Encoded every chunk (no caching)
        - Direct temporal correspondence with output
        - Uses standard VACE path with masking (produces 96 channels: 32 masked + 64 mask_encoding)
        - Can be combined with reference images for style + structure guidance
        """
        input_frames_data = block_state.vace_input_frames
        if input_frames_data is None:
            raise ValueError(
                f"VaceEncodingBlock._encode_with_conditioning: vace_input_frames required at chunk {current_start}"
            )

        # Validate vace_input_frames shape
        if input_frames_data.dim() != 5:
            raise ValueError(
                f"VaceEncodingBlock._encode_with_conditioning: vace_input_frames must be [B, C, F, H, W], got shape {input_frames_data.shape}"
            )

        batch_size, channels, num_frames, height, width = input_frames_data.shape

        # Validate resolution
        if height != block_state.height or width != block_state.width:
            raise ValueError(
                f"VaceEncodingBlock._encode_with_conditioning: Input resolution {height}x{width} "
                f"does not match target resolution {block_state.height}x{block_state.width}"
            )

        # Check if we have reference images too (for combined guidance)
        ref_image_paths = block_state.vace_ref_images
        has_ref_images = ref_image_paths is not None and len(ref_image_paths) > 0

        # Use main VAE for consistency with video encoding.
        # This ensures temporal consistency in inpainting mode where unmasked portions
        # are encoded in both VACE context and video latents.
        vae = components.vae

        # Import vace_utils for standard encoding path
        from ..utils.encoding import vace_encode_frames, vace_encode_masks, vace_latent

        # Ensure 3-channel input for VAE (conditioning maps should already be 3-channel RGB)
        if channels == 1:
            input_frames_data = input_frames_data.repeat(1, 3, 1, 1, 1)
        elif channels != 3:
            raise ValueError(
                f"VaceEncodingBlock._encode_with_conditioning: Expected 1 or 3 channels, got {channels}"
            )

        vae_dtype = next(vae.parameters()).dtype
        input_frames_data = input_frames_data.to(dtype=vae_dtype)

        # Convert to list of [C, F, H, W] for vace_encode_frames
        input_frames_list = [input_frames_data[b] for b in range(batch_size)]

        # Get vace_input_masks from block_state or default to ones (all white)
        input_masks_data = block_state.vace_input_masks
        if input_masks_data is None:
            # Default to ones (all white) - apply conditioning everywhere
            input_masks_list = [
                torch.ones(
                    (1, num_frames, height, width),
                    dtype=vae_dtype,
                    device=input_frames_data.device,
                )
                for _ in range(batch_size)
            ]
        else:
            # Validate vace_input_masks shape
            if input_masks_data.dim() != 5:
                raise ValueError(
                    f"VaceEncodingBlock._encode_with_conditioning: vace_input_masks must be [B, 1, F, H, W], got shape {input_masks_data.shape}"
                )

            mask_batch, mask_channels, mask_frames, mask_height, mask_width = (
                input_masks_data.shape
            )
            if mask_channels != 1:
                raise ValueError(
                    f"VaceEncodingBlock._encode_with_conditioning: vace_input_masks must have 1 channel, got {mask_channels}"
                )
            if (
                mask_frames != num_frames
                or mask_height != height
                or mask_width != width
            ):
                raise ValueError(
                    f"VaceEncodingBlock._encode_with_conditioning: vace_input_masks shape mismatch: "
                    f"expected [B, 1, {num_frames}, {height}, {width}], got [B, 1, {mask_frames}, {mask_height}, {mask_width}]"
                )

            # Convert to list of [1, F, H, W] for vace_encode_masks
            input_masks_data = input_masks_data.to(dtype=vae_dtype)
            input_masks_list = [input_masks_data[b] for b in range(batch_size)]

        # Load and prepare reference images if provided (for combined guidance)
        ref_images = None
        prepared_refs = None
        if has_ref_images:
            from ..utils.encoding import load_and_prepare_reference_images

            prepared_refs = load_and_prepare_reference_images(
                ref_image_paths,
                block_state.height,
                block_state.width,
                components.config.device,
            )
            # Wrap in list for batch dimension
            ref_images = [prepared_refs]

        # Lazily initialize encoder caches on first use (need VAE to create them)
        # These caches persist across chunks for temporal continuity
        # WanVAE.create_encoder_cache() returns None (no MemBlock issue)
        # TAEWrapper.create_encoder_cache() returns TAEEncoderCache
        if not self._caches_initialized:
            self._inactive_cache = vae.create_encoder_cache()
            self._reactive_cache = vae.create_encoder_cache()
            self._caches_initialized = True

        # Standard VACE encoding path (matching wan_vace.py lines 339-341)
        # z0 = vace_encode_frames(vace_input_frames, vace_ref_images, masks=vace_input_masks)
        # When masks are provided, set pad_to_96=False because mask encoding (64 channels) will be added later
        # Pass explicit caches to prevent TAE MemBlock memory pollution between inactive/reactive streams
        z0 = vace_encode_frames(
            vae,
            input_frames_list,
            ref_images,
            masks=input_masks_list,
            pad_to_96=False,
            inactive_cache=self._inactive_cache,
            reactive_cache=self._reactive_cache,
        )

        # m0 = vace_encode_masks(input_masks, ref_images)
        m0 = vace_encode_masks(input_masks_list, ref_images)

        # z = vace_latent(z0, m0)
        z = vace_latent(z0, m0)

        return z, prepared_refs
