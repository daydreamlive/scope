"""
VACE Encoding Block for Conditioning Video Generation.

This block handles encoding of VACE conditioning inputs and prepares vace_context for
the denoising block. Supports flexible combinations:
- Reference images only (R2V): Static reference images for style/character consistency
- Conditioning input only: Per-chunk guidance (depth, flow, pose, scribble, etc.)
- Both combined: Reference images + conditioning input for style + structural guidance

The mode is implicit based on what inputs are provided - no explicit mode parameter needed.

For conditioning inputs (depth, flow, etc.), follows original VACE architecture:
- input_frames = conditioning maps (3-channel RGB from annotators)
- input_masks = spatial control masks (ones for full-frame, regional for masked areas)
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

from ..vace_utils import load_and_prepare_reference_images

logger = logging.getLogger(__name__)


class VaceEncodingBlock(ModularPipelineBlocks):
    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("vae", torch.nn.Module),
            ComponentSpec("vace_vae", torch.nn.Module, required=False),
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
                "ref_images",
                default=None,
                description="List of reference image paths for style/character consistency (can be combined with input_frames)",
            ),
            InputParam(
                "input_frames",
                default=None,
                description="VACE conditioning input frames [B, C, F, H, W]: depth, flow, pose, scribble maps, etc. (12 frames per chunk, can be combined with ref_images)",
            ),
            InputParam(
                "input_masks",
                default=None,
                description="Spatial control masks [B, 1, F, H, W]: defines WHERE to apply conditioning (white=generate, black=preserve). Defaults to ones (all white) when None. Works with any input_frames type.",
            ),
            InputParam(
                "use_dummy_frames",
                default=False,
                type_hint=bool,
                description="For ref_images only mode: whether to use dummy (zero) frames as temporal placeholder. If False, only ref images are encoded. Including them results in different behavior per original VACE implementation.",
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

        ref_images = block_state.ref_images
        input_frames = block_state.input_frames
        current_start = block_state.current_start_frame

        # If neither input is provided, skip VACE conditioning
        if (ref_images is None or len(ref_images) == 0) and input_frames is None:
            block_state.vace_context = None
            block_state.vace_ref_images = None
            self.set_block_state(state, block_state)
            return components, state

        # Determine encoding path based on what's provided (implicit mode detection)
        has_ref_images = ref_images is not None and len(ref_images) > 0
        has_input_frames = input_frames is not None

        if has_input_frames:
            # Standard VACE path: conditioning input (depth, flow, pose, etc.)
            # with optional reference images
            logger.info(
                f"VaceEncodingBlock.__call__: Using standard VACE path with conditioning input "
                f"(has_ref_images={has_ref_images}, chunk_start={current_start})"
            )
            block_state.vace_context, block_state.vace_ref_images = (
                self._encode_with_conditioning(components, block_state, current_start)
            )
        elif has_ref_images:
            # Reference images only mode (R2V)
            # Encode reference images whenever they are provided
            logger.info(
                f"VaceEncodingBlock.__call__: Encoding reference images "
                f"({len(ref_images)} ref images, chunk_start={current_start})"
            )
            block_state.vace_context, block_state.vace_ref_images = (
                self._encode_reference_only(components, block_state, current_start)
            )

        self.set_block_state(state, block_state)
        return components, state

    def _encode_reference_only(self, components, block_state, current_start):
        """
        Encode reference images only (R2V mode) following original VACE pattern.

        R2V characteristics (from notes/VACE/vace/models/wan/wan_vace.py lines 127-155):
        - Static reference images (1-3 frames)
        - Encoded fresh each chunk (application layer manages reuse)
        - Optionally uses dummy frames as 'frames' parameter to vace_encode_frames()
        - Refs passed as 'ref_images' parameter (prepended temporally inside vace_encode_frames)
        - masks=None (no masking path for R2V)
        - Uses vace_in_dim=96 (padded inside vace_encode_frames with pad_to_96=True)
        """
        ref_image_paths = block_state.ref_images
        if ref_image_paths is None or len(ref_image_paths) == 0:
            logger.warning(
                "VaceEncodingBlock._encode_reference_only: No ref_images provided"
            )
            return None, None

        use_dummy_frames = block_state.use_dummy_frames

        logger.info(
            f"VaceEncodingBlock._encode_reference_only: Encoding {len(ref_image_paths)} reference images "
            f"(use_dummy_frames={use_dummy_frames}, chunk_start={current_start})"
        )

        # Load and prepare reference images
        prepared_refs = load_and_prepare_reference_images(
            ref_image_paths,
            block_state.height,
            block_state.width,
            components.config.device,
        )

        # Use dedicated vace_vae if available, otherwise use main VAE
        vace_vae = getattr(components, "vace_vae", None)
        if vace_vae is None:
            logger.warning(
                "VaceEncodingBlock._encode_reference_only: vace_vae not found, using main VAE (may affect autoregressive cache)"
            )
            vace_vae = components.vae

        # Import vace_utils for R2V encoding path
        from ..vace_utils import vace_encode_frames

        # Calculate number of frames for dummy input (should match chunk size)
        num_frames = (
            components.config.num_frame_per_block
            * components.config.vae_temporal_downsample_factor
        )

        # Create dummy frames or encode refs-only based on use_dummy_frames flag
        if use_dummy_frames:
            # Original VACE pattern: dummy frames as temporal placeholder
            # Following notes/VACE/vace/models/wan/wan_vace.py lines 339-341
            dummy_frames = [
                torch.zeros(
                    (3, num_frames, block_state.height, block_state.width),
                    device=components.config.device,
                    dtype=next(vace_vae.parameters()).dtype,
                )
            ]
            logger.info(
                f"VaceEncodingBlock._encode_reference_only: Created dummy_frames with shape {dummy_frames[0].shape}"
            )

            # Encode with ref_images passed as parameter (R2V path)
            # Original VACE pattern: vace_encode_frames(frames, ref_images, masks=None)
            # - dummy_frames as 'frames' (full chunk)
            # - prepared_refs as 'ref_images' (wrapped in list for batch dimension)
            # - masks=None (no masking for R2V)
            # - pad_to_96=True (pads to 96 channels for VACE compatibility)
            vace_context = vace_encode_frames(
                vace_vae,
                dummy_frames,
                ref_images=[prepared_refs],
                masks=None,
                pad_to_96=True,
            )
        else:
            # Alternative: Encode only reference images (no dummy frames at all)
            # This prevents reference images from appearing in the output video
            logger.info(
                "VaceEncodingBlock._encode_reference_only: Encoding reference images only (no dummy frames)"
            )

            # Stack refs: list of [C, 1, H, W] -> [1, C, num_refs, H, W]
            prepared_refs_stacked = torch.cat(prepared_refs, dim=1).unsqueeze(0)
            # Convert to VAE's dtype (typically bfloat16)
            vae_dtype = next(vace_vae.parameters()).dtype
            prepared_refs_stacked = prepared_refs_stacked.to(dtype=vae_dtype)
            ref_latents_out = vace_vae.encode_to_latent(
                prepared_refs_stacked, use_cache=False
            )

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

            logger.info(
                f"VaceEncodingBlock._encode_reference_only: Encoded {len(prepared_refs)} reference images only, "
                f"shape={ref_latent_batch.shape}"
            )

        logger.info(
            f"VaceEncodingBlock._encode_reference_only: Encoded R2V context, "
            f"list length={len(vace_context)}, first shape={vace_context[0].shape}"
        )

        return vace_context, prepared_refs

    def _encode_with_conditioning(self, components, block_state, current_start):
        """
        Encode VACE input using the standard VACE path, with optional reference images.

        Supports any type of conditioning input (depth, flow, pose, scribble, etc.) following
        original VACE approach from notes/VACE/vace/models/wan/wan_vace.py:
        - input_frames = conditioning maps (3-channel RGB from annotators)
        - input_masks = spatial control masks (defaults to ones if None)
        - ref_images = optional (for combined style + structural guidance)
        - Standard encoding: z0 = vace_encode_frames(input_frames, ref_images, masks=input_masks)
                           m0 = vace_encode_masks(input_masks, ref_images)
                           z = vace_latent(z0, m0)

        Characteristics:
        - Per-chunk conditioning (12 frames matching output chunk)
        - Encoded every chunk (no caching)
        - Direct temporal correspondence with output
        - Uses standard VACE path with masking (produces 96 channels: 32 masked + 64 mask_encoding)
        - Can be combined with reference images for style + structure guidance
        """
        input_frames_data = block_state.input_frames
        if input_frames_data is None:
            raise ValueError(
                f"VaceEncodingBlock._encode_with_conditioning: input_frames required at chunk {current_start}"
            )

        # Validate input_frames shape
        if input_frames_data.dim() != 5:
            raise ValueError(
                f"VaceEncodingBlock._encode_with_conditioning: input_frames must be [B, C, F, H, W], got shape {input_frames_data.shape}"
            )

        batch_size, channels, num_frames, height, width = input_frames_data.shape

        # Validate frame count: should be 12 (output frames per chunk)
        expected_output_frames = (
            components.config.num_frame_per_block
            * components.config.vae_temporal_downsample_factor
        )
        if num_frames != expected_output_frames:
            raise ValueError(
                f"VaceEncodingBlock._encode_with_conditioning: Expected {expected_output_frames} frames "
                f"(num_frame_per_block={components.config.num_frame_per_block} * "
                f"vae_temporal_downsample_factor={components.config.vae_temporal_downsample_factor}), "
                f"got {num_frames} frames at chunk {current_start}"
            )

        # Validate resolution
        if height != block_state.height or width != block_state.width:
            raise ValueError(
                f"VaceEncodingBlock._encode_with_conditioning: Input resolution {height}x{width} "
                f"does not match target resolution {block_state.height}x{block_state.width}"
            )

        # Check if we have reference images too (for combined guidance)
        ref_image_paths = block_state.ref_images
        has_ref_images = ref_image_paths is not None and len(ref_image_paths) > 0

        logger.info(
            f"VaceEncodingBlock._encode_with_conditioning: Encoding {num_frames} conditioning frames "
            f"(with_ref_images={has_ref_images}, chunk {current_start})"
        )

        # Use dedicated vace_vae if available, otherwise use main VAE
        vace_vae = getattr(components, "vace_vae", None)
        if vace_vae is None:
            logger.warning(
                "VaceEncodingBlock._encode_with_conditioning: vace_vae not found, using main VAE (may affect autoregressive cache)"
            )
            vace_vae = components.vae

        # Import vace_utils for standard encoding path
        from ..vace_utils import vace_encode_frames, vace_encode_masks, vace_latent

        # Ensure 3-channel input for VAE (conditioning maps should already be 3-channel RGB)
        if channels == 1:
            logger.info(
                "VaceEncodingBlock._encode_with_conditioning: Converting 1-channel input to 3-channel RGB"
            )
            input_frames_data = input_frames_data.repeat(1, 3, 1, 1, 1)
        elif channels != 3:
            raise ValueError(
                f"VaceEncodingBlock._encode_with_conditioning: Expected 1 or 3 channels, got {channels}"
            )

        vae_dtype = next(vace_vae.parameters()).dtype
        input_frames_data = input_frames_data.to(dtype=vae_dtype)

        # Convert to list of [C, F, H, W] for vace_encode_frames
        input_frames_list = [input_frames_data[b] for b in range(batch_size)]

        # Get input_masks from block_state or default to ones (all white)
        input_masks_data = block_state.input_masks
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
            logger.info(
                "VaceEncodingBlock._encode_with_conditioning: input_masks not provided, defaulting to ones (all white)"
            )
        else:
            # Validate input_masks shape
            if input_masks_data.dim() != 5:
                raise ValueError(
                    f"VaceEncodingBlock._encode_with_conditioning: input_masks must be [B, 1, F, H, W], got shape {input_masks_data.shape}"
                )

            mask_batch, mask_channels, mask_frames, mask_height, mask_width = (
                input_masks_data.shape
            )
            if mask_channels != 1:
                raise ValueError(
                    f"VaceEncodingBlock._encode_with_conditioning: input_masks must have 1 channel, got {mask_channels}"
                )
            if (
                mask_frames != num_frames
                or mask_height != height
                or mask_width != width
            ):
                raise ValueError(
                    f"VaceEncodingBlock._encode_with_conditioning: input_masks shape mismatch: "
                    f"expected [B, 1, {num_frames}, {height}, {width}], got [B, 1, {mask_frames}, {mask_height}, {mask_width}]"
                )

            # Convert to list of [1, F, H, W] for vace_encode_masks
            input_masks_data = input_masks_data.to(dtype=vae_dtype)
            input_masks_list = [input_masks_data[b] for b in range(batch_size)]
            logger.info(
                "VaceEncodingBlock._encode_with_conditioning: Using provided input_masks"
            )

        # Load and prepare reference images if provided (for combined guidance)
        ref_images = None
        prepared_refs = None
        if has_ref_images:
            from ..vace_utils import load_and_prepare_reference_images

            prepared_refs = load_and_prepare_reference_images(
                ref_image_paths,
                block_state.height,
                block_state.width,
                components.config.device,
            )
            # Wrap in list for batch dimension
            ref_images = [prepared_refs]
            logger.info(
                f"VaceEncodingBlock._encode_with_conditioning: Loaded {len(ref_image_paths)} reference images for combined guidance"
            )

        # Standard VACE encoding path (matching wan_vace.py lines 339-341)
        # z0 = vace_encode_frames(input_frames, ref_images, masks=input_masks)
        # When masks are provided, set pad_to_96=False because mask encoding (64 channels) will be added later
        z0 = vace_encode_frames(
            vace_vae,
            input_frames_list,
            ref_images,
            masks=input_masks_list,
            pad_to_96=False,
        )

        # m0 = vace_encode_masks(input_masks, ref_images)
        m0 = vace_encode_masks(input_masks_list, ref_images)

        # z = vace_latent(z0, m0)
        z = vace_latent(z0, m0)

        # Validate latent frame count
        expected_latent_frames = components.config.num_frame_per_block
        actual_latent_frames = z[0].shape[1]
        if actual_latent_frames != expected_latent_frames:
            raise ValueError(
                f"VaceEncodingBlock._encode_with_conditioning: Expected {expected_latent_frames} latent frames, "
                f"got {actual_latent_frames} after VAE encoding"
            )

        logger.info(
            f"VaceEncodingBlock._encode_with_conditioning: Encoded VACE context via standard path, "
            f"latent shape={z[0].shape} "
            f"({expected_latent_frames} latent frames from {num_frames} output frames)"
        )

        return z, prepared_refs
