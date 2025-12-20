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

from ..utils.encoding import (
    load_and_prepare_reference_images,
    vace_encode_frames,
    vace_encode_masks,
    vace_latent,
)

logger = logging.getLogger(__name__)


class VaceEncodingBlock(ModularPipelineBlocks):
    """
    VACE encoding block with internal routing logic.

    Architectural Note: This block does NOT use AutoPipelineBlock pattern despite having
    multiple execution paths. Rationale:

    1. Single Operation: All paths perform the same conceptual operation (VACE encoding)
       with shared logic (VAE selection, reference image loading, validation).

    2. OR Condition: Block should run if EITHER ref_images OR input_frames is provided.
       AutoPipelineBlock uses sequential first-match logic, making OR conditions awkward
       (would require duplicate block instances with different triggers).

    3. Simple Skip Logic: The skip condition is trivial (3 lines in __call__) and
       self-contained. Using AutoPipelineBlock would add complexity without benefit.

    Compare to AutoPrepareLatentsBlock: That routes between fundamentally different
    operations (prepare fresh latents vs. encode video) with a single trigger (video).
    VACE has one operation with multiple potential triggers, better handled internally.
    """

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
                "extension_mode",
                default=None,
                type_hint=str,
                description="Extension mode for temporal generation: 'firstframe' (ref at start, generate after), 'lastframe' (generate before, ref at end), or 'firstlastframe' (refs at both ends). Applies to specific chunks based on current_start_frame.",
            ),
            InputParam(
                "first_frame_image",
                default=None,
                description="Path to first frame reference image for extension mode (used with 'firstframe' or 'firstlastframe')",
            ),
            InputParam(
                "last_frame_image",
                default=None,
                description="Path to last frame reference image for extension mode (used with 'lastframe' or 'firstlastframe')",
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
        extension_mode = block_state.extension_mode
        first_frame_image = block_state.first_frame_image
        last_frame_image = block_state.last_frame_image
        current_start = block_state.current_start_frame

        # If no inputs provided, skip VACE conditioning
        has_ref_images = ref_images is not None and len(ref_images) > 0
        has_input_frames = input_frames is not None
        has_extension_mode = extension_mode is not None

        if not has_ref_images and input_frames is None and not has_extension_mode:
            block_state.vace_context = None
            block_state.vace_ref_images = None
            self.set_block_state(state, block_state)
            return components, state

        # Determine encoding path based on what's provided (implicit mode detection)
        if has_extension_mode:
            # Extension mode: Generate frames before/after reference frame(s)
            # Uses explicit first_frame_image and last_frame_image parameters
            # This is different from R2V - reference frames appear in output video
            if extension_mode == "firstframe":
                if first_frame_image is None:
                    raise ValueError(
                        "VaceEncodingBlock: extension_mode 'firstframe' requires first_frame_image"
                    )
            elif extension_mode == "lastframe":
                if last_frame_image is None:
                    raise ValueError(
                        "VaceEncodingBlock: extension_mode 'lastframe' requires last_frame_image"
                    )
            elif extension_mode == "firstlastframe":
                if first_frame_image is None or last_frame_image is None:
                    raise ValueError(
                        "VaceEncodingBlock: extension_mode 'firstlastframe' requires both first_frame_image and last_frame_image"
                    )
            block_state.vace_context, block_state.vace_ref_images = (
                self._encode_extension_mode(components, block_state, current_start)
            )
        elif has_input_frames:
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

        # Load and prepare reference images
        prepared_refs = load_and_prepare_reference_images(
            ref_image_paths,
            block_state.height,
            block_state.width,
            components.config.device,
        )

        # CRITICAL: Use vace_vae for encoding reference images.
        # Reference images (1-2 frames) are too small for streaming VAE encoders that
        # use 3D convolutions with temporal caching. Pipelines with streaming VAEs
        # MUST provide a separate standard WanVAE as vace_vae for VACE encoding.
        if hasattr(components, "vace_vae") and components.vace_vae is not None:
            vace_vae = components.vace_vae
        else:
            # Fallback to main VAE only for pipelines with standard (non-streaming) VAEs
            # This will fail for pipelines with streaming VAEs
            vace_vae = components.vae
            logger.warning(
                "VaceEncodingBlock._encode_reference_only: Using main VAE for VACE context encoding. "
                "If the main VAE is a streaming VAE, this will fail with temporal dimension errors."
            )

        # Import vace_utils for R2V encoding path
        from ..utils.encoding import vace_encode_frames

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

            # Stack refs: list of [C, 1, H, W] -> [1, C, num_refs, H, W]
            prepared_refs_stacked = torch.cat(prepared_refs, dim=1).unsqueeze(0)
            # Convert to VAE's dtype (typically bfloat16)
            vae_dtype = next(vace_vae.parameters()).dtype
            prepared_refs_stacked = prepared_refs_stacked.to(dtype=vae_dtype)
            # Reference images are static, so cache doesn't matter, but use False to avoid affecting video cache
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

        return vace_context, prepared_refs

    def _encode_extension_mode(self, components, block_state, current_start):
        """
        Encode extension mode for chunk-based autoregressive temporal generation.

        Extension mode characteristics (from notes/VACE/vace/annotators/frameref.py lines 70-119):
        - Reference frames appear in output video (unlike R2V where they only condition)
        - Gray/dummy frames (zeros in normalized [-1,1] space) fill non-reference positions
        - Masks indicate which frames to inpaint: 0 for reference frames (keep), 1 for dummy frames (inpaint)
        - Modes:
          * firstframe: first_frame_image at chunk 0, dummy frames for rest of chunk
          * lastframe: Dummy frames in final chunk, last_frame_image at end
          * firstlastframe: first_frame_image in chunk 0, last_frame_image at end of final chunk

        Architecture notes:
        - Autoregressive generation works in fixed chunks (e.g., 12 frames per chunk)
        - Extension mode determines which chunks contain reference frames
        - Application layer decides when to apply extension mode based on current_start_frame
        - Dummy frames are torch.zeros() = gray (127.5/255 → 0.0 in normalized space)
        - Uses explicit first_frame_image and last_frame_image parameters (not ref_images)
        - CRITICAL: Masks must be encoded via vace_encode_frames + vace_encode_masks + vace_latent
          to produce correct 96-channel VACE context (32 channels masked encoding + 64 channels mask encoding)

        This is different from R2V mode:
        - R2V: Reference images (ref_images) condition entire video, don't appear as frames
        - Extension: Reference frames ARE output frames, generate continuation before/after
        """
        extension_mode = block_state.extension_mode
        first_frame_image = block_state.first_frame_image
        last_frame_image = block_state.last_frame_image

        if extension_mode not in ["firstframe", "lastframe", "firstlastframe"]:
            raise ValueError(
                f"VaceEncodingBlock._encode_extension_mode: Invalid extension_mode '{extension_mode}', "
                f"must be 'firstframe', 'lastframe', or 'firstlastframe'"
            )

        # Determine which image(s) to load based on mode and current chunk
        images_to_load = []
        if extension_mode == "firstframe":
            images_to_load = [first_frame_image]
        elif extension_mode == "lastframe":
            images_to_load = [last_frame_image]
        elif extension_mode == "firstlastframe":
            # Use first_frame_image for first chunk, last_frame_image for last chunk
            if current_start == 0:
                images_to_load = [first_frame_image]
            else:
                images_to_load = [last_frame_image]

        # Load and prepare reference images
        print(f"_encode_extension_mode: Loading images: {images_to_load}")
        prepared_refs = load_and_prepare_reference_images(
            images_to_load,
            block_state.height,
            block_state.width,
            components.config.device,
        )
        print(f"_encode_extension_mode: Loaded {len(prepared_refs)} reference images")
        print(
            f"_encode_extension_mode: prepared_refs[0] shape={prepared_refs[0].shape}, dtype={prepared_refs[0].dtype}, device={prepared_refs[0].device}"
        )
        print(
            f"_encode_extension_mode: prepared_refs[0] value range=[{prepared_refs[0].min():.3f}, {prepared_refs[0].max():.3f}]"
        )

        # Use vace_vae for encoding if available, otherwise use main VAE
        # Extension mode encodes with use_cache=False, so streaming VAEs work fine
        if hasattr(components, "vace_vae") and components.vace_vae is not None:
            vace_vae = components.vace_vae
        else:
            vace_vae = components.vae
            logger.info(
                "VaceEncodingBlock._encode_extension_mode: Using main VAE for VACE encoding (no separate vace_vae configured)"
            )

        vae_dtype = next(vace_vae.parameters()).dtype

        # Calculate number of frames for this chunk at LATENT resolution
        # IMPORTANT: VACE context must match the temporal structure of the LATENTS, not decoded video
        # The VAE will encode these frames down to latent resolution
        # For LongLive: num_frame_per_block=3 latent frames -> 12 pixel frames (4x temporal upsample)
        # But VACE context encodes at PIXEL resolution then VAE compresses to latent resolution
        # So we need 12 pixel frames here, which will become 3 latent frames after VAE encoding
        num_frames = (
            components.config.num_frame_per_block
            * components.config.vae_temporal_downsample_factor
        )

        # Build frames and masks for this chunk based on extension mode
        # Following notes/VACE/vace/annotators/frameref.py:
        # - Reference frames: mask = 0 (zeros = keep frame)
        # - Dummy frames: mask = 1 (ones = inpaint this area)
        frames_to_encode = []
        masks_to_encode = []

        if extension_mode == "firstframe" or (
            extension_mode == "firstlastframe" and current_start == 0
        ):
            # Reference frames first, then dummy frames
            # CRITICAL FIX: Replicate reference to fill entire temporal VAE group
            # For 12 frames with 4:1 compression: frames 0-3 -> latent frame 0
            # Replicating ref across frames 0-3 prevents dilution with dummy frames
            temporal_group_size = components.config.vae_temporal_downsample_factor
            num_ref_frames = temporal_group_size  # Replicate ref to fill temporal group
            num_dummy_frames = num_frames - num_ref_frames
            print(
                f"_encode_extension_mode: Building firstframe pattern: [{num_ref_frames} replicated refs, {num_dummy_frames} dummies]"
            )

            # Build frames: [ref, ref, ref, zeros, zeros, ...]
            # Replicate reference across temporal group to prevent dilution
            ref_replicated = prepared_refs[0].repeat(
                1, num_ref_frames, 1, 1
            )  # [C, 1, H, W] -> [C, num_ref_frames, H, W]
            ref_with_dummies = torch.cat(
                [
                    ref_replicated,  # Replicated reference frames
                    torch.zeros(
                        (3, num_dummy_frames, block_state.height, block_state.width),
                        device=components.config.device,
                        dtype=vae_dtype,
                    ),
                ],
                dim=1,
            )
            frames_to_encode.append(ref_with_dummies)

            # Build masks: [0, 0, 0, 1, 1, ...] (0 = keep refs, 1 = inpaint dummies)
            ref_masks = torch.zeros(
                (1, num_ref_frames, block_state.height, block_state.width),
                device=components.config.device,
                dtype=torch.float32,
            )
            dummy_masks = torch.ones(
                (1, num_dummy_frames, block_state.height, block_state.width),
                device=components.config.device,
                dtype=torch.float32,
            )
            mask = torch.cat([ref_masks, dummy_masks], dim=1)
            masks_to_encode.append(mask)

            print(
                f"_encode_extension_mode: ref_with_dummies shape={ref_with_dummies.shape}"
            )
            print(f"_encode_extension_mode: mask shape={mask.shape}")
            print(
                f"_encode_extension_mode: Frame 0 (first ref) value range=[{ref_with_dummies[:, 0].min():.3f}, {ref_with_dummies[:, 0].max():.3f}]"
            )
            print(
                f"_encode_extension_mode: Frame {num_ref_frames-1} (last ref) value range=[{ref_with_dummies[:, num_ref_frames-1].min():.3f}, {ref_with_dummies[:, num_ref_frames-1].max():.3f}]"
            )
            print(
                f"_encode_extension_mode: Frame {num_ref_frames} (first dummy) value range=[{ref_with_dummies[:, num_ref_frames].min():.3f}, {ref_with_dummies[:, num_ref_frames].max():.3f}]"
            )
            print(
                f"_encode_extension_mode: Mask frame 0 (first ref) = {mask[0, 0].mean():.3f} (should be 0)"
            )
            print(
                f"_encode_extension_mode: Mask frame {num_ref_frames-1} (last ref) = {mask[0, num_ref_frames-1].mean():.3f} (should be 0)"
            )
            print(
                f"_encode_extension_mode: Mask frame {num_ref_frames} (first dummy) = {mask[0, num_ref_frames].mean():.3f} (should be 1)"
            )

        elif extension_mode == "lastframe" or (
            extension_mode == "firstlastframe" and current_start != 0
        ):
            # Dummy frames first, then reference frames
            # CRITICAL FIX: Replicate reference to fill entire temporal VAE group
            # For 12 frames with 4:1 compression: frames 9-11 -> latent frame 2
            # Replicating ref across frames 9-11 prevents dilution with dummy frames
            temporal_group_size = components.config.vae_temporal_downsample_factor
            num_ref_frames = temporal_group_size  # Replicate ref to fill temporal group
            num_dummy_frames = num_frames - num_ref_frames
            print(
                f"_encode_extension_mode: Building lastframe pattern: [{num_dummy_frames} dummies, {num_ref_frames} replicated refs]"
            )

            # Build frames: [zeros, zeros, ..., ref, ref, ref]
            # Replicate reference across temporal group to prevent dilution
            ref_replicated = prepared_refs[0].repeat(
                1, num_ref_frames, 1, 1
            )  # [C, 1, H, W] -> [C, num_ref_frames, H, W]
            dummy_with_ref = torch.cat(
                [
                    torch.zeros(
                        (3, num_dummy_frames, block_state.height, block_state.width),
                        device=components.config.device,
                        dtype=vae_dtype,
                    ),
                    ref_replicated,  # Replicated reference frames
                ],
                dim=1,
            )
            frames_to_encode.append(dummy_with_ref)

            # Build masks: [1, 1, ..., 0, 0, 0] (1 = inpaint dummies, 0 = keep refs)
            dummy_masks = torch.ones(
                (1, num_dummy_frames, block_state.height, block_state.width),
                device=components.config.device,
                dtype=torch.float32,
            )
            ref_masks = torch.zeros(
                (1, num_ref_frames, block_state.height, block_state.width),
                device=components.config.device,
                dtype=torch.float32,
            )
            mask = torch.cat([dummy_masks, ref_masks], dim=1)
            masks_to_encode.append(mask)

            print(
                f"_encode_extension_mode: dummy_with_ref shape={dummy_with_ref.shape}"
            )
            print(f"_encode_extension_mode: mask shape={mask.shape}")
            print(
                f"_encode_extension_mode: Frame 0 (dummy) value range=[{dummy_with_ref[:, 0].min():.3f}, {dummy_with_ref[:, 0].max():.3f}]"
            )
            print(
                f"_encode_extension_mode: Frame {num_dummy_frames} (first ref) value range=[{dummy_with_ref[:, num_dummy_frames].min():.3f}, {dummy_with_ref[:, num_dummy_frames].max():.3f}]"
            )
            print(
                f"_encode_extension_mode: Frame {num_frames-1} (last ref) value range=[{dummy_with_ref[:, num_frames-1].min():.3f}, {dummy_with_ref[:, num_frames-1].max():.3f}]"
            )
            print(
                f"_encode_extension_mode: Mask frame 0 (dummy) = {mask[0, 0].mean():.3f} (should be 1)"
            )
            print(
                f"_encode_extension_mode: Mask frame {num_dummy_frames} (first ref) = {mask[0, num_dummy_frames].mean():.3f} (should be 0)"
            )
            print(
                f"_encode_extension_mode: Mask frame {num_frames-1} (last ref) = {mask[0, num_frames-1].mean():.3f} (should be 0)"
            )

        # Now encode using the full VACE pipeline with masks
        # Following notes/VACE/vace/models/wan/wan_vace.py lines 339-341:
        # z0 = self.vace_encode_frames(input_frames, input_ref_images, masks=input_masks)
        # m0 = self.vace_encode_masks(input_masks, input_ref_images)
        # z = self.vace_latent(z0, m0)

        print(
            "_encode_extension_mode: Encoding frames with masks via vace_encode_frames..."
        )
        # vace_encode_frames expects list of [C, F, H, W] frames
        # We have frames as [C, F, H, W] already, so just use as list
        z0 = vace_encode_frames(
            vae=vace_vae,
            frames=frames_to_encode,
            ref_images=[
                None
            ],  # No separate ref_images in extension mode (refs are IN the frames)
            masks=masks_to_encode,
            pad_to_96=False,  # Don't pad yet, mask encoding will be added
            use_cache=False,  # Extension mode is per-chunk, don't use streaming cache
        )
        print(
            f"_encode_extension_mode: z0[0] shape={z0[0].shape} (should be ~32 channels: 16 inactive + 16 reactive)"
        )
        print(
            f"_encode_extension_mode: z0[0] value range=[{z0[0].min():.3f}, {z0[0].max():.3f}]"
        )
        print(f"_encode_extension_mode: z0[0] mean={z0[0].mean():.3f}")

        print("_encode_extension_mode: Encoding masks via vace_encode_masks...")
        # Get VAE stride from config
        vae_stride = (
            components.config.vae_temporal_downsample_factor,
            components.config.vae_spatial_downsample_factor,
            components.config.vae_spatial_downsample_factor,
        )
        m0 = vace_encode_masks(
            masks=masks_to_encode,
            ref_images=[None],  # No separate ref_images
            vae_stride=vae_stride,
        )
        print(
            f"_encode_extension_mode: m0[0] shape={m0[0].shape} (should be 64 channels)"
        )
        print(
            f"_encode_extension_mode: m0[0] value range=[{m0[0].min():.3f}, {m0[0].max():.3f}]"
        )
        print(f"_encode_extension_mode: m0[0] mean={m0[0].mean():.3f}")

        # Diagnostic: Check mask encoding for reference vs dummy regions
        # For lastframe: should have different characteristics in latent frames 0-1 (dummies) vs frame 2 (refs)
        if m0[0].shape[1] >= 3:
            print(
                f"_encode_extension_mode: m0[0] latent frame 0 (dummy region) mean={m0[0][:, 0].mean():.3f}"
            )
            print(
                f"_encode_extension_mode: m0[0] latent frame 2 (ref region) mean={m0[0][:, 2].mean():.3f}"
            )

        print("_encode_extension_mode: Concatenating via vace_latent...")
        vace_context = vace_latent(z0, m0)
        print(
            f"_encode_extension_mode: vace_context[0] shape={vace_context[0].shape} (should be 96 channels)"
        )

        logger.info(
            f"_encode_extension_mode: mode={extension_mode}, current_start={current_start}, "
            f"num_frames={num_frames}, vace_context_shape={vace_context[0].shape}"
        )
        print(
            f"_encode_extension_mode: Final vace_context[0] shape={vace_context[0].shape}"
        )
        print(
            f"_encode_extension_mode: Final vace_context[0] value range=[{vace_context[0].min():.3f}, {vace_context[0].max():.3f}]"
        )
        print(
            "_encode_extension_mode: COMPLETE - Returning vace_context and prepared_refs"
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

        # Use vace_vae if available (for pipelines with streaming VAEs that can't handle
        # conditioning frame batches), otherwise use main VAE for consistency with video encoding.
        # Main VAE is preferred when available since it ensures temporal consistency in inpainting
        # mode where unmasked portions are encoded in both VACE context and video latents.
        if hasattr(components, "vace_vae") and components.vace_vae is not None:
            vace_vae = components.vace_vae
        else:
            vace_vae = components.vae

        # Import vace_utils for standard encoding path
        from ..utils.encoding import vace_encode_frames, vace_encode_masks, vace_latent

        # Ensure 3-channel input for VAE (conditioning maps should already be 3-channel RGB)
        if channels == 1:
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
        # When reference images are combined with conditioning, they are concatenated
        # along the frame dimension, so we need to account for them
        expected_latent_frames = components.config.num_frame_per_block
        if has_ref_images:
            # Reference images are concatenated to frame latents in vace_encode_frames
            # Each reference image adds 1 latent frame
            expected_latent_frames += len(ref_image_paths)
        actual_latent_frames = z[0].shape[1]
        if actual_latent_frames != expected_latent_frames:
            raise ValueError(
                f"VaceEncodingBlock._encode_with_conditioning: Expected {expected_latent_frames} latent frames "
                f"({components.config.num_frame_per_block} from conditioning frames"
                f"{f' + {len(ref_image_paths)} from reference images' if has_ref_images else ''}), "
                f"got {actual_latent_frames} after VAE encoding"
            )

        return z, prepared_refs
