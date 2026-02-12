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
            ConfigSpec("vae_spatial_downsample_factor", 8),  # Used by vae_stride tuple
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
                "first_frame_image",
                default=None,
                description="Path to first frame reference image for extension mode. When provided alone, enables 'firstframe' mode (ref at start, generate after). When provided with last_frame_image, enables 'firstlastframe' mode (refs at both ends).",
            ),
            InputParam(
                "last_frame_image",
                default=None,
                description="Path to last frame reference image for extension mode. When provided alone, enables 'lastframe' mode (generate before, ref at end). When provided with first_frame_image, enables 'firstlastframe' mode (refs at both ends).",
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
        first_frame_image: str | None = block_state.first_frame_image
        last_frame_image: str | None = block_state.last_frame_image
        current_start = block_state.current_start_frame

        # If no inputs provided, skip VACE conditioning
        has_ref_images = vace_ref_images is not None and len(vace_ref_images) > 0
        has_input_frames = vace_input_frames is not None
        has_first_frame = first_frame_image is not None
        has_last_frame = last_frame_image is not None
        has_extension = has_first_frame or has_last_frame

        if not has_ref_images and not has_input_frames and not has_extension:
            block_state.vace_context = None
            block_state.vace_ref_images = None
            self.set_block_state(state, block_state)
            return components, state

        # Determine encoding path based on what's provided (implicit mode detection)
        if has_extension:
            # Extension mode: Generate frames before/after reference frame(s)
            # Mode is inferred from which frame images are provided
            if has_first_frame and has_last_frame:
                extension_mode = "firstlastframe"
            elif has_first_frame:
                extension_mode = "firstframe"
            else:
                extension_mode = "lastframe"

            block_state.vace_context, block_state.vace_ref_images = (
                self._encode_extension_mode(
                    components, block_state, current_start, extension_mode
                )
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

        # Load and prepare reference images (spatial masks unused in R2V mode)
        prepared_refs, _ = load_and_prepare_reference_images(
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

    def _encode_extension_mode(
        self, components, block_state, current_start, extension_mode: str
    ):
        """
        Encode VACE context with reference frames and dummy frames for temporal extension.

        Loads reference image based on extension_mode (inferred from provided images),
        replicates it across a temporal group, fills remaining frames with zeros (dummy frames),
        and encodes with masks indicating which frames to inpaint (1=dummy, 0=reference).

        Supports combining with conditioning inputs (inpainting/depth):
        - If vace_input_frames provided: use those for dummy positions instead of zeros
        - If vace_input_masks provided: use those for spatial control on dummy positions

        Spatial masking: Auto-generates spatial mask based on padding detection
        (0=image region, 1=padding region). This allows the first/last frame influence
        to be regional rather than full-frame, letting the model freely generate in
        padded areas while preserving the reference image's influence in its actual region.

        Args:
            extension_mode: Inferred mode ('firstframe', 'lastframe', or 'firstlastframe')
        """
        first_frame_image = block_state.first_frame_image
        last_frame_image = block_state.last_frame_image

        # Load reference images based on mode
        if extension_mode == "firstframe":
            images_to_load = [first_frame_image]
        elif extension_mode == "lastframe":
            images_to_load = [last_frame_image]
        elif extension_mode == "firstlastframe":
            # Load BOTH images for firstlastframe mode
            images_to_load = [first_frame_image, last_frame_image]

        # Load and crop-to-fill reference images (spatial masks always zeros with crop strategy)
        prepared_refs, spatial_masks = load_and_prepare_reference_images(
            images_to_load,
            block_state.height,
            block_state.width,
            components.config.device,
        )

        vae = components.vae
        vae_dtype = next(vae.parameters()).dtype

        num_frames = (
            components.config.num_frame_per_block
            * components.config.vae_temporal_downsample_factor
        )

        # Determine ref placement
        ref_at_start = extension_mode in ("firstframe", "firstlastframe")
        ref_at_end = extension_mode in ("lastframe", "firstlastframe")

        # Check for conditioning inputs (inpainting/depth) to combine with extension
        conditioning_frames = None
        conditioning_masks = None
        if block_state.vace_input_frames is not None:
            # Extract conditioning frames for dummy positions
            input_frames = block_state.vace_input_frames
            if input_frames.dim() == 5:
                # [B, C, F, H, W] -> [C, F, H, W] (take first batch)
                input_frames = input_frames[0]
            conditioning_frames = input_frames.to(
                device=components.config.device, dtype=vae_dtype
            )

            # Also get conditioning masks if provided
            if block_state.vace_input_masks is not None:
                input_masks = block_state.vace_input_masks
                if input_masks.dim() == 5:
                    # [B, 1, F, H, W] -> [1, F, H, W] (take first batch)
                    input_masks = input_masks[0]
                conditioning_masks = input_masks.to(device=components.config.device)

        frames, masks = self._build_extension_frames_and_masks(
            prepared_refs=prepared_refs,
            num_frames=num_frames,
            temporal_group_size=components.config.vae_temporal_downsample_factor,
            ref_at_start=ref_at_start,
            ref_at_end=ref_at_end,
            device=components.config.device,
            dtype=vae_dtype,
            height=block_state.height,
            width=block_state.width,
            spatial_masks=spatial_masks,
            conditioning_frames=conditioning_frames,
            conditioning_masks=conditioning_masks,
        )

        frames_to_encode = [frames]
        masks_to_encode = [masks]

        # Check for R2V reference images to combine with extension
        ref_image_paths = block_state.vace_ref_images
        has_ref_images = ref_image_paths is not None and len(ref_image_paths) > 0
        r2v_refs = None
        if has_ref_images:
            r2v_prepared_refs, _ = load_and_prepare_reference_images(
                ref_image_paths,
                block_state.height,
                block_state.width,
                components.config.device,
            )
            r2v_refs = [r2v_prepared_refs]

        # Lazily initialize encoder caches for combined mode with conditioning
        if conditioning_frames is not None and not self._caches_initialized:
            self._inactive_cache = vae.create_encoder_cache()
            self._reactive_cache = vae.create_encoder_cache()
            self._caches_initialized = True

        z0 = vace_encode_frames(
            vae=vae,
            frames=frames_to_encode,
            ref_images=r2v_refs if r2v_refs else [None],
            masks=masks_to_encode,
            pad_to_96=False,
            use_cache=False if conditioning_frames is None else True,
            inactive_cache=self._inactive_cache
            if conditioning_frames is not None
            else None,
            reactive_cache=self._reactive_cache
            if conditioning_frames is not None
            else None,
        )

        vae_stride = (
            components.config.vae_temporal_downsample_factor,
            components.config.vae_spatial_downsample_factor,
            components.config.vae_spatial_downsample_factor,
        )
        m0 = vace_encode_masks(
            masks=masks_to_encode,
            ref_images=r2v_refs if r2v_refs else [None],
            vae_stride=vae_stride,
        )

        vace_context = vace_latent(z0, m0)

        # Log mode info
        has_padding = any((mask > 0).any() for mask in spatial_masks)
        combined_info = []
        if conditioning_frames is not None:
            combined_info.append("conditioning")
        if has_ref_images:
            combined_info.append("r2v")
        combined_str = f" + {' + '.join(combined_info)}" if combined_info else ""
        logger.info(
            f"_encode_extension_mode: mode={extension_mode}{combined_str}, current_start={current_start}, "
            f"num_frames={num_frames}, vace_context_shape={vace_context[0].shape}, "
            f"spatial_masking={'active (padding detected)' if has_padding else 'inactive (no padding)'}"
        )

        return vace_context, prepared_refs

    def _build_extension_frames_and_masks(
        self,
        prepared_refs: list[torch.Tensor],
        num_frames: int,
        temporal_group_size: int,
        ref_at_start: bool,
        ref_at_end: bool,
        device: torch.device,
        dtype: torch.dtype,
        height: int,
        width: int,
        spatial_masks: list[torch.Tensor] | None = None,
        conditioning_frames: torch.Tensor | None = None,
        conditioning_masks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build frames and masks for extension mode using a two-stage approach:

        Stage 1: Build base sequence from conditioning (or zeros/ones default)
        Stage 2: Overlay reference frames at anchor positions

        This cleanly separates "what fills the sequence" from "where anchors go".

        Args:
            prepared_refs: Reference images [C, 1, H, W]. Single for first/last, two for firstlast.
            num_frames: Total frames to generate
            temporal_group_size: Frames per VAE temporal group (typically 4)
            ref_at_start/ref_at_end: Where to place reference anchors
            spatial_masks: Optional padding masks for reference positions [1, 1, H, W]
            conditioning_frames: Optional base frames [C, F, H, W] (inpainting/depth)
            conditioning_masks: Optional base masks [1, F, H, W] (inpainting spatial control)

        Returns:
            (frames, masks): [C, F, H, W] and [1, F, H, W] tensors ready for VACE encoding
        """
        # Stage 1: Build base frames and masks for entire sequence
        frames, masks = self._build_base_sequence(
            num_frames=num_frames,
            conditioning_frames=conditioning_frames,
            conditioning_masks=conditioning_masks,
            device=device,
            dtype=dtype,
            height=height,
            width=width,
        )

        # Stage 2: Overlay references at anchor positions
        # Clone once here so _overlay_reference_at can mutate in-place
        frames = frames.clone()
        masks = masks.clone()

        first_spatial = spatial_masks[0] if spatial_masks else None
        last_spatial = (
            spatial_masks[1]
            if spatial_masks and len(spatial_masks) > 1
            else first_spatial
        )

        if ref_at_start and ref_at_end:
            # firstlastframe mode
            if num_frames < 2 * temporal_group_size:
                raise ValueError(
                    f"Not enough frames for firstlastframe: need {2 * temporal_group_size}, got {num_frames}"
                )
            frames, masks = self._overlay_reference_at(
                frames,
                masks,
                prepared_refs[0],
                first_spatial,
                start_idx=0,
                temporal_group_size=temporal_group_size,
                conditioning_masks=conditioning_masks,
                device=device,
                height=height,
                width=width,
            )
            frames, masks = self._overlay_reference_at(
                frames,
                masks,
                prepared_refs[1],
                last_spatial,
                start_idx=num_frames - temporal_group_size,
                temporal_group_size=temporal_group_size,
                conditioning_masks=conditioning_masks,
                device=device,
                height=height,
                width=width,
            )
        elif ref_at_start:
            frames, masks = self._overlay_reference_at(
                frames,
                masks,
                prepared_refs[0],
                first_spatial,
                start_idx=0,
                temporal_group_size=temporal_group_size,
                conditioning_masks=conditioning_masks,
                device=device,
                height=height,
                width=width,
            )
        elif ref_at_end:
            frames, masks = self._overlay_reference_at(
                frames,
                masks,
                prepared_refs[0],
                first_spatial,
                start_idx=num_frames - temporal_group_size,
                temporal_group_size=temporal_group_size,
                conditioning_masks=conditioning_masks,
                device=device,
                height=height,
                width=width,
            )
        else:
            raise ValueError("At least one of ref_at_start or ref_at_end must be True")

        return frames, masks

    def _build_base_sequence(
        self,
        num_frames: int,
        conditioning_frames: torch.Tensor | None,
        conditioning_masks: torch.Tensor | None,
        device: torch.device,
        dtype: torch.dtype,
        height: int,
        width: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build the base frame/mask sequence from conditioning or defaults.

        If conditioning provided: use those frames/masks (padded if needed)
        If no conditioning: zeros for frames, ones for masks (generate everything)
        """
        if conditioning_frames is not None:
            frames = self._extract_and_pad(conditioning_frames, num_frames, dtype)
            if conditioning_masks is not None:
                masks = self._extract_and_pad(
                    conditioning_masks, num_frames, torch.float32
                )
            else:
                masks = torch.ones(
                    (1, num_frames, height, width), device=device, dtype=torch.float32
                )
        else:
            frames = torch.zeros(
                (3, num_frames, height, width), device=device, dtype=dtype
            )
            masks = torch.ones(
                (1, num_frames, height, width), device=device, dtype=torch.float32
            )

        return frames, masks

    def _extract_and_pad(
        self,
        tensor: torch.Tensor,
        target_frames: int,
        target_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Extract frames from tensor, padding with last frame if needed."""
        available = tensor.shape[1]
        if available >= target_frames:
            return tensor[:, :target_frames, :, :].to(dtype=target_dtype)
        else:
            padding = target_frames - available
            pad_tensor = tensor[:, -1:, :, :].repeat(1, padding, 1, 1)
            return torch.cat([tensor, pad_tensor], dim=1).to(dtype=target_dtype)

    def _overlay_reference_at(
        self,
        frames: torch.Tensor,
        masks: torch.Tensor,
        ref_tensor: torch.Tensor,
        spatial_mask: torch.Tensor | None,
        start_idx: int,
        temporal_group_size: int,
        conditioning_masks: torch.Tensor | None,
        device: torch.device,
        height: int,
        width: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Overlay a reference frame at a specific position in the sequence.

        Two behaviors based on whether we're combining with inpainting:

        With inpainting (conditioning_masks provided):
            - Composite: reference where mask=1 (inpaint region), keep base where mask=0 (preserve)
            - Output mask: all zeros (preserve the composite entirely)

        Without inpainting:
            - Replace frames with replicated reference
            - Output mask: spatial_mask (for padding regions) or zeros (preserve all)
        """
        end_idx = start_idx + temporal_group_size
        ref_replicated = ref_tensor.repeat(1, temporal_group_size, 1, 1)

        if conditioning_masks is not None:
            # Inpainting mode: composite reference into masked regions only
            cond_slice = conditioning_masks[:, start_idx:end_idx, :, :]
            if cond_slice.shape[1] < temporal_group_size:
                pad = temporal_group_size - cond_slice.shape[1]
                cond_slice = torch.cat(
                    [cond_slice, cond_slice[:, -1:, :, :].repeat(1, pad, 1, 1)], dim=1
                )

            # Composite: reference where mask=1, existing base where mask=0
            mask_expanded = cond_slice.expand_as(ref_replicated)
            base_frames = frames[:, start_idx:end_idx, :, :]
            composited = torch.where(mask_expanded > 0.5, ref_replicated, base_frames)

            frames[:, start_idx:end_idx, :, :] = composited
            masks[:, start_idx:end_idx, :, :] = 0.0  # Preserve composite entirely
        else:
            # Standard mode: replace with reference
            frames[:, start_idx:end_idx, :, :] = ref_replicated

            if spatial_mask is not None:
                ref_mask = spatial_mask.repeat(1, temporal_group_size, 1, 1)
            else:
                ref_mask = torch.zeros(
                    (1, temporal_group_size, height, width),
                    device=device,
                    dtype=torch.float32,
                )
            masks[:, start_idx:end_idx, :, :] = ref_mask

        return frames, masks

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
                f"VaceEncodingBlock._encode_with_conditioning: Input resolution {width}x{height} "
                f"does not match target resolution {block_state.width}x{block_state.height}"
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

            # Spatial masks unused in conditioning mode (masks come from vace_input_masks)
            prepared_refs, _ = load_and_prepare_reference_images(
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
