"""
VACE Encoding Block for Reference-to-Video and Depth-Guided Generation.

This block handles encoding of VACE conditioning inputs (reference images or depth maps)
and prepares vace_context for the denoising block.

Modes:
- R2V (Reference-to-Video): Encodes 1-3 static reference images, cached across all chunks
- Depth: Encodes 12-frame depth maps per chunk via standard VACE path with masking

For depth mode, follows original VACE architecture (notes/VACE/vace/models/wan/wan_vace.py):
- vace_input = depth maps (3-channel RGB from annotators)
- masks = ones (all white masks, goes through masking path)
- ref_images = None
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
        return "VaceEncodingBlock: Encode VACE context (R2V or depth) for conditioning"

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "guidance_mode",
                default=None,
                type_hint=str | None,
                description="VACE guidance mode: 'r2v' or 'depth'. None disables VACE.",
            ),
            InputParam(
                "ref_images",
                default=None,
                description="List of reference image paths for R2V mode",
            ),
            InputParam(
                "vace_input",
                default=None,
                description="VACE input frames [B, C, F, H, W]: depth maps for depth mode, video frames for other modes (12 frames per chunk)",
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

        guidance_mode = block_state.guidance_mode
        if guidance_mode is None:
            block_state.vace_context = None
            block_state.vace_ref_images = None
            self.set_block_state(state, block_state)
            return components, state

        if guidance_mode not in ["r2v", "depth"]:
            raise ValueError(
                f"VaceEncodingBlock: guidance_mode must be 'r2v', 'depth', or None, got '{guidance_mode}'"
            )

        current_start = block_state.current_start_frame

        if guidance_mode == "r2v":
            block_state.vace_context, block_state.vace_ref_images = self._encode_r2v(
                components, block_state, current_start, state
            )
        elif guidance_mode == "depth":
            block_state.vace_context, block_state.vace_ref_images = (
                self._encode_standard_vace(components, block_state, current_start)
            )

        self.set_block_state(state, block_state)
        return components, state

    def _encode_r2v(self, components, block_state, current_start, state):
        """
        Encode reference images for R2V mode.

        R2V characteristics:
        - Static reference images (1-3 frames)
        - Encoded once on first chunk (current_start == 0)
        - Cached in state, reused across all chunks
        - Prepended temporally to latent sequence
        - Uses vace_in_dim=96 (16 base * 6 for masking)
        """
        # Check cache first
        cached_context = state.get("_vace_r2v_context_cache")
        cached_refs = state.get("_vace_r2v_refs_cache")

        if current_start > 0 and cached_context is not None:
            logger.info(
                f"VaceEncodingBlock._encode_r2v: Reusing cached R2V context for chunk starting at frame {current_start}"
            )
            return cached_context, cached_refs

        # First chunk or cache miss: encode reference images
        ref_image_paths = block_state.ref_images
        if ref_image_paths is None or len(ref_image_paths) == 0:
            logger.warning(
                "VaceEncodingBlock._encode_r2v: No ref_images provided for R2V mode"
            )
            return None, None

        logger.info(
            f"VaceEncodingBlock._encode_r2v: Encoding {len(ref_image_paths)} reference images for first chunk"
        )

        # Load and prepare reference images
        prepared_refs = load_and_prepare_reference_images(
            ref_image_paths,
            block_state.height,
            block_state.width,
            components.config.device,
        )

        # Encode using main VAE (no cache)
        vae = components.vae
        vae_dtype = next(vae.parameters()).dtype

        # Convert refs to list of [C, 1, H, W] for encoding
        ref_images_list = [ref.to(dtype=vae_dtype) for ref in prepared_refs]

        # Stack refs: list of [C, 1, H, W] -> [num_refs, C, 1, H, W]
        refs_stacked = torch.stack(ref_images_list, dim=0)

        # Encode: [num_refs, C, 1, H, W] -> [num_refs, 1, C_latent, H_latent, W_latent]
        ref_latent_out = vae.encode_to_latent(refs_stacked, use_cache=False)

        # Transpose: [num_refs, 1, C_latent, H_latent, W_latent] -> [num_refs, C_latent, 1, H_latent, W_latent]
        ref_latents = [lat.permute(1, 0, 2, 3) for lat in ref_latent_out]

        # Concatenate along frame dimension: [C_latent, num_refs, H_latent, W_latent]
        vace_context = torch.cat(ref_latents, dim=1)

        # Pad to 96 channels for VACE compatibility (R2V mode uses 96 channels)
        current_channels = vace_context.shape[0]
        if current_channels < 96:
            pad_channels = 96 - current_channels
            padding = torch.zeros(
                (
                    pad_channels,
                    vace_context.shape[1],
                    vace_context.shape[2],
                    vace_context.shape[3],
                ),
                dtype=vace_context.dtype,
                device=vace_context.device,
            )
            vace_context = torch.cat([vace_context, padding], dim=0)

        # Return as list for consistency with model expectations
        vace_context_list = [vace_context]

        # Cache for subsequent chunks
        state.set("_vace_r2v_context_cache", vace_context_list)
        state.set("_vace_r2v_refs_cache", prepared_refs)

        logger.info(
            f"VaceEncodingBlock._encode_r2v: Encoded R2V context shape={vace_context.shape}, "
            f"cached for subsequent chunks"
        )

        return vace_context_list, prepared_refs

    def _encode_standard_vace(self, components, block_state, current_start):
        """
        Encode VACE input using the standard VACE path matching original architecture.

        For depth mode (original VACE approach from notes/VACE/vace/models/wan/wan_vace.py):
        - vace_input = depth maps (3-channel RGB from annotators)
        - masks = ones (all white masks, goes through standard masking path)
        - ref_images = None (no reference images)
        - Standard encoding: z0 = vace_encode_frames(input_frames, None, masks=ones)
                           m0 = vace_encode_masks(masks, None)
                           z = vace_latent(z0, m0)

        Depth characteristics:
        - Per-chunk depth maps (12 frames matching output chunk)
        - Encoded every chunk (no caching)
        - Direct temporal correspondence with output
        - Uses standard VACE path with masking (produces 96 channels: 32 masked + 64 mask_encoding)
        """
        vace_input = block_state.vace_input
        if vace_input is None:
            raise ValueError(
                f"VaceEncodingBlock._encode_standard_vace: vace_input required for depth mode at chunk {current_start}"
            )

        # Validate vace_input shape
        if vace_input.dim() != 5:
            raise ValueError(
                f"VaceEncodingBlock._encode_standard_vace: vace_input must be [B, C, F, H, W], got shape {vace_input.shape}"
            )

        batch_size, channels, num_frames, height, width = vace_input.shape

        # Validate frame count: should be 12 (output frames per chunk)
        expected_output_frames = (
            components.config.num_frame_per_block
            * components.config.vae_temporal_downsample_factor
        )
        if num_frames != expected_output_frames:
            raise ValueError(
                f"VaceEncodingBlock._encode_standard_vace: Expected {expected_output_frames} frames "
                f"(num_frame_per_block={components.config.num_frame_per_block} * "
                f"vae_temporal_downsample_factor={components.config.vae_temporal_downsample_factor}), "
                f"got {num_frames} frames at chunk {current_start}"
            )

        # Validate resolution
        if height != block_state.height or width != block_state.width:
            raise ValueError(
                f"VaceEncodingBlock._encode_standard_vace: Input resolution {height}x{width} "
                f"does not match target resolution {block_state.height}x{block_state.width}"
            )

        logger.info(
            f"VaceEncodingBlock._encode_standard_vace: Encoding {num_frames} frames for depth mode using standard VACE path, chunk {current_start}"
        )
        print(
            f"_encode_standard_vace: Starting depth encoding, vace_input.shape={vace_input.shape}"
        )

        # Use dedicated vace_vae if available, otherwise use main VAE
        vace_vae = getattr(components, "vace_vae", None)
        if vace_vae is None:
            logger.warning(
                "VaceEncodingBlock._encode_standard_vace: vace_vae not found, using main VAE (may affect autoregressive cache)"
            )
            vace_vae = components.vae

        # Import vace_utils for standard encoding path
        from ..vace_utils import vace_encode_frames, vace_encode_masks, vace_latent

        # Ensure 3-channel input for VAE (depth maps should already be 3-channel RGB)
        if channels == 1:
            logger.info(
                "VaceEncodingBlock._encode_standard_vace: Converting 1-channel input to 3-channel RGB"
            )
            vace_input = vace_input.repeat(1, 3, 1, 1, 1)
            print(
                f"_encode_standard_vace: Converted 1-channel to 3-channel, new shape={vace_input.shape}"
            )
        elif channels != 3:
            raise ValueError(
                f"VaceEncodingBlock._encode_standard_vace: Expected 1 or 3 channels, got {channels}"
            )

        vae_dtype = next(vace_vae.parameters()).dtype
        vace_input = vace_input.to(dtype=vae_dtype)

        # Convert to list of [C, F, H, W] for vace_encode_frames
        input_frames = [vace_input[b] for b in range(batch_size)]
        print(
            f"_encode_standard_vace: input_frames list length={len(input_frames)}, first shape={input_frames[0].shape}"
        )

        # For depth mode: masks = ones (all white), ref_images = None
        # This matches original VACE architecture where depth goes through standard masking path
        masks = [
            torch.ones(
                (1, num_frames, height, width),
                dtype=vae_dtype,
                device=vace_input.device,
            )
            for _ in range(batch_size)
        ]
        ref_images = None
        print("_encode_standard_vace: Created masks=ones (all white), ref_images=None")
        print(
            f"_encode_standard_vace: masks list length={len(masks)}, first shape={masks[0].shape}"
        )

        # Standard VACE encoding path (matching wan_vace.py lines 339-341)
        # z0 = vace_encode_frames(input_frames, input_ref_images, masks=input_masks)
        # When masks are provided, set pad_to_96=False because mask encoding (64 channels) will be added later
        print(
            "_encode_standard_vace: Calling vace_encode_frames with masks=ones, pad_to_96=False..."
        )
        z0 = vace_encode_frames(
            vace_vae, input_frames, ref_images, masks=masks, pad_to_96=False
        )
        print(
            f"_encode_standard_vace: After vace_encode_frames, z0 list length={len(z0)}, first shape={z0[0].shape}"
        )

        # m0 = vace_encode_masks(input_masks, input_ref_images)
        print("_encode_standard_vace: Calling vace_encode_masks...")
        m0 = vace_encode_masks(masks, ref_images)
        print(
            f"_encode_standard_vace: After vace_encode_masks, m0 list length={len(m0)}, first shape={m0[0].shape}"
        )

        # z = vace_latent(z0, m0)
        print("_encode_standard_vace: Calling vace_latent to concatenate z0 and m0...")
        z = vace_latent(z0, m0)
        print(
            f"_encode_standard_vace: After vace_latent, z list length={len(z)}, first shape={z[0].shape}"
        )

        # Validate latent frame count
        expected_latent_frames = components.config.num_frame_per_block
        actual_latent_frames = z[0].shape[1]
        if actual_latent_frames != expected_latent_frames:
            raise ValueError(
                f"VaceEncodingBlock._encode_standard_vace: Expected {expected_latent_frames} latent frames, "
                f"got {actual_latent_frames} after VAE encoding"
            )

        logger.info(
            f"VaceEncodingBlock._encode_standard_vace: Encoded VACE context via standard path, "
            f"latent shape={z[0].shape} "
            f"({expected_latent_frames} latent frames from {num_frames} output frames)"
        )

        return z, None
