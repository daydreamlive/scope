from typing import Any

import torch
from diffusers.modular_pipelines import (
    ModularPipelineBlocks,
    PipelineState,
)
from diffusers.modular_pipelines.modular_pipeline_utils import (
    ConfigSpec,
    InputParam,
    OutputParam,
)

from scope.core.pipelines.process import preprocess_chunk


class PreprocessVideoBlock(ModularPipelineBlocks):
    model_name = "Wan2.1"

    @property
    def expected_configs(self) -> list[ConfigSpec]:
        return [
            ConfigSpec("num_frame_per_block", 3),
            ConfigSpec("vae_temporal_downsample_factor", 4),
        ]

    @property
    def description(self) -> str:
        return "Preprocess Video block transforms video so that it is ready for downstream blocks"

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "video",
                default=None,
                type_hint=list[torch.Tensor] | torch.Tensor | None,
                description="Input video to convert into noisy latents",
            ),
            InputParam(
                "vace_input_frames",
                default=None,
                type_hint=list[torch.Tensor] | torch.Tensor | None,
                description="Input frames for VACE conditioning",
            ),
            InputParam(
                "vace_input_masks",
                default=None,
                type_hint=torch.Tensor | None,
                description="Spatial control masks [B, 1, F, H, W] for VACE conditioning",
            ),
            InputParam(
                "height",
                required=True,
                type_hint=int,
                description="Height of the video",
            ),
            InputParam(
                "width",
                required=True,
                type_hint=int,
                description="Width of the video",
            ),
            InputParam(
                "current_start_frame",
                required=True,
                type_hint=int,
                description="Current starting frame index for current block",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "video",
                type_hint=torch.Tensor,
                description="Input video to convert into noisy latents",
            ),
            OutputParam(
                "vace_input_masks",
                type_hint=torch.Tensor,
                description="Resampled VACE spatial control masks [B, 1, F, H, W]",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> tuple[Any, PipelineState]:
        block_state = self.get_block_state(state)

        target_num_frames = (
            components.config.num_frame_per_block
            * components.config.vae_temporal_downsample_factor
        )
        # The first block will require an additional frame due to behavior of VAE
        if block_state.current_start_frame == 0:
            target_num_frames += 1

        if block_state.video is not None and isinstance(block_state.video, list):
            block_state.video = preprocess_video(
                block_state.video,
                device=components.config.device,
                dtype=components.config.dtype,
                height=block_state.height,
                width=block_state.width,
                target_num_frames=target_num_frames,
            )

        if block_state.vace_input_frames is not None and isinstance(
            block_state.vace_input_frames, list
        ):
            block_state.vace_input_frames = preprocess_video(
                block_state.vace_input_frames,
                device=components.config.device,
                dtype=components.config.dtype,
                height=block_state.height,
                width=block_state.width,
                target_num_frames=target_num_frames,
            )

        # Resample vace_input_masks to match target_num_frames.
        # On the first chunk (current_start_frame == 0), target_num_frames is one
        # greater than the default chunk size, so masks arriving from the queue
        # (or a client parameter) would otherwise be one frame short, causing a
        # shape mismatch inside VaceEncodingBlock._encode_with_conditioning.
        if block_state.vace_input_masks is not None:
            masks = block_state.vace_input_masks
            if isinstance(masks, list):
                masks = (
                    torch.stack(masks, dim=2)
                    if masks[0].dim() == 4
                    else torch.stack(masks, dim=0)
                )
            mask_frames = masks.shape[2]
            if mask_frames != target_num_frames:
                indices = (
                    torch.linspace(
                        0,
                        mask_frames - 1,
                        target_num_frames,
                        device=masks.device,
                    )
                    .round()
                    .long()
                )
                masks = masks[:, :, indices]
            block_state.vace_input_masks = masks

        self.set_block_state(state, block_state)
        return components, state


def preprocess_video(
    video: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    height: int,
    width: int,
    target_num_frames: int,
) -> torch.Tensor:
    video = preprocess_chunk(
        video,
        device,
        dtype,
        height=height,
        width=width,
    )

    _, _, num_frames, _, _ = video.shape
    # If we do not have enough frames use linear interpolation to resample existing frames
    # to meet the required number of frames
    if num_frames != target_num_frames:
        indices = (
            torch.linspace(
                0,
                num_frames - 1,
                target_num_frames,
                device=video.device,
            )
            .round()
            .long()
        )
        video = video[:, :, indices]

    return video
