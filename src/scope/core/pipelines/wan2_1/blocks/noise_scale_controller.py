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


class NoiseScaleControllerBlock(ModularPipelineBlocks):
    """
    Motion-aware noise scale controller block that dynamically adjusts noise scale
    based on the amount of motion detected between frames.

    High motion -> lower noise scale (preserve input frames more)
    Low motion -> higher noise scale (rely on model generation more)
    """

    model_name = "Wan2.1"

    def __init__(self):
        super().__init__()
        self.last_frame = None

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return []

    @property
    def expected_configs(self) -> list[ConfigSpec]:
        return [
            ConfigSpec("num_frame_per_block", 3),
        ]

    @property
    def description(self) -> str:
        return "Motion-aware noise scale controller that adjusts noise scale based on frame motion"

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "video",
                required=True,
                type_hint=torch.Tensor,
                description="Input video tensor to analyze for motion",
            ),
            InputParam(
                "noise_scale",
                type_hint=float,
                default=0.7,
                description="Amount of noise added to video",
            ),
            InputParam(
                "current_noise_scale",
                type_hint=float,
                description="Current noise scale",
            ),
            InputParam(
                "noise_controller",
                type_hint=bool,
                default=True,
                description="Whether to enable motion-aware noise control",
            ),
            InputParam(
                "init_cache",
                type_hint=bool,
                default=False,
                description="Whether caches are being initialized (resets last_frame)",
            ),
            InputParam(
                "manage_cache",
                default=True,
                type_hint=bool,
                description="Whether to automatically determine to (re)initialize caches",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "noise_scale",
                type_hint=float,
                description="Amount of noise added to video",
            ),
            InputParam(
                "current_noise_scale",
                type_hint=float,
                description="Current noise scale",
            ),
            OutputParam(
                "init_cache",
                type_hint=bool,
                description="Whether caches are being initialized (resets last_frame)",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> tuple[Any, PipelineState]:
        block_state = self.get_block_state(state)

        # Reset last_frame when initializing caches
        if block_state.init_cache:
            self.last_frame = None

        # Only apply motion-aware control if enabled
        if block_state.noise_controller:
            video = block_state.video
            chunk_size = components.config.num_frame_per_block

            # Apply motion-aware noise control
            block_state.noise_scale = self._apply_motion_aware_noise_controller(
                video, chunk_size, block_state.noise_scale
            )
        elif (
            block_state.manage_cache
            and block_state.current_noise_scale != block_state.noise_scale
        ):
            block_state.init_cache = True

        block_state.current_noise_scale = block_state.noise_scale

        self.set_block_state(state, block_state)
        return components, state

    def _apply_motion_aware_noise_controller(
        self, input: torch.Tensor, chunk_size: int, noise_scale: float
    ) -> float:
        """
        Calculate motion-aware noise scale based on frame-to-frame differences.

        Args:
            input: Input video tensor of shape (B, C, T, H, W)
            chunk_size: Number of frames in the chunk
            noise_scale: Current noise scale value

        Returns:
            Updated noise scale value
        """
        # The prev seq is the last chunk_size frames of the current input
        prev_seq = input[:, :, -chunk_size:]

        if self.last_frame is None:
            # Shift one position to the left and get chunk_size frames for the curr seq
            curr_seq = input[:, :, -chunk_size - 1 : -1]
        else:
            # Concat the last frame of the previous input with the last chunk_size
            # frames of the current input excluding the last frame
            curr_seq = torch.concat(
                [self.last_frame, input[:, :, -chunk_size:-1]], dim=2
            )

        # In order to calculate the amount of motion in this chunk we calculate the max L2 distance found in the sequences defined above.
        # 1. The squared diff op gives us the squared pixel diffs at each spatial location and frame
        # 2. The average op over B (0), C (1), H (3) and W (4) dimensions gives us the MSE for each frame averaged across all pixels and channels
        # 3. The square root op gives us the RMSE for each frame eg the L2 distance per frame
        # 4. The max op gives us the greatest RMSE/L2 distance of all frames
        # 5. The divison by 0.2 op scales the max L2 distance to a target range
        # 6. The clamping op normalizes to [0, 1]
        max_l2_dist = (
            torch.sqrt(((prev_seq - curr_seq) ** 2).mean(dim=(0, 1, 3, 4))).max() / 0.2
        ).clamp(0, 1)

        # Augment noise scale using the max L2 distance
        # High motion -> high max L2 distance closer to 1.0 -> we want lower noise scale to preserve input frames more
        # Low motion -> low max L2 distance closer to 0.0 -> we want higher noise to rely on input frames less
        max_noise_scale_no_motion = 0.8
        motion_sensitivity_factor = 0.2
        # Bias towards new measurements with some smoothing
        new_measurement_weight = 0.9
        prev_measurement_weight = 0.1

        # 1. Scale the noise scale based on motion
        # 2. Smooth the update to the noise scale -> (new_measurement_weight * new_noise_scale) + (prev_measurement_weight * prev_noise_scale)
        new_noise_scale = (
            max_noise_scale_no_motion - motion_sensitivity_factor * max_l2_dist.item()
        ) * new_measurement_weight + noise_scale * prev_measurement_weight

        # Store the last frame for next iteration
        self.last_frame = input[:, :, [-1]]

        return new_noise_scale
