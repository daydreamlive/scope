import logging

import torch
from diffusers.modular_pipelines import PipelineState

from ..blending import PromptBlender, handle_transition_prepare
from ..interface import Pipeline, Requirements
from ..process import postprocess_chunk, preprocess_chunk
from .modular_blocks import StreamDiffusionV2Blocks
from .modular_pipeline_wrapper import StreamDiffusionV2ModularPipeline

# https://github.com/daydreamlive/scope/blob/0cf1766186be3802bf97ce550c2c978439f22068/pipelines/streamdiffusionv2/vendor/causvid/models/wan/causal_model.py#L306
MAX_ROPE_FREQ_TABLE_SEQ_LEN = 1024
CURRENT_START_RESET_RATIO = 0.5
# The VAE compresses a pixel frame into a latent frame which consists of patches
# The patch embedding converts spatial patches into tokens
# The VAE does 8x spatial downsampling
# The patch embedding does 2x spatial downsampling
# Thus, we end up spatially scaling down by 16
SCALE_SIZE = 16

logger = logging.getLogger(__name__)


class StreamDiffusionV2Pipeline(Pipeline):
    def __init__(
        self,
        config,
        chunk_size: int = 4,
        start_chunk_size: int = 5,
        noise_scale: float = 0.7,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        if device is None:
            device = torch.device("cuda")

        # The height and width must be divisible by SCALE_SIZE
        req_height = config.get("height", 512)
        req_width = config.get("width", 512)
        self.height = round(req_height / SCALE_SIZE) * SCALE_SIZE
        self.width = round(req_width / SCALE_SIZE) * SCALE_SIZE

        config["height"] = self.height
        config["width"] = self.width

        self.device = device
        self.dtype = dtype

        self.chunk_size = chunk_size
        self.start_chunk_size = start_chunk_size
        self.noise_scale = noise_scale
        self.base_seed = config.get("seed", 42)

        self.prompts = None
        self.denoising_step_list = None

        # Initialize Modular Diffusers blocks and pipeline
        self.modular_blocks = StreamDiffusionV2Blocks()
        self.modular_pipeline = StreamDiffusionV2ModularPipeline(
            config, device, dtype, config.model_dir
        )

        # Prompt blending with cache reset callback for transitions
        self.prompt_blender = PromptBlender(
            device, dtype, cache_reset_callback=self._initialize_stream_caches
        )

        self.last_frame = None
        self.current_start = 0
        self.current_end = self.modular_pipeline.stream.frame_seq_length * 2

    def prepare(self, should_prepare: bool = False, **kwargs) -> Requirements:
        if should_prepare:
            logger.info("prepare: Initiating pipeline prepare for request")

        manage_cache = kwargs.get("manage_cache", None)
        prompts = kwargs.get("prompts", None)
        prompt_interpolation_method = kwargs.get(
            "prompt_interpolation_method", "linear"
        )
        transition = kwargs.get("transition", None)
        denoising_step_list = kwargs.get("denoising_step_list", None)
        noise_controller = kwargs.get("noise_controller", None)
        noise_scale = kwargs.get("noise_scale", None)

        # Check if prompts changed using prompt blender
        if self.prompt_blender.should_update(prompts, prompt_interpolation_method):
            logger.info("prepare: Initiating pipeline prepare for prompt update")
            should_prepare = True

        # Handle prompt transition requests
        should_prepare_from_transition, target_prompts = handle_transition_prepare(
            transition, self.prompt_blender, self.modular_pipeline.stream.text_encoder
        )
        if target_prompts:
            self.prompts = target_prompts
        if should_prepare_from_transition:
            should_prepare = True

        # If manage_cache is True let the pipeline handle cache management for other param updates
        if manage_cache:
            if (
                denoising_step_list is not None
                and denoising_step_list != self.denoising_step_list
            ):
                logger.info("Initating pipeline prepare for denoising step list update")
                should_prepare = True

            if (
                not noise_controller
                and noise_scale is not None
                and noise_scale != self.noise_scale
            ):
                logger.info("Initating pipeline prepare for noise scale update")
                should_prepare = True

        # CausalWanModel uses a RoPE frequency table with a max sequence length of 1024
        # This means that it has positions for 1024 latent frames
        # Each latent frame consists frame_seq_length tokens
        # current_start is used to index into this table and shifts frame_seq_length tokens forward each pipeline call
        # We need to make sure that current_start does not shift past the max sequence length of the RoPE frequency table
        # When we hit the limit we reset the caches and indices
        # See this issue for more context https://github.com/daydreamlive/scope/issues/95
        max_current_start = (
            MAX_ROPE_FREQ_TABLE_SEQ_LEN * self.modular_pipeline.stream.frame_seq_length
        )
        # We reset at whatever is smaller the theoretically max value or some % of it
        max_current_start = min(
            int(max_current_start * CURRENT_START_RESET_RATIO), max_current_start
        )
        if self.current_start >= max_current_start:
            logger.info("Initiating pipeline prepare to reset indices")
            should_prepare = True

        if should_prepare:
            # Update internal state before preparing pipeline
            if denoising_step_list is not None:
                self.denoising_step_list = denoising_step_list
                self.modular_pipeline.stream.denoising_step_list = torch.tensor(
                    denoising_step_list, dtype=torch.long, device=self.device
                )

            if not noise_controller and noise_scale is not None:
                self.noise_scale = noise_scale

            # Prepare pipeline
            # (PromptBlender.blend() returns None if transitioning, which skips cache reset)
            self._prepare_pipeline(prompts, prompt_interpolation_method)

        if self.last_frame is None:
            return Requirements(input_size=self.start_chunk_size)
        else:
            return Requirements(input_size=self.chunk_size)

    @torch.no_grad()
    def _prepare_pipeline(self, prompts=None, interpolation_method="linear"):
        # Trigger KV + cross-attn cache re-initialization in prepare()
        self.modular_pipeline.stream.kv_cache1 = None

        # Apply prompt blending and set conditional_dict
        self._apply_prompt_blending(prompts, interpolation_method)

        self.modular_pipeline.stream.vae.model.first_batch = True

        self.last_frame = None
        self.current_start = 0
        self.current_end = self.modular_pipeline.stream.frame_seq_length * 2

    def _apply_motion_aware_noise_controller(self, input: torch.Tensor):
        # The prev seq is the last chunk_size frames of the current input
        prev_seq = input[:, :, -self.chunk_size :]
        if self.last_frame is None:
            # Shift one position to the left and get chunk_size frames for the curr seq
            curr_seq = input[:, :, -self.chunk_size - 1 : -1]
        else:
            # Concat the last frame of the previous input with the last chunk_size
            # frames of the current input excluding the last frame
            curr_seq = torch.concat(
                [self.last_frame, input[:, :, -self.chunk_size : -1]], dim=2
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
        self.noise_scale = (
            max_noise_scale_no_motion - motion_sensitivity_factor * max_l2_dist.item()
        ) * new_measurement_weight + self.noise_scale * prev_measurement_weight

    @torch.no_grad()
    def __call__(
        self,
        input: torch.Tensor | list[torch.Tensor] | None = None,
        noise_controller: bool = True,
    ) -> torch.Tensor:
        if input is None:
            raise ValueError("Input cannot be None for StreamDiffusionV2Pipeline")

        # Update prompt embedding for this generation call
        # Handles both static blending and temporal transitions
        next_embedding = self.prompt_blender.get_next_embedding(
            self.modular_pipeline.stream.text_encoder
        )

        if next_embedding is not None:
            self.modular_pipeline.stream.conditional_dict = {
                "prompt_embeds": next_embedding
            }

        # Note: The caller must call prepare() before __call__()
        # We just need to get the expected chunk size based on current state
        exp_chunk_size = (
            self.start_chunk_size if self.last_frame is None else self.chunk_size
        )

        curr_chunk_size = len(input) if isinstance(input, list) else input.shape[2]

        # Validate chunk size
        if curr_chunk_size != exp_chunk_size:
            raise RuntimeError(
                f"Incorrect chunk size expected {exp_chunk_size} got {curr_chunk_size}"
            )

        # If a torch.Tensor is passed assume that the input is ready for inference
        if isinstance(input, list):
            # Preprocess input for inference
            input = preprocess_chunk(
                input, self.device, self.dtype, height=self.height, width=self.width
            )

        if noise_controller:
            self._apply_motion_aware_noise_controller(input)

        # Use Modular Diffusers blocks to process the input
        state = PipelineState()

        # Set up state for modular blocks
        state.set("input", input)
        state.set("noise_scale", self.noise_scale)
        state.set("base_seed", self.base_seed)
        state.set("current_start", self.current_start)
        state.set("current_end", self.current_end)
        state.set(
            "denoising_step_list", self.modular_pipeline.stream.denoising_step_list
        )

        # Determine the number of denoising steps
        current_step = int(1000 * self.noise_scale) - 100
        state.set("current_step", current_step)

        # Execute modular blocks (returns tuple: components, state)
        _, state = self.modular_blocks(self.modular_pipeline, state)

        # Get output from state
        output = state.values.get("output")

        if output is None:
            raise RuntimeError("Modular blocks did not produce output")

        # Ensure output is in the right format
        if not isinstance(output, torch.Tensor):
            output = output[0] if isinstance(output, list) else output

        # Update tracking variables for next input
        self.last_frame = input[:, :, [-1]]
        self.current_start = self.current_end
        self.current_end += (
            self.chunk_size // 4
        ) * self.modular_pipeline.stream.frame_seq_length

        return postprocess_chunk(output)

    def _initialize_stream_caches(self):
        """Initialize stream caches without overriding conditional_dict."""
        noise = torch.zeros(1, 1).to(self.device, self.dtype)
        saved = self.modular_pipeline.stream.conditional_dict
        self.modular_pipeline.stream.prepare(noise, text_prompts=[""])
        self.modular_pipeline.stream.conditional_dict = saved

    def _apply_prompt_blending(self, prompts=None, interpolation_method="linear"):
        """Apply weighted blending of cached prompt embeddings."""
        combined_embeds = self.prompt_blender.blend(
            prompts, interpolation_method, self.modular_pipeline.stream.text_encoder
        )

        if combined_embeds is None:
            return

        # Set the blended embeddings on the stream
        self.modular_pipeline.stream.conditional_dict = {
            "prompt_embeds": combined_embeds
        }

        # Initialize caches without overriding conditional_dict
        self._initialize_stream_caches()
