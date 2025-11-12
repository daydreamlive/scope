import logging
import os
import time

import torch

from ..blending import PromptBlender, handle_transition_prepare
from ..interface import Pipeline, Requirements
from ..process import postprocess_chunk, preprocess_chunk
from .vendor.causvid.models.wan.causal_stream_inference import (
    CausalStreamInferencePipeline,
)

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
        enable_i2v: bool = False,
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

        self.enable_i2v = enable_i2v
        self.stream = CausalStreamInferencePipeline(config, device, enable_i2v=enable_i2v).to(
            device=device, dtype=dtype
        )
        self.device = device
        self.dtype = dtype

        start = time.time()
        state_dict = torch.load(
            os.path.join(config.model_dir, "StreamDiffusionV2/model.pt"),
            map_location="cpu",
        )["generator"]
        self.stream.generator.load_state_dict(state_dict, strict=True)
        print(f"Loaded diffusion state dict in {time.time() - start:.3f}s")

        self.chunk_size = chunk_size
        self.start_chunk_size = start_chunk_size
        self.noise_scale = noise_scale
        self.base_seed = config.get("seed", 42)

        self.prompts = None
        self.denoising_step_list = None

        # Prompt blending with cache reset callback for transitions
        self.prompt_blender = PromptBlender(
            device, dtype, cache_reset_callback=self._initialize_stream_caches
        )

        self.last_frame = None
        self.current_start = 0
        self.current_end = self.stream.frame_seq_length * 2

        # I2V conditioning cache
        self.i2v_visual_context = None  # CLIP visual features for cross-attention
        self.i2v_cond_concat = None  # VAE latent for channel concatenation
        self.i2v_first_chunk = True

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
            transition, self.prompt_blender, self.stream.text_encoder
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
        max_current_start = MAX_ROPE_FREQ_TABLE_SEQ_LEN * self.stream.frame_seq_length
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
                self.stream.denoising_step_list = torch.tensor(
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
        self.stream.kv_cache1 = None

        # Apply prompt blending and set conditional_dict
        self._apply_prompt_blending(prompts, interpolation_method)

        self.stream.vae.model.first_batch = True

        self.last_frame = None
        self.current_start = 0
        self.current_end = self.stream.frame_seq_length * 2

        # Reset I2V first chunk flag when cache is reset
        if self.enable_i2v:
            self.i2v_first_chunk = True

    def set_i2v_conditioning(self, visual_context: torch.Tensor, cond_concat: torch.Tensor):
        """
        Set I2V conditioning that will be used for subsequent generations.

        The Wan 2.1 I2V architecture uses two types of conditioning:
        1. CLIP visual features for cross-attention
        2. VAE-encoded latent for channel concatenation
        """
        self.i2v_visual_context = visual_context
        self.i2v_cond_concat = cond_concat
        self.i2v_first_chunk = True  # Reset for new image
        logger.info(f"I2V conditioning set: visual_context={visual_context.shape}, cond_concat={cond_concat.shape}")

    def clear_i2v_conditioning(self):
        """Clear I2V conditioning (e.g., when switching modes)."""
        self.i2v_visual_context = None
        self.i2v_cond_concat = None
        self.i2v_first_chunk = True
        logger.info("I2V conditioning cleared")

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
            self.stream.text_encoder
        )

        if next_embedding is not None:
            self.stream.conditional_dict = {"prompt_embeds": next_embedding}

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

        # Determine if we're using I2V mode (only for first chunk)
        use_i2v = (self.enable_i2v and
                   self.i2v_visual_context is not None and
                   self.i2v_first_chunk)

        if noise_controller and not use_i2v:
            self._apply_motion_aware_noise_controller(input)

        # Determine the number of denoising steps
        # Higher noise scale -> more denoising steps, more intense changes to input
        # Lower noise scale -> less denoising steps, less intense changes to input
        current_step = int(1000 * self.noise_scale) - 100

        # Create generator from seed for reproducible generation
        # Derive unique seed per chunk using current_start as offset
        frame_seed = self.base_seed + self.current_start
        rng = torch.Generator(device=self.device).manual_seed(frame_seed)

        if use_i2v:
            # I2V mode: Start from pure noise, don't encode input frames
            # The conditioning comes from CLIP visual features (cross-attention only)
            # Input shape should match expected latent dimensions
            # For start_chunk_size=5 frames: [B, 16, 5//4+1=2, H//8, W//8]
            batch_size = 1
            latent_channels = 16
            latent_t = (curr_chunk_size // 4) + 1  # 5 frames -> 2 temporal latents
            latent_h = self.height // 8
            latent_w = self.width // 8

            noisy_latents = torch.randn(
                (batch_size, latent_channels, latent_t, latent_h, latent_w),
                device=self.device,
                dtype=self.dtype,
                generator=rng,
            )
            logger.info(f"I2V first chunk: Using pure noise with shape {noisy_latents.shape}")

            # Mark that we'll need to reset encoder cache after decode
            self._i2v_just_decoded = True
        else:
            # T2V mode or subsequent chunks: Encode frames to latents using VAE
            # If this is the first encode after I2V, use non-streaming encode to avoid cache issues
            if self.enable_i2v and hasattr(self, '_i2v_just_decoded'):
                logger.info("First encode after I2V - using non-streaming VAE encode")
                # Use non-streaming encode for this chunk to avoid cache corruption
                # Convert BCTHW to list of [C, T, H, W]
                input_list = [input[0]]  # Remove batch dim
                latents_list = self.stream.vae.model.encode(
                    input_list,
                    scale=[self.stream.vae.mean.to(self.device), 1.0 / self.stream.vae.std.to(self.device)]
                )
                latents = latents_list[0].unsqueeze(0)  # Add batch dim back

                # Reset VAE to streaming mode for subsequent chunks
                self.stream.vae.model.first_batch = True
                delattr(self, '_i2v_just_decoded')
            else:
                latents = self.stream.vae.model.stream_encode(input)
            # Transpose latents
            latents = latents.transpose(2, 1)

            noise = torch.randn(
                latents.shape,
                device=latents.device,
                dtype=latents.dtype,
                generator=rng,
            )
            # Determine how noisy the latents should be
            # Higher noise scale -> noiser latents, less of inputs preserved
            # Lower noise scale -> less noisy latents, more of inputs preserved
            noisy_latents = noise * self.noise_scale + latents * (1 - self.noise_scale)

        if use_i2v:
            # I2V mode: Generate frames conditioned on image (first chunk only)
            logger.info("Using I2V conditioning for first chunk")
            denoised_pred = self.stream.inference(
                noise=noisy_latents,
                current_start=self.current_start,
                current_end=self.current_end,
                current_step=current_step,
                generator=rng,
                visual_context=self.i2v_visual_context,
                cond_concat=self.i2v_cond_concat,
            )
            # Mark that we've processed the first chunk
            self.i2v_first_chunk = False
        else:
            # T2V mode: existing behavior (or subsequent chunks after I2V)
            denoised_pred = self.stream.inference(
                noise=noisy_latents,
                current_start=self.current_start,
                current_end=self.current_end,
                current_step=current_step,
                generator=rng,
            )

        # # Update tracking variables for next input
        self.last_frame = input[:, :, [-1]]
        self.current_start = self.current_end
        self.current_end += (self.chunk_size // 4) * self.stream.frame_seq_length

        # Decode to pixel space
        output = self.stream.vae.stream_decode_to_pixel(denoised_pred)
        return postprocess_chunk(output)

    def _initialize_stream_caches(self):
        """Initialize stream caches without overriding conditional_dict."""
        noise = torch.zeros(1, 1).to(self.device, self.dtype)
        saved = self.stream.conditional_dict
        self.stream.prepare(noise, text_prompts=[""])
        self.stream.conditional_dict = saved

    def _apply_prompt_blending(self, prompts=None, interpolation_method="linear"):
        """Apply weighted blending of cached prompt embeddings."""
        combined_embeds = self.prompt_blender.blend(
            prompts, interpolation_method, self.stream.text_encoder
        )

        if combined_embeds is None:
            return

        # Set the blended embeddings on the stream
        self.stream.conditional_dict = {"prompt_embeds": combined_embeds}

        # Initialize caches without overriding conditional_dict
        self._initialize_stream_caches()
