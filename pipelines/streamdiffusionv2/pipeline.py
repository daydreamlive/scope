import logging
import os
import time

import torch

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

        self.stream = CausalStreamInferencePipeline(config, device).to(
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

        self.prompts = None
        self.last_frame = None
        self.current_start = 0
        self.current_end = self.stream.frame_seq_length * 2

    def prepare(
        self, prompts: list[str] = None, should_prepare: bool = False
    ) -> Requirements:
        if should_prepare:
            logger.info("Initiating pipeline prepare for request")

        if prompts is not None and prompts != self.prompts:
            logger.info("Initiating pipeline prepare for prompt update")
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
            if prompts is not None:
                self.prompts = prompts
            self._prepare_pipeline()

        if self.last_frame is None:
            return Requirements(input_size=self.start_chunk_size)
        else:
            return Requirements(input_size=self.chunk_size)

    @torch.no_grad()
    def _prepare_pipeline(self):
        # Trigger KV + cross-attn cache re-initialization in prepare()
        self.stream.kv_cache1 = None

        # The CausalInferenceStreamPipeline.prepare() call only uses the batch_size dim
        # of the noise param so we can generate a dummy tensor here with the right batch_size dim
        noise = torch.zeros(1, 1).to(self.device, self.dtype)
        self.stream.prepare(noise, self.prompts)

        self.stream.vae.model.first_batch = True

        self.last_frame = None
        self.current_start = 0
        self.current_end = self.stream.frame_seq_length * 2

    @torch.no_grad()
    def __call__(
        self,
        input: torch.Tensor | list[torch.Tensor] | None = None,
        prompts: list[str] = None,
    ) -> torch.Tensor:
        if input is None:
            raise ValueError("Input cannot be None for StreamDiffusionV2Pipeline")

        exp_chunk_size = self.prepare().input_size

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
        # Determine the number of denoising steps
        # Higher noise scale -> more denoising steps, more intense changes to input
        # Lower noise scale -> less denoising steps, less intense changes to input
        current_step = int(1000 * self.noise_scale) - 100

        # Encode frames to latents using VAE
        latents = self.stream.vae.model.stream_encode(input)
        # Transpose latents
        latents = latents.transpose(2, 1)

        noise = torch.randn_like(latents)
        # Determine how noisy the latents should be
        # Higher noise scale -> noiser latents, less of inputs preserved
        # Lower noise scale -> less noisy latents, more of inputs preserved
        noisy_latents = noise * self.noise_scale + latents * (1 - self.noise_scale)
        denoised_pred = self.stream.inference(
            noise=noisy_latents,
            current_start=self.current_start,
            current_end=self.current_end,
            current_step=current_step,
        )

        # # Update tracking variables for next input
        self.last_frame = input[:, :, [-1]]
        self.current_start = self.current_end
        self.current_end += (self.chunk_size // 4) * self.stream.frame_seq_length

        # Decode to pixel space
        output = self.stream.vae.stream_decode_to_pixel(denoised_pred)
        return postprocess_chunk(output)
