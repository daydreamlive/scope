import logging
import os
import time

import torch
from diffusers.modular_pipelines import PipelineState
from diffusers.modular_pipelines.components_manager import ComponentsManager

from ..blending import PromptBlender, handle_transition_prepare
from ..interface import Pipeline, Requirements
from ..process import postprocess_chunk, preprocess_chunk
from .modular_blocks import StreamDiffusionV2Blocks
from .vendor.causvid.models.wan.causal_stream_inference import (
    CausalStreamInferencePipeline,
)
from .vendor.causvid.models.wan.wan_wrapper import (
    WanTextEncoder,
    WanVAEWrapper,
    CausalWanDiffusionWrapper,
)
from .vendor.causvid.models.sdxl.sdxl_wrapper import SDXLWrapper, SDXLTextEncoder, SDXLVAE

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


# Wrapper functions moved from vendor/causvid/models/__init__.py
DIFFUSION_NAME_TO_CLASS = {
    "sdxl": SDXLWrapper,
    "wan": None,  # Not used in streamdiffusionv2
    "causal_wan": CausalWanDiffusionWrapper,
}


def get_diffusion_wrapper(model_name):
    return DIFFUSION_NAME_TO_CLASS[model_name]


TEXTENCODER_NAME_TO_CLASS = {
    "sdxl": SDXLTextEncoder,
    "wan": WanTextEncoder,
    "causal_wan": WanTextEncoder,
}


def get_text_encoder_wrapper(model_name):
    return TEXTENCODER_NAME_TO_CLASS[model_name]


VAE_NAME_TO_CLASS = {
    "sdxl": SDXLVAE,
    "wan": WanVAEWrapper,
    "causal_wan": WanVAEWrapper,  # TODO: Change the VAE to the causal version
}


def get_vae_wrapper(model_name):
    return VAE_NAME_TO_CLASS[model_name]


class ComponentProvider:
    """Simple wrapper to provide component access from ComponentsManager to blocks."""

    def __init__(self, components_manager: ComponentsManager, component_name: str, collection: str = "streamdiffusionv2"):
        """
        Initialize the component provider.

        Args:
            components_manager: The ComponentsManager instance
            component_name: Name of the component to provide
            collection: Collection name for retrieving the component
        """
        self.components_manager = components_manager
        self.component_name = component_name
        self.collection = collection
        # Cache components to avoid repeated lookups
        self._stream = None
        self._text_encoder = None
        self._vae = None
        self._generator = None

    @property
    def stream(self):
        """Provide access to the stream component."""
        if self._stream is None:
            self._stream = self.components_manager.get_one(
                name=self.component_name, collection=self.collection
            )
        return self._stream

    @property
    def text_encoder(self):
        """Provide access to the text_encoder component."""
        if self._text_encoder is None:
            self._text_encoder = self.components_manager.get_one(
                name="text_encoder", collection=self.collection
            )
        return self._text_encoder

    @property
    def vae(self):
        """Provide access to the vae component."""
        if self._vae is None:
            self._vae = self.components_manager.get_one(
                name="vae", collection=self.collection
            )
        return self._vae

    @property
    def generator(self):
        """Provide access to the generator component."""
        if self._generator is None:
            self._generator = self.components_manager.get_one(
                name="generator", collection=self.collection
            )
        return self._generator


def load_stream_component(
    config,
    device,
    dtype,
    model_dir,
    components_manager: ComponentsManager,
    collection: str = "streamdiffusionv2",
) -> ComponentProvider:
    """
    Load the CausalStreamInferencePipeline and add it to ComponentsManager.
    Components (text_encoder, vae, generator) are created directly here instead of inside CausalStreamInferencePipeline.

    Args:
        config: Configuration dictionary for the pipeline
        device: Device to run the pipeline on
        dtype: Data type for the pipeline
        model_dir: Directory containing the model files
        components_manager: ComponentsManager instance to add component to
        collection: Collection name for organizing components

    Returns:
        ComponentProvider: A provider that gives access to the stream component
    """
    # Check if components already exist in ComponentsManager
    try:
        existing_stream = components_manager.get_one(name="stream", collection=collection)
        # Components exist, verify individual components are also registered
        try:
            components_manager.get_one(name="text_encoder", collection=collection)
            components_manager.get_one(name="vae", collection=collection)
            components_manager.get_one(name="generator", collection=collection)
            # All components exist, create provider for it
            print(f"Reusing existing stream components from collection '{collection}'")
            return ComponentProvider(components_manager, "stream", collection)
        except Exception:
            # Individual components not registered, register them now
            print(f"Stream component exists but individual components missing, registering them...")
            text_encoder_id = components_manager.add(
                "text_encoder",
                existing_stream.text_encoder,
                collection=collection,
            )
            vae_id = components_manager.add(
                "vae",
                existing_stream.vae,
                collection=collection,
            )
            generator_id = components_manager.add(
                "generator",
                existing_stream.generator,
                collection=collection,
            )
            print(f"Registered individual components: text_encoder={text_encoder_id}, vae={vae_id}, generator={generator_id}")
            return ComponentProvider(components_manager, "stream", collection)
    except Exception:
        # Component doesn't exist, create and add it
        pass

    # Create components directly in pipeline.py instead of inside CausalStreamInferencePipeline
    model_name = config.get("model_name", "causal_wan")
    generator_model_name = config.get("generator_name", model_name)
    text_encoder_path = config.get("text_encoder_path", None)
    tokenizer_path = config.get("tokenizer_path", None)

    # Create generator
    start = time.time()
    generator = get_diffusion_wrapper(model_name=generator_model_name)(
        model_dir=model_dir
    )
    print(f"Loaded diffusion wrapper in {time.time() - start:.3f}s")

    # Create text encoder
    start = time.time()
    text_encoder = get_text_encoder_wrapper(model_name=model_name)(
        model_dir=model_dir,
        text_encoder_path=text_encoder_path,
        tokenizer_path=tokenizer_path,
    )
    print(f"Loaded text encoder in {time.time() - start:.3f}s")

    # Create VAE
    start = time.time()
    vae = get_vae_wrapper(model_name=model_name)(model_dir=model_dir)
    print(f"Loaded VAE in {time.time() - start:.3f}s")

    # Load the generator state dict
    start = time.time()
    model_path = os.path.join(model_dir, "StreamDiffusionV2/model.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at {model_path}. "
            "Please ensure StreamDiffusionV2/model.pt exists in the model directory."
        )

    state_dict_data = torch.load(model_path, map_location="cpu")

    # Handle both dict with "generator" key and direct state dict
    if isinstance(state_dict_data, dict) and "generator" in state_dict_data:
        state_dict = state_dict_data["generator"]
    else:
        state_dict = state_dict_data

    generator.load_state_dict(state_dict, strict=True)
    print(f"Loaded diffusion state dict in {time.time() - start:.3f}s")

    # Create and initialize the stream pipeline with the components
    stream = CausalStreamInferencePipeline(
        config, device, generator=generator, text_encoder=text_encoder, vae=vae
    ).to(device=device, dtype=dtype)

    # Add individual components to ComponentsManager for modular blocks first
    # Each block should depend on only what it needs (text_encoder, vae, generator)
    text_encoder_id = components_manager.add(
        "text_encoder",
        text_encoder,
        collection=collection,
    )
    print(f"Added text_encoder component to ComponentsManager with ID: {text_encoder_id}")

    vae_id = components_manager.add(
        "vae",
        vae,
        collection=collection,
    )
    print(f"Added vae component to ComponentsManager with ID: {vae_id}")

    generator_id = components_manager.add(
        "generator",
        generator,
        collection=collection,
    )
    print(f"Added generator component to ComponentsManager with ID: {generator_id}")

    # Add stream component to ComponentsManager
    component_id = components_manager.add(
        "stream",
        stream,
        collection=collection,
    )
    print(f"Added stream component to ComponentsManager with ID: {component_id}")

    # Create and return provider
    return ComponentProvider(components_manager, "stream", collection)


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

        # Initialize ComponentsManager
        self.components_manager = ComponentsManager()

        # Initialize Modular Diffusers blocks
        self.modular_blocks = StreamDiffusionV2Blocks()

        # Load stream component using ComponentsManager
        self.component_provider = load_stream_component(
            config, device, dtype, config.model_dir, self.components_manager
        )

        # Prompt blending with cache reset callback for transitions
        self.prompt_blender = PromptBlender(
            device, dtype, cache_reset_callback=self._initialize_stream_caches
        )

        self.last_frame = None
        self.current_start = 0
        self.current_end = self.component_provider.stream.frame_seq_length * 2

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
            transition, self.prompt_blender, self.component_provider.stream.text_encoder
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
            MAX_ROPE_FREQ_TABLE_SEQ_LEN * self.component_provider.stream.frame_seq_length
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
                self.component_provider.stream.denoising_step_list = torch.tensor(
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
        self.component_provider.stream.kv_cache1 = None

        # Apply prompt blending and set conditional_dict
        self._apply_prompt_blending(prompts, interpolation_method)

        self.component_provider.stream.vae.model.first_batch = True

        self.last_frame = None
        self.current_start = 0
        self.current_end = self.component_provider.stream.frame_seq_length * 2

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
            self.component_provider.stream.text_encoder
        )

        if next_embedding is not None:
            self.component_provider.stream.conditional_dict = {
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
            "denoising_step_list", self.component_provider.stream.denoising_step_list
        )

        # Determine the number of denoising steps
        current_step = int(1000 * self.noise_scale) - 100
        state.set("current_step", current_step)

        # Set prompt_embeds in state if available from conditional_dict to skip text_encoder work
        if (
            hasattr(self.component_provider.stream, "conditional_dict")
            and self.component_provider.stream.conditional_dict is not None
            and "prompt_embeds" in self.component_provider.stream.conditional_dict
        ):
            state.set("prompt_embeds", self.component_provider.stream.conditional_dict["prompt_embeds"])

        # Execute modular blocks (returns tuple: components, state)
        # Pass component_provider which provides components.stream access
        _, state = self.modular_blocks(self.component_provider, state)

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
        ) * self.component_provider.stream.frame_seq_length

        return postprocess_chunk(output)

    def _initialize_stream_caches(self):
        """Initialize stream caches without overriding conditional_dict."""
        noise = torch.zeros(1, 1).to(self.device, self.dtype)
        saved = self.component_provider.stream.conditional_dict
        self.component_provider.stream.prepare(noise, text_prompts=[""])
        self.component_provider.stream.conditional_dict = saved

    def _apply_prompt_blending(self, prompts=None, interpolation_method="linear"):
        """Apply weighted blending of cached prompt embeddings."""
        combined_embeds = self.prompt_blender.blend(
            prompts, interpolation_method, self.component_provider.stream.text_encoder
        )

        if combined_embeds is None:
            return

        # Set the blended embeddings on the stream
        self.component_provider.stream.conditional_dict = {
            "prompt_embeds": combined_embeds
        }

        # Initialize caches without overriding conditional_dict
        self._initialize_stream_caches()
