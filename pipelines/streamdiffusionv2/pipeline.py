import base64
import logging
import time
from io import BytesIO

import torch
import torchvision.transforms.functional as TF
from diffusers.modular_pipelines import PipelineState
from PIL import Image

from ..blending import EmbeddingBlender
from ..components import ComponentsManager
from ..interface import Pipeline, Requirements
from ..process import postprocess_chunk
from ..wan2_1.components import WanTextEncoderWrapper
from .components import WanVAEWrapper
from .modular_blocks import StreamDiffusionV2Blocks
from .modules.causal_model import CausalWanModel
from .wrapper import CausalWanDiffusionWrapper

logger = logging.getLogger(__name__)

DEFAULT_DENOISING_STEP_LIST = [750, 250]

# Chunk size for streamdiffusionv2
CHUNK_SIZE = 4


class StreamDiffusionV2Pipeline(Pipeline):
    def __init__(
        self,
        config,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
        **kwargs,  # Allow extra kwargs
    ):
        model_dir = getattr(config, "model_dir", None)
        generator_path = getattr(config, "generator_path", None)
        text_encoder_path = getattr(config, "text_encoder_path", None)
        tokenizer_path = getattr(config, "tokenizer_path", None)

        model_config = getattr(config, "model_config", {})
        base_model_name = getattr(model_config, "base_model_name", "Wan2.1-T2V-1.3B")
        base_model_kwargs = getattr(model_config, "base_model_kwargs", {})
        generator_model_name = getattr(
            model_config, "generator_model_name", "generator"
        )

        # Load generator
        start = time.time()
        generator = CausalWanDiffusionWrapper(
            CausalWanModel,
            model_name=base_model_name,
            model_dir=model_dir,
            generator_path=generator_path,
            generator_model_name=generator_model_name,
            enable_clip=True,
            dtype=dtype,  # Pass dtype here
            **base_model_kwargs,
        )

        print(f"Loaded diffusion model in {time.time() - start:.3f}s")

        generator = generator.to(device=device, dtype=dtype)

        start = time.time()
        text_encoder = WanTextEncoderWrapper(
            model_name=base_model_name,
            model_dir=model_dir,
            text_encoder_path=text_encoder_path,
            tokenizer_path=tokenizer_path,
        )
        print(f"Loaded text encoder in {time.time() - start:.3f}s")
        # Move text encoder to target device but use dtype of weights
        text_encoder = text_encoder.to(device=device)

        # Load VAE
        start = time.time()
        vae = WanVAEWrapper(model_dir=model_dir)
        print(f"Loaded VAE in {time.time() - start:.3f}s")
        # Move VAE to target device and use target dtype
        vae = vae.to(device=device, dtype=dtype)

        # Create components config
        components_config = {}
        components_config.update(model_config)
        components_config["device"] = device
        components_config["dtype"] = dtype

        components = ComponentsManager(components_config)
        components.add("generator", generator)
        components.add("scheduler", generator.get_scheduler())
        components.add("vae", vae)
        components.add("text_encoder", text_encoder)

        embedding_blender = EmbeddingBlender(
            device=device,
            dtype=dtype,
        )
        components.add("embedding_blender", embedding_blender)

        self.blocks = StreamDiffusionV2Blocks()
        self.components = components
        self.state = PipelineState()
        # These need to be set right now because InputParam.default on the blocks
        # does not work properly
        self.state.set("current_start_frame", 0)
        self.state.set("manage_cache", True)
        self.state.set("kv_cache_attention_bias", 1.0)
        self.state.set("noise_scale", 0.7)
        self.state.set("noise_controller", True)
        self.state.set("clip_conditioning_scale", 1.0)

        self.state.set("height", config.height)
        self.state.set("width", config.width)
        self.state.set("base_seed", getattr(config, "seed", 42))

        # Persistent I2V state
        self.i2v_visual_context = None
        self.i2v_cond_concat = None
        self.i2v_first_chunk = False

        self.first_call = True

        # expose self as stream for tests if needed, or just aliasing
        self.stream = self

    def prepare(self, should_prepare: bool = False, **kwargs) -> Requirements:
        return Requirements(input_size=CHUNK_SIZE)

    def encode_image_for_i2v(
        self, image_tensor: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode an image tensor for I2V conditioning.
        Args:
            image_tensor: Image tensor [3, H, W] in range [-1, 1] or [0, 1] depending on previous steps?
                          WanCLIPImageEncoder expects [-1, 1] if we use encode_image_from_tensor
        Returns:
            visual_context: CLIP features
            cond_concat: VAE latents
        """
        # Ensure tensor is on device
        image_tensor = image_tensor.to(
            device=self.components.config.device, dtype=self.components.config.dtype
        )

        if len(image_tensor.shape) == 3:
            image_tensor = image_tensor.unsqueeze(0)  # [1, 3, H, W]

        # 1. Encode CLIP features
        clip_encoder = self.components.generator.clip_encoder
        if clip_encoder is None:
            logger.warning("CLIP encoder not available.")
            visual_context = None
        else:
            visual_context = clip_encoder.encode_image_from_tensor(
                image_tensor
            )  # [1, 257, 1280]

        # 2. Encode VAE latents for channel concatenation
        # VAE expects [B, C, T, H, W]
        video_tensor = image_tensor.unsqueeze(2)  # [1, 3, 1, H, W]

        # Assuming input is [-1, 1]. WanVAEWrapper doesn't seem to normalize inside, it expects input.
        # DenoiseBlock usually receives normalized latents? No, VAE encode_to_latent takes pixel.
        cond_latents = self.components.vae.encode_to_latent(
            video_tensor
        )  # [1, 16, 1, h, w]

        # Create mask for I2V (4 channels of ones for the first frame)
        # Wan2.1 I2V expects 20 channels: 16 latent + 4 mask
        # The mask should be 1s for the conditioned frame
        mask = torch.ones(
            cond_latents.shape[0],
            4,
            cond_latents.shape[2],
            cond_latents.shape[3],
            cond_latents.shape[4],
            device=cond_latents.device,
            dtype=cond_latents.dtype,
        )

        cond_concat = torch.cat([cond_latents, mask], dim=1)  # [1, 20, 1, h, w]

        return visual_context, cond_concat

    def set_i2v_conditioning(
        self, visual_context: torch.Tensor, cond_concat: torch.Tensor
    ):
        """Set the persistent I2V conditioning."""
        self.i2v_visual_context = visual_context
        self.i2v_cond_concat = cond_concat
        self.i2v_first_chunk = True  # Maybe used for something?

    def clear_i2v_conditioning(self):
        """Clear the persistent I2V conditioning."""
        self.i2v_visual_context = None
        self.i2v_cond_concat = None
        self.i2v_first_chunk = False

    def __call__(
        self,
        **kwargs,
    ) -> torch.Tensor:
        if self.first_call:
            self.state.set("init_cache", True)
            self.first_call = False
        else:
            # This will be overriden if the init_cache is passed in kwargs
            self.state.set("init_cache", False)

        return self._generate(**kwargs)

    def _generate(self, **kwargs) -> torch.Tensor:
        for k, v in kwargs.items():
            self.state.set(k, v)

        # Handle input_image for CLIP conditioning (Ephemeral / One-off)
        input_image = kwargs.get("input_image")
        clip_features = self.i2v_visual_context  # Start with persistent if available
        i2v_latents = self.i2v_cond_concat  # Start with persistent

        logger.info(f"_generate called: input_image={'present' if input_image else 'None'}, persistent_clip={'present' if self.i2v_visual_context is not None else 'None'}")

        model_type = getattr(self.components.generator.model, "model_type", "t2v")

        # Handle explicit None to clear the image
        if input_image is None and "input_image" in kwargs:
            logger.info("Clearing CLIP features (input_image=None)")
            self.i2v_visual_context = None
            self.i2v_cond_concat = None
            clip_features = None
            i2v_latents = None

        if input_image:
            try:
                image = None
                if isinstance(input_image, str):
                    # Decode base64 image
                    img_str = input_image
                    if "," in img_str:
                        img_str = img_str.split(",", 1)[1]
                    image_bytes = base64.b64decode(img_str)
                    image = Image.open(BytesIO(image_bytes))
                elif isinstance(input_image, Image.Image):
                    image = input_image

                if image:
                    # Encode CLIP (Only for I2V model, as T2V doesn't support it properly yet)
                    if (
                        model_type == "i2v"
                        and hasattr(self.components.generator, "clip_encoder")
                        and self.components.generator.clip_encoder is not None
                    ):
                        clip_features = (
                            self.components.generator.clip_encoder.encode_image(image)
                        )
                        # Store persistently for future frames
                        self.i2v_visual_context = clip_features
                        logger.info(f"Encoded CLIP features from input image, shape: {clip_features.shape}")
                    elif model_type == "t2v":
                        logger.warning("CLIP features are not supported for T2V model. Ignoring input image for CLIP conditioning.")
                        clip_features = None
                    else:
                        logger.warning("CLIP encoder not available, cannot encode input image")

                    # Encode VAE (Channel Concat) - Only if model is I2V
                    if model_type == "i2v":
                        # Convert to tensor
                        image_tensor = (
                            TF.to_tensor(image).sub_(0.5).div_(0.5)
                        )  # [-1, 1]
                        # [C, H, W] -> [1, C, H, W]
                        image_tensor = image_tensor.unsqueeze(0).to(
                            device=self.components.config.device,
                            dtype=self.components.config.dtype,
                        )
                        # [1, C, H, W] -> [1, C, T=1, H, W]
                        video_tensor = image_tensor.unsqueeze(2)

                        cond_latents = self.components.vae.encode_to_latent(
                            video_tensor
                        )

                        # Create mask
                        mask = torch.ones(
                            cond_latents.shape[0],
                            4,
                            cond_latents.shape[2],
                            cond_latents.shape[3],
                            cond_latents.shape[4],
                            device=cond_latents.device,
                            dtype=cond_latents.dtype,
                        )
                        i2v_latents = torch.cat(
                            [cond_latents, mask], dim=1
                        )  # [1, 20, 1, h, w]

            except Exception as e:
                logger.error(f"Failed to encode input image: {e}")

        self.state.set("clip_features", clip_features)

        if clip_features is not None:
            logger.info(f"Setting CLIP features in state, clip_conditioning_scale={kwargs.get('clip_conditioning_scale', 1.0)}")
        else:
            logger.debug("No CLIP features to set in state")

        # Only set i2v_conditioning_latent if model supports it (I2V)
        # TODO: This is not implemented at the moment
        if model_type == "i2v":
            self.state.set("i2v_conditioning_latent", i2v_latents)

        # Clear transition from state if not provided to prevent stale transitions
        if "transition" not in kwargs:
            self.state.set("transition", None)

        if self.state.get("denoising_step_list") is None:
            self.state.set("denoising_step_list", DEFAULT_DENOISING_STEP_LIST)

        _, self.state = self.blocks(self.components, self.state)

        # Update state flags
        if self.i2v_first_chunk:
            self.i2v_first_chunk = False

        return postprocess_chunk(self.state.values["output_video"])
