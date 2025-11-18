from ..model_interface import (
    DiffusionModelInterface,
    TextEncoderInterface,
    VAEInterface,
)
from .wan_base.modules.tokenizers import HuggingfaceTokenizer
from .wan_base.modules.model import WanModel
from .wan_base.modules.vae import _video_vae
from .wan_base.modules.t5 import umt5_xxl
from .wan_base.modules.clip import CLIPModel
from .flow_match import FlowMatchScheduler
from .causal_model import CausalWanModel
from typing import List, Tuple, Dict, Optional
import torch
import os
from safetensors import safe_open
from PIL import Image
import numpy as np

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
default_model_dir = os.path.join(repo_root, "wan_models")


class WanTextEncoder(TextEncoderInterface):
    def __init__(
        self,
        model_dir: Optional[str] = None,
        text_encoder_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
    ) -> None:
        super().__init__()

        # Determine paths with priority: specific paths > model_dir > default
        if text_encoder_path is None:
            model_dir = model_dir if model_dir is not None else default_model_dir
            text_encoder_path = os.path.join(
                model_dir, "Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth"
            )

        if tokenizer_path is None:
            model_dir = model_dir if model_dir is not None else default_model_dir
            tokenizer_path = os.path.join(model_dir, "Wan2.1-T2V-1.3B/google/umt5-xxl/")

        # Load weights first, then create model with those weights
        state_dict = self._load_state_dict(text_encoder_path)

        # Create model with meta device for fast initialization
        with torch.device("meta"):
            self.text_encoder = (
                umt5_xxl(
                    encoder_only=True,
                    return_tokenizer=False,
                    dtype=torch.float32,
                    device=torch.device("meta"),
                )
                .eval()
                .requires_grad_(False)
            )

        # Directly assign weights and materialize on CPU
        self.text_encoder.load_state_dict(state_dict, assign=True)
        self.text_encoder = self.text_encoder.to("cpu")

        self.tokenizer = HuggingfaceTokenizer(
            name=tokenizer_path, seq_len=512, clean="whitespace"
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, text_prompts: List[str]) -> dict:
        ids, mask = self.tokenizer(
            text_prompts, return_mask=True, add_special_tokens=True
        )
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.text_encoder(ids, mask)

        for u, v in zip(context, seq_lens):
            u[v:] = 0.0  # set padding to 0.0

        return {"prompt_embeds": context}

    def _load_state_dict(self, weights_path: str) -> dict:
        """Load text encoder weights with automatic format detection."""
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"Text encoder weights not found at: {weights_path}"
            )

        if weights_path.endswith(".safetensors"):
            # Load from safetensors and convert keys
            state_dict = {}
            with safe_open(weights_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)

        elif weights_path.endswith(".pth") or weights_path.endswith(".pt"):
            # Load from PyTorch format (assume already in correct format)
            state_dict = torch.load(
                weights_path, map_location="cpu", weights_only=False
            )

        else:
            raise ValueError(
                f"Unsupported file format. Expected .safetensors, .pth, or .pt, got: {weights_path}"
            )

        return state_dict


class WanCLIPImageEncoder:
    """
    CLIP Image Encoder for extracting visual features from images.
    Used for image-to-video conditioning via cross-attention.
    """

    def __init__(
        self,
        model_dir: Optional[str] = None,
        clip_checkpoint_path: Optional[str] = None,
        clip_tokenizer_path: Optional[str] = None,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ):
        """
        Initialize CLIP Image Encoder.

        Args:
            model_dir: Base directory for models
            clip_checkpoint_path: Path to CLIP checkpoint (.pth file)
            clip_tokenizer_path: Path to CLIP tokenizer directory
            dtype: Data type for the model
            device: Device to load the model on
        """
        # Determine paths with priority: specific paths > model_dir > default
        if model_dir is None:
            model_dir = default_model_dir

        if clip_checkpoint_path is None:
            clip_checkpoint_path = os.path.join(
                model_dir,
                "Wan2.1-T2V-1.3B/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
            )

        if clip_tokenizer_path is None:
            clip_tokenizer_path = os.path.join(
                model_dir, "Wan2.1-T2V-1.3B/xlm-roberta-large/"
            )

        # Initialize CLIP model
        print(f"Loading CLIP model from: {clip_checkpoint_path}")
        print(f"Loading CLIP tokenizer from: {clip_tokenizer_path}")
        self.clip_model = CLIPModel(
            dtype=dtype,
            device=device,
            checkpoint_path=clip_checkpoint_path,
            tokenizer_path=clip_tokenizer_path,
        )

        self.device = device
        self.dtype = dtype

    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """
        Encode a PIL Image to CLIP features.

        Args:
            image: PIL Image to encode

        Returns:
            CLIP features with shape [1, 257, 1280]
        """
        # Convert PIL image to tensor and normalize
        # CLIP expects images in range [0, 1] with specific normalization
        image_np = np.array(image.convert("RGB")).astype(np.float32) / 255.0

        # Convert to tensor: [H, W, 3] -> [3, H, W]
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)

        # Add frame dimension to match CLIP expectation [C, F, H, W]
        image_tensor = image_tensor.unsqueeze(1)

        # Move to device/dtype and normalize to [-1, 1]
        image_tensor = (
            image_tensor.to(device=self.device, dtype=self.dtype) * 2.0 - 1.0
        )

        # Extract CLIP features
        with torch.no_grad():
            clip_features = self.clip_model.visual([image_tensor])

        return clip_features

    def encode_image_from_tensor(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Encode an image tensor to CLIP features.

        Args:
            image_tensor: Image tensor with shape [B, 3, H, W] in range [-1, 1]

        Returns:
            CLIP features with shape [B, 257, 1280]
        """
        with torch.no_grad():
            # Ensure tensor is in the right format and device
            # Ensure tensor has frame dimension [C, F, H, W]
            if image_tensor.ndim == 3:
                image_tensor = image_tensor.unsqueeze(1)
            elif image_tensor.shape[1] != 1:
                # If tensor is [B, 3, H, W], treat batch as frame dimension
                image_tensor = image_tensor.permute(1, 0, 2, 3)

            image_tensor = image_tensor.to(device=self.device, dtype=self.dtype)

            # Extract CLIP features
            clip_features = self.clip_model.visual([image_tensor])

        return clip_features


class WanVAEWrapper(VAEInterface):
    def __init__(self, model_dir: Optional[str] = None):
        super().__init__()

        model_dir = model_dir if model_dir is not None else default_model_dir
        mean = [
            -0.7571,
            -0.7089,
            -0.9113,
            0.1075,
            -0.1745,
            0.9653,
            -0.1517,
            1.5508,
            0.4134,
            -0.0715,
            0.5517,
            -0.3632,
            -0.1922,
            -0.9497,
            0.2503,
            -0.2921,
        ]
        std = [
            2.8184,
            1.4541,
            2.3275,
            2.6558,
            1.2196,
            1.7708,
            2.6052,
            2.0743,
            3.2687,
            2.1526,
            2.8652,
            1.5579,
            1.6382,
            1.1253,
            2.8251,
            1.9160,
        ]
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

        # init model
        self.model = (
            _video_vae(
                pretrained_path=os.path.join(
                    model_dir, "Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"
                ),
                z_dim=16,
            )
            .eval()
            .requires_grad_(False)
        )

    def decode_to_pixel(self, latent: torch.Tensor) -> torch.Tensor:
        # from [batch_size, num_frames, num_channels, height, width]
        # to [batch_size, num_channels, num_frames, height, width]
        zs = latent.permute(0, 2, 1, 3, 4)

        device, dtype = latent.device, latent.dtype
        scale = [
            self.mean.to(device=device, dtype=dtype),
            1.0 / self.std.to(device=device, dtype=dtype),
        ]

        output = [
            self.model.decode(u.unsqueeze(0), scale).float().clamp_(-1, 1).squeeze(0)
            for u in zs
        ]
        output = torch.stack(output, dim=0)
        # from [batch_size, num_channels, num_frames, height, width]
        # to [batch_size, num_frames, num_channels, height, width]
        output = output.permute(0, 2, 1, 3, 4)
        return output

    def stream_decode_to_pixel(self, latent: torch.Tensor) -> torch.Tensor:
        zs = latent.permute(0, 2, 1, 3, 4)
        zs = zs.to(torch.bfloat16).to("cuda")
        device, dtype = latent.device, latent.dtype
        scale = [
            self.mean.to(device=device, dtype=dtype),
            1.0 / self.std.to(device=device, dtype=dtype),
        ]
        output = self.model.stream_decode(zs, scale).float().clamp_(-1, 1)
        output = output.permute(0, 2, 1, 3, 4)
        return output


class WanDiffusionWrapper(DiffusionModelInterface):
    def __init__(self, model_dir: Optional[str] = None):
        super().__init__()

        model_dir = model_dir if model_dir is not None else default_model_dir

        self.model = WanModel.from_pretrained(
            os.path.join(model_dir, "Wan2.1-T2V-1.3B/")
        )
        self.model.eval()

        self.uniform_timestep = True

        self.scheduler = FlowMatchScheduler(
            shift=8.0, sigma_min=0.0, extra_one_step=True
        )
        self.scheduler.set_timesteps(1000, training=True)

        self.seq_len = 32760  # [1, 21, 16, 60, 104]
        super().post_init()

    def enable_gradient_checkpointing(self) -> None:
        self.model.enable_gradient_checkpointing()

    def _convert_flow_pred_to_x0(
        self, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert flow matching's prediction to x0 prediction.
        flow_pred: the prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = noise - x0
        x_t = (1-sigma_t) * x0 + sigma_t * noise
        we have x0 = x_t - sigma_t * pred
        see derivations https://chatgpt.com/share/67bf8589-3d04-8008-bc6e-4cf1a24e2d0e
        """
        # use higher precision for calculations
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(flow_pred.device),
            [flow_pred, xt, self.scheduler.sigmas, self.scheduler.timesteps],
        )

        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1
        )
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        x0_pred = xt - sigma_t * flow_pred
        return x0_pred.to(original_dtype)

    @staticmethod
    def _convert_x0_to_flow_pred(
        scheduler, x0_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert x0 prediction to flow matching's prediction.
        x0_pred: the x0 prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = (x_t - x_0) / sigma_t
        """
        # use higher precision for calculations
        original_dtype = x0_pred.dtype
        x0_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(x0_pred.device),
            [x0_pred, xt, scheduler.sigmas, scheduler.timesteps],
        )
        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1
        )
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        flow_pred = (xt - x0_pred) / sigma_t
        return flow_pred.to(original_dtype)

    def forward(
        self,
        noisy_image_or_video: torch.Tensor,
        conditional_dict: dict,
        timestep: torch.Tensor,
        kv_cache: Optional[List[dict]] = None,
        crossattn_cache: Optional[List[dict]] = None,
        current_start: Optional[int] = None,
        current_end: Optional[int] = None,
    ) -> torch.Tensor:
        prompt_embeds = conditional_dict["prompt_embeds"]

        # [B, F] -> [B]
        if self.uniform_timestep:
            input_timestep = timestep[:, 0]
        else:
            input_timestep = timestep

        if kv_cache is not None:
            flow_pred = self.model(
                noisy_image_or_video.permute(0, 2, 1, 3, 4),
                t=input_timestep,
                context=prompt_embeds,
                seq_len=self.seq_len,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start,
                current_end=current_end,
            ).permute(0, 2, 1, 3, 4)
        else:
            flow_pred = self.model(
                noisy_image_or_video.permute(0, 2, 1, 3, 4),
                t=input_timestep,
                context=prompt_embeds,
                seq_len=self.seq_len,
            ).permute(0, 2, 1, 3, 4)

        pred_x0 = self._convert_flow_pred_to_x0(
            flow_pred=flow_pred.flatten(0, 1),
            xt=noisy_image_or_video.flatten(0, 1),
            timestep=timestep.flatten(0, 1),
        ).unflatten(0, flow_pred.shape[:2])

        return pred_x0


class CausalWanDiffusionWrapper(WanDiffusionWrapper):
    def __init__(self, model_dir: Optional[str] = None, enable_clip: bool = False):
        """
        Initialize Causal Wan Diffusion Wrapper with optional CLIP support.

        Args:
            model_dir: Directory containing model checkpoints
            enable_clip: If True, initialize CLIP encoder for image conditioning
        """
        super().__init__(model_dir=model_dir)

        model_dir = model_dir if model_dir is not None else default_model_dir

        self.model = CausalWanModel.from_pretrained(
            os.path.join(model_dir, "Wan2.1-T2V-1.3B/")
        )
        self.model.eval()

        # Initialize CLIP embedding layer after loading pretrained weights
        # This ensures the layer is created on the correct device
        self.model._post_init_clip_embedding()

        self.uniform_timestep = False

        # Initialize CLIP encoder if enabled
        self.clip_encoder = None
        if enable_clip:
            try:
                print(f"Initializing CLIP encoder with model_dir={model_dir}")
                self.clip_encoder = WanCLIPImageEncoder(
                    model_dir=model_dir,
                    dtype=torch.float16,
                    device="cuda",
                )
                print("✓ CLIP Image Encoder initialized for image conditioning")
            except Exception as e:
                import traceback
                print(f"⚠ Warning: Failed to initialize CLIP encoder: {e}")
                print(f"Traceback: {traceback.format_exc()}")
                print("  Image conditioning will be disabled.")
                self.clip_encoder = None

    def forward(
        self,
        noisy_image_or_video: torch.Tensor,
        conditional_dict: dict,
        timestep: torch.Tensor,
        kv_cache: Optional[List[dict]] = None,
        crossattn_cache: Optional[List[dict]] = None,
        current_start: Optional[int] = None,
        current_end: Optional[int] = None,
        clip_features: Optional[torch.Tensor] = None,
        clip_conditioning_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Forward pass with optional CLIP features for image conditioning.

        Args:
            noisy_image_or_video: Noisy latent tensor
            conditional_dict: Dictionary containing prompt embeddings
            timestep: Timestep tensor
            kv_cache: KV cache for efficient inference
            crossattn_cache: Cross-attention cache
            current_start: Start frame index for causal attention
            current_end: End frame index for causal attention
            clip_features: Optional CLIP image features [B, 257, 1280]
            clip_conditioning_scale: Scale factor for CLIP conditioning strength

        Returns:
            Predicted clean latent tensor
        """
        prompt_embeds = conditional_dict["prompt_embeds"]

        # [B, F] -> [B]
        if self.uniform_timestep:
            input_timestep = timestep[:, 0]
        else:
            input_timestep = timestep

        # Prepare model inputs
        # Apply conditioning scale to CLIP features if present
        scaled_clip_features = None
        if clip_features is not None:
            scaled_clip_features = clip_features * clip_conditioning_scale

        model_kwargs = {
            "t": input_timestep,
            "context": prompt_embeds,
            "seq_len": self.seq_len,
            "clip_fea": scaled_clip_features,  # Add scaled CLIP features
        }

        # Add cache arguments if provided
        if kv_cache is not None:
            model_kwargs.update({
                "kv_cache": kv_cache,
                "crossattn_cache": crossattn_cache,
                "current_start": current_start,
                "current_end": current_end,
            })

        # Forward through model
        flow_pred = self.model(
            noisy_image_or_video.permute(0, 2, 1, 3, 4),
            **model_kwargs
        ).permute(0, 2, 1, 3, 4)

        # Convert flow prediction to x0
        pred_x0 = self._convert_flow_pred_to_x0(
            flow_pred=flow_pred.flatten(0, 1),
            xt=noisy_image_or_video.flatten(0, 1),
            timestep=timestep.flatten(0, 1),
        ).unflatten(0, flow_pred.shape[:2])

        return pred_x0
