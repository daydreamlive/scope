import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)


def clip_preprocess(
    image: torch.Tensor,
    size: int = 224,
    mean: list[float] = [0.48145466, 0.4578275, 0.40821073],
    std: list[float] = [0.26862954, 0.26130258, 0.27577711],
    crop: bool = True,
) -> torch.Tensor:
    """clip_preprocess: Preprocess image for CLIP Vision model.

    Args:
        image: Input image tensor with shape [B, H, W, C] and values in [0, 1]
        size: Target size for the image (default: 224)
        mean: Mean values for normalization
        std: Standard deviation values for normalization
        crop: Whether to crop the image (default: True)

    Returns:
        Preprocessed image tensor with shape [B, C, H, W]
    """
    image = image[:, :, :, :3] if image.shape[3] > 3 else image
    mean = torch.tensor(mean, device=image.device, dtype=image.dtype)
    std = torch.tensor(std, device=image.device, dtype=image.dtype)
    image = image.movedim(-1, 1)

    if not (image.shape[2] == size and image.shape[3] == size):
        if crop:
            scale = size / min(image.shape[2], image.shape[3])
            scale_size = (round(scale * image.shape[2]), round(scale * image.shape[3]))
        else:
            scale_size = (size, size)

        image = F.interpolate(image, size=scale_size, mode="bicubic", antialias=True)
        h = (image.shape[2] - size) // 2
        w = (image.shape[3] - size) // 2
        image = image[:, :, h : h + size, w : w + size]

    image = torch.clip((255.0 * image), 0, 255).round() / 255.0
    return (image - mean.view([3, 1, 1])) / std.view([3, 1, 1])


class CLIPVisionAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(q, k, v)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.embed_dim)

        return self.out_proj(attn_output)


class CLIPMLP(nn.Module):
    def __init__(self, embed_dim: int, intermediate_size: int):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, intermediate_size, bias=True)
        self.fc2 = nn.Linear(intermediate_size, embed_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x


class CLIPEncoderLayer(nn.Module):
    def __init__(
        self, embed_dim: int, num_heads: int, intermediate_size: int, eps: float = 1e-5
    ):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(embed_dim, eps=eps)
        self.self_attn = CLIPVisionAttention(embed_dim, num_heads)
        self.layer_norm2 = nn.LayerNorm(embed_dim, eps=eps)
        self.mlp = CLIPMLP(embed_dim, intermediate_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.layer_norm1(x))
        x = x + self.mlp(self.layer_norm2(x))
        return x


class CLIPVisionEmbeddings(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_channels: int = 3,
        patch_size: int = 14,
        image_size: int = 224,
    ):
        super().__init__()
        num_patches = (image_size // patch_size) ** 2
        self.class_embedding = nn.Parameter(torch.zeros(embed_dim))
        self.patch_embedding = nn.Conv2d(
            num_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )
        self.position_embedding = nn.Embedding(num_patches + 1, embed_dim)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        embeds = self.patch_embedding(pixel_values).flatten(2).transpose(1, 2)
        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeds = torch.cat([class_embeds, embeds], dim=1)
        embeds = embeds + self.position_embedding.weight
        return embeds


class CLIPVisionTransformer(nn.Module):
    def __init__(
        self,
        embed_dim: int = 1664,
        num_heads: int = 16,
        num_layers: int = 48,
        intermediate_size: int = 8192,
        num_channels: int = 3,
        patch_size: int = 14,
        image_size: int = 224,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.embeddings = CLIPVisionEmbeddings(
            embed_dim, num_channels, patch_size, image_size
        )
        self.pre_layrnorm = nn.LayerNorm(embed_dim, eps=eps)
        self.encoder_layers = nn.ModuleList(
            [
                CLIPEncoderLayer(embed_dim, num_heads, intermediate_size, eps)
                for _ in range(num_layers)
            ]
        )
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=eps)
        self.num_layers = num_layers

    def forward(
        self, pixel_values: torch.Tensor, return_penultimate: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = self.embeddings(pixel_values)
        x = self.pre_layrnorm(x)

        penultimate = None
        for i, layer in enumerate(self.encoder_layers):
            x = layer(x)
            if return_penultimate and i == self.num_layers - 2:
                penultimate = x

        pooled_output = self.post_layernorm(x[:, 0, :])

        return x, penultimate, pooled_output


class CLIPVisionModel(nn.Module):
    def __init__(
        self,
        embed_dim: int = 1664,
        projection_dim: int = 1280,
        num_heads: int = 16,
        num_layers: int = 48,
        intermediate_size: int = 8192,
        image_size: int = 224,
        patch_size: int = 14,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.vision_model = CLIPVisionTransformer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            intermediate_size=intermediate_size,
            num_channels=3,
            patch_size=patch_size,
            image_size=image_size,
            eps=eps,
        )
        self.visual_projection = nn.Linear(embed_dim, projection_dim, bias=False)
        self.image_size = image_size

    def forward(
        self, pixel_values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """forward: Forward pass through the CLIP Vision model.

        Returns:
            last_hidden_state: [B, num_patches + 1, projection_dim]
            penultimate_hidden_states: [B, num_patches + 1, projection_dim]
            image_embeds: [B, projection_dim]
        """
        x, penultimate, pooled_output = self.vision_model(pixel_values)

        # Apply visual projection to all tokens (needed for Wan models)
        batch_size, seq_len, _ = x.shape
        x_projected = self.visual_projection(x.reshape(-1, x.shape[-1])).reshape(
            batch_size, seq_len, -1
        )

        if penultimate is not None:
            penultimate_projected = self.visual_projection(
                penultimate.reshape(-1, penultimate.shape[-1])
            ).reshape(batch_size, seq_len, -1)
        else:
            penultimate_projected = None

        image_embeds = self.visual_projection(pooled_output)

        return x_projected, penultimate_projected, image_embeds


class CLIPVisionEncoder:
    """CLIPVisionEncoder: Wrapper for CLIP Vision model that handles loading and encoding."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        self.device = device if device is not None else torch.device("cpu")
        self.dtype = dtype
        self.checkpoint_path = Path(checkpoint_path)

        logger.info(
            f"CLIPVisionEncoder: Loading CLIP Vision model from {self.checkpoint_path}"
        )

        self.model = CLIPVisionModel(
            embed_dim=1664,
            projection_dim=1280,
            num_heads=16,
            num_layers=48,
            intermediate_size=8192,
            image_size=224,
            patch_size=14,
        )

        state_dict = torch.load(
            self.checkpoint_path, map_location="cpu", weights_only=True
        )

        if "visual.transformer.resblocks.0.attn.in_proj_weight" in state_dict:
            state_dict = self._convert_openclip_to_transformers(state_dict)

        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning(f"CLIPVisionEncoder: Missing keys: {missing}")
        if unexpected:
            logger.warning(f"CLIPVisionEncoder: Unexpected keys: {unexpected}")

        self.model.to(device=self.device, dtype=self.dtype)
        self.model.eval()

        logger.info("CLIPVisionEncoder: Model loaded successfully")

    def _convert_openclip_to_transformers(self, state_dict: dict) -> dict:
        """_convert_openclip_to_transformers: Convert OpenCLIP state dict to transformers format."""
        logger.info(
            "CLIPVisionEncoder: Converting OpenCLIP state dict to transformers format"
        )

        new_state_dict = {}

        for key, value in state_dict.items():
            if not key.startswith("visual."):
                continue

            new_key = key.replace("visual.", "vision_model.")

            if "transformer.resblocks." in new_key:
                new_key = new_key.replace("transformer.resblocks.", "encoder_layers.")

            if "in_proj_weight" in new_key or "in_proj_bias" in new_key:
                continue

            if "attn.out_proj" in new_key:
                new_key = new_key.replace("attn.out_proj", "self_attn.out_proj")

            if "ln_1" in new_key:
                new_key = new_key.replace("ln_1", "layer_norm1")
            elif "ln_2" in new_key:
                new_key = new_key.replace("ln_2", "layer_norm2")
            elif "ln_pre" in new_key:
                new_key = new_key.replace("ln_pre", "pre_layrnorm")
            elif "ln_post" in new_key:
                new_key = new_key.replace("ln_post", "post_layernorm")

            if "mlp.c_fc" in new_key:
                new_key = new_key.replace("mlp.c_fc", "mlp.fc1")
            elif "mlp.c_proj" in new_key:
                new_key = new_key.replace("mlp.c_proj", "mlp.fc2")

            if "class_embedding" in new_key:
                new_key = "vision_model.embeddings.class_embedding"
            elif "conv1" in new_key:
                new_key = new_key.replace("conv1", "embeddings.patch_embedding")
            elif "positional_embedding" in new_key:
                new_key = new_key.replace(
                    "positional_embedding", "embeddings.position_embedding.weight"
                )

            if (
                "proj" in new_key
                and "out_proj" not in new_key
                and "embeddings" not in new_key
            ):
                new_key = new_key.replace("proj", "visual_projection.weight")
                value = value.T

            new_state_dict[new_key] = value

        return new_state_dict

    @torch.no_grad()
    def encode_image(
        self, image: torch.Tensor | Image.Image | str | Path, crop: bool = True
    ) -> torch.Tensor:
        """encode_image: Encode an image using the CLIP Vision model.

        Args:
            image: Input image as tensor [H, W, C] or [B, H, W, C] with values in [0, 1],
                   PIL Image, or path to image file
            crop: Whether to crop the image during preprocessing

        Returns:
            Encoded image features with shape [B, 257, projection_dim] (penultimate hidden states projected)
            For ViT-huge-14: [B, 257, 1280]
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")

        if isinstance(image, Image.Image):
            import numpy as np

            image = torch.from_numpy(np.array(image)).float() / 255.0

        if image.ndim == 3:
            image = image.unsqueeze(0)

        image = image.to(device=self.device, dtype=self.dtype)

        pixel_values = clip_preprocess(image, size=self.model.image_size, crop=crop)
        pixel_values = pixel_values.to(dtype=self.dtype)

        _, penultimate, _ = self.model(pixel_values)

        return penultimate

    def to(self, device: torch.device | None = None, dtype: torch.dtype | None = None):
        """to: Move the model to a different device and/or dtype."""
        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype
        self.model.to(device=self.device, dtype=self.dtype)
        return self
