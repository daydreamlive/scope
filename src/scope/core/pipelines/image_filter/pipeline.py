"""Image Filter post processor pipeline.

Applies real-time color grading and effects to video frames.
Inspired by ComfyUI-Purz Image Filter Live node.
"""

import logging
from typing import TYPE_CHECKING

import torch
from einops import rearrange

from ..interface import Pipeline, Requirements
from ..process import normalize_frame_sizes, postprocess_chunk, preprocess_chunk
from .schema import FilterPreset, ImageFilterConfig

if TYPE_CHECKING:
    from ..schema import BasePipelineConfig

logger = logging.getLogger(__name__)


# Preset definitions - each preset defines parameter overrides
# Parameters not specified default to neutral values (0 for adjustments, 1 for gamma)
PRESETS: dict[FilterPreset, dict[str, float | int]] = {
    # === Cinematic ===
    FilterPreset.CINEMATIC: {
        "contrast": 0.15,
        "saturation": -0.1,
        "temperature": 0.1,
        "shadows": 0.1,
        "highlights": -0.1,
        "vignette": 0.3,
        "grain": 0.02,
    },
    FilterPreset.BLOCKBUSTER: {
        "contrast": 0.25,
        "saturation": 0.1,
        "temperature": -0.15,
        "tint": 0.05,
        "shadows": 0.15,
        "highlights": -0.05,
        "sharpen": 0.3,
        "vignette": 0.4,
    },
    FilterPreset.NOIR: {
        "saturation": -1.0,  # Full B&W
        "contrast": 0.4,
        "gamma": 0.9,
        "shadows": -0.1,
        "highlights": 0.1,
        "vignette": 0.5,
        "grain": 0.05,
    },
    FilterPreset.SCIFI: {
        "contrast": 0.2,
        "saturation": -0.2,
        "temperature": -0.3,
        "tint": 0.1,
        "highlights": 0.1,
        "sharpen": 0.4,
        "vignette": 0.2,
    },
    FilterPreset.HORROR: {
        "contrast": 0.3,
        "saturation": -0.3,
        "exposure": -0.2,
        "gamma": 0.85,
        "temperature": -0.1,
        "tint": 0.1,
        "shadows": -0.2,
        "vignette": 0.6,
        "grain": 0.04,
    },
    # === Film ===
    FilterPreset.KODACHROME: {
        "contrast": 0.15,
        "saturation": 0.2,
        "vibrance": 0.15,
        "temperature": 0.15,
        "shadows": 0.1,
        "highlights": -0.05,
        "grain": 0.02,
    },
    FilterPreset.POLAROID: {
        "contrast": -0.1,
        "saturation": -0.15,
        "exposure": 0.1,
        "temperature": 0.2,
        "tint": -0.05,
        "highlights": 0.15,
        "vignette": 0.25,
        "grain": 0.03,
    },
    FilterPreset.VINTAGE_70S: {
        "contrast": -0.1,
        "saturation": -0.2,
        "exposure": 0.05,
        "temperature": 0.25,
        "tint": 0.1,
        "gamma": 1.1,
        "vignette": 0.35,
        "grain": 0.06,
        "sepia": 0.15,
    },
    FilterPreset.CROSS_PROCESS: {
        "contrast": 0.3,
        "saturation": 0.25,
        "temperature": -0.2,
        "tint": 0.15,
        "shadows": 0.2,
        "highlights": -0.1,
        "vignette": 0.2,
    },
    # === Portrait ===
    FilterPreset.SOFT_PORTRAIT: {
        "contrast": -0.1,
        "saturation": -0.05,
        "exposure": 0.1,
        "highlights": 0.1,
        "blur": 0.5,
        "sharpen": 0.2,
    },
    FilterPreset.WARM_PORTRAIT: {
        "saturation": 0.05,
        "temperature": 0.2,
        "tint": 0.05,
        "highlights": 0.1,
        "vignette": 0.15,
    },
    FilterPreset.COOL_PORTRAIT: {
        "saturation": -0.1,
        "temperature": -0.15,
        "tint": -0.05,
        "highlights": 0.15,
        "vignette": 0.15,
    },
    # === Landscape ===
    FilterPreset.VIVID_NATURE: {
        "contrast": 0.15,
        "saturation": 0.3,
        "vibrance": 0.25,
        "highlights": -0.1,
        "shadows": 0.1,
        "sharpen": 0.3,
    },
    FilterPreset.GOLDEN_HOUR: {
        "contrast": 0.1,
        "saturation": 0.15,
        "exposure": 0.1,
        "temperature": 0.35,
        "tint": 0.05,
        "highlights": 0.15,
        "vignette": 0.2,
    },
    FilterPreset.MISTY_MORNING: {
        "contrast": -0.2,
        "saturation": -0.25,
        "exposure": 0.15,
        "gamma": 1.15,
        "temperature": -0.1,
        "highlights": 0.2,
        "blur": 0.3,
    },
    # === Black & White ===
    FilterPreset.BW_HIGH_CONTRAST: {
        "saturation": -1.0,
        "contrast": 0.5,
        "gamma": 0.85,
        "shadows": -0.15,
        "highlights": 0.15,
    },
    FilterPreset.BW_FILM_NOIR: {
        "saturation": -1.0,
        "contrast": 0.35,
        "gamma": 0.9,
        "vignette": 0.5,
        "grain": 0.04,
    },
    FilterPreset.BW_SOFT: {
        "saturation": -1.0,
        "contrast": -0.1,
        "gamma": 1.1,
        "highlights": 0.1,
    },
    # === Mood ===
    FilterPreset.DREAMY: {
        "contrast": -0.15,
        "saturation": 0.1,
        "exposure": 0.15,
        "gamma": 1.15,
        "highlights": 0.2,
        "blur": 1.0,
        "vignette": 0.2,
    },
    FilterPreset.MOODY_BLUE: {
        "contrast": 0.15,
        "saturation": -0.15,
        "temperature": -0.3,
        "tint": 0.1,
        "shadows": -0.1,
        "vignette": 0.35,
    },
    FilterPreset.WARM_SUNSET: {
        "contrast": 0.1,
        "saturation": 0.2,
        "exposure": 0.05,
        "temperature": 0.4,
        "tint": 0.1,
        "highlights": 0.1,
        "vignette": 0.25,
    },
    # === Creative ===
    FilterPreset.NEON_NIGHTS: {
        "contrast": 0.35,
        "saturation": 0.5,
        "vibrance": 0.3,
        "temperature": -0.2,
        "tint": 0.2,
        "shadows": -0.1,
        "sharpen": 0.3,
        "vignette": 0.4,
    },
    FilterPreset.RETRO_GAMING: {
        "contrast": 0.2,
        "saturation": 0.3,
        "posterize_levels": 8,
        "sharpen": 0.5,
    },
    FilterPreset.DUOTONE_POP: {
        "contrast": 0.25,
        "saturation": -0.5,
        "temperature": 0.3,
        "tint": -0.2,
        "vignette": 0.2,
    },
}

# Add creative effect presets that use the new effects
CREATIVE_PRESETS: dict[FilterPreset, dict[str, float | int]] = {
    FilterPreset.NEON_NIGHTS: {
        "contrast": 0.35,
        "saturation": 0.5,
        "vibrance": 0.3,
        "temperature": -0.2,
        "tint": 0.2,
        "shadows": -0.1,
        "sharpen": 0.3,
        "vignette": 0.4,
        "chromatic_aberration": 3.0,
        "scanlines": 0.2,
    },
    FilterPreset.RETRO_GAMING: {
        "contrast": 0.2,
        "saturation": 0.3,
        "posterize_levels": 8,
        "pixelate": 4,
        "sharpen": 0.5,
        "scanlines": 0.3,
    },
    FilterPreset.VHS_TAPE: {
        "contrast": 0.1,
        "saturation": -0.2,
        "temperature": 0.1,
        "blur": 0.5,
        "chromatic_aberration": 5.0,
        "rgb_shift": 0.3,
        "scanlines": 0.4,
        "grain": 0.08,
        "noise": 0.1,
        "vignette": 0.25,
    },
    FilterPreset.GLITCH_ART: {
        "contrast": 0.2,
        "saturation": 0.2,
        "glitch": 0.7,
        "chromatic_aberration": 8.0,
        "rgb_shift": 0.5,
        "posterize_levels": 16,
    },
    FilterPreset.CYBERPUNK: {
        "contrast": 0.4,
        "saturation": 0.3,
        "temperature": -0.3,
        "tint": 0.3,
        "gamma": 0.9,
        "shadows": -0.15,
        "sharpen": 0.4,
        "chromatic_aberration": 4.0,
        "scanlines": 0.15,
        "vignette": 0.5,
        "grain": 0.02,
    },
    FilterPreset.COMIC_BOOK: {
        "contrast": 0.4,
        "saturation": 0.3,
        "posterize_levels": 6,
        "edge_detect": 0.3,
        "sharpen": 0.6,
    },
    FilterPreset.SKETCH: {
        "saturation": -1.0,
        "contrast": 0.3,
        "edge_detect": 0.8,
        "invert": 0.5,
        "gamma": 1.2,
    },
    FilterPreset.THERMAL: {
        # Note: thermal-like effect via color manipulation
        "invert": 1.0,
        "hue_shift": 0.6,
        "saturation": 0.5,
        "contrast": 0.4,
        "posterize_levels": 10,
        "temperature": 0.3,
    },
}

# Merge creative presets into main presets
PRESETS.update(CREATIVE_PRESETS)


def rgb_to_hsv(rgb: torch.Tensor) -> torch.Tensor:
    """Convert RGB tensor to HSV.

    Args:
        rgb: Tensor of shape (..., 3) with values in [0, 1]

    Returns:
        HSV tensor of same shape with H in [0, 1], S in [0, 1], V in [0, 1]
    """
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    max_c, _ = rgb.max(dim=-1)
    min_c, _ = rgb.min(dim=-1)
    diff = max_c - min_c

    # Value
    v = max_c

    # Saturation
    s = torch.where(max_c > 0, diff / (max_c + 1e-8), torch.zeros_like(max_c))

    # Hue
    h = torch.zeros_like(max_c)

    # Red is max
    mask_r = (max_c == r) & (diff > 0)
    h = torch.where(mask_r, ((g - b) / (diff + 1e-8)) % 6, h)

    # Green is max
    mask_g = (max_c == g) & (diff > 0)
    h = torch.where(mask_g, (b - r) / (diff + 1e-8) + 2, h)

    # Blue is max
    mask_b = (max_c == b) & (diff > 0)
    h = torch.where(mask_b, (r - g) / (diff + 1e-8) + 4, h)

    h = h / 6.0  # Normalize to [0, 1]
    h = h % 1.0  # Wrap around

    return torch.stack([h, s, v], dim=-1)


def hsv_to_rgb(hsv: torch.Tensor) -> torch.Tensor:
    """Convert HSV tensor to RGB.

    Args:
        hsv: Tensor of shape (..., 3) with H in [0, 1], S in [0, 1], V in [0, 1]

    Returns:
        RGB tensor of same shape with values in [0, 1]
    """
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    h = h * 6.0
    i = h.floor()
    f = h - i
    p = v * (1 - s)
    q = v * (1 - s * f)
    t = v * (1 - s * (1 - f))

    i = i.long() % 6

    # Build RGB based on which sextant we're in
    rgb = torch.zeros_like(hsv)

    mask0 = i == 0
    mask1 = i == 1
    mask2 = i == 2
    mask3 = i == 3
    mask4 = i == 4
    mask5 = i == 5

    rgb[..., 0] = torch.where(
        mask0,
        v,
        torch.where(mask1, q, torch.where(mask2 | mask3, p, torch.where(mask4, t, v))),
    )
    rgb[..., 1] = torch.where(
        mask0, t, torch.where(mask1 | mask2, v, torch.where(mask3, q, p))
    )
    rgb[..., 2] = torch.where(
        mask0 | mask1, p, torch.where(mask2, t, torch.where(mask3 | mask4, v, q))
    )

    return rgb


class ImageFilterPipeline(Pipeline):
    """Image filter pipeline that applies color grading and effects."""

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return ImageFilterConfig

    def __init__(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
        **kwargs,
    ):
        """Initialize the Image Filter pipeline.

        Args:
            device: Target device (defaults to CUDA if available)
            dtype: Data type for processing (default: float32 for quality)
            **kwargs: Additional parameters (height, width, preset, etc.)
                      These are passed by the pipeline manager from load_params.
        """
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.dtype = dtype

        # Build config from kwargs if provided
        config_kwargs = {}
        for field_name in ImageFilterConfig.model_fields:
            if field_name in kwargs:
                config_kwargs[field_name] = kwargs[field_name]

        self.config = (
            ImageFilterConfig(**config_kwargs) if config_kwargs else ImageFilterConfig()
        )
        logger.info(
            f"Image Filter pipeline initialized with preset: {self.config.preset.value}"
        )

    def prepare(self, **kwargs) -> Requirements:
        return Requirements(input_size=4)

    def _apply_brightness(self, img: torch.Tensor, amount: float) -> torch.Tensor:
        """Apply brightness adjustment."""
        if amount == 0:
            return img
        return (img + amount).clamp(0, 1)

    def _apply_contrast(self, img: torch.Tensor, amount: float) -> torch.Tensor:
        """Apply contrast adjustment."""
        if amount == 0:
            return img
        return ((img - 0.5) * (1.0 + amount) + 0.5).clamp(0, 1)

    def _apply_saturation(self, img: torch.Tensor, amount: float) -> torch.Tensor:
        """Apply saturation adjustment."""
        if amount == 0:
            return img
        # Luminance weights (Rec. 709)
        weights = torch.tensor(
            [0.2126, 0.7152, 0.0722], device=img.device, dtype=img.dtype
        )
        gray = (img * weights).sum(dim=-1, keepdim=True)
        return (gray + (img - gray) * (1.0 + amount)).clamp(0, 1)

    def _apply_exposure(self, img: torch.Tensor, amount: float) -> torch.Tensor:
        """Apply exposure adjustment in stops."""
        if amount == 0:
            return img
        return (img * (2.0**amount)).clamp(0, 1)

    def _apply_gamma(self, img: torch.Tensor, gamma: float) -> torch.Tensor:
        """Apply gamma correction."""
        if gamma == 1.0:
            return img
        # Avoid division by zero or negative values
        gamma = max(gamma, 0.01)
        return img.clamp(min=1e-8).pow(1.0 / gamma).clamp(0, 1)

    def _apply_hue_shift(self, img: torch.Tensor, amount: float) -> torch.Tensor:
        """Apply hue shift."""
        if amount == 0:
            return img
        hsv = rgb_to_hsv(img)
        hsv[..., 0] = (hsv[..., 0] + amount) % 1.0
        return hsv_to_rgb(hsv).clamp(0, 1)

    def _apply_temperature(self, img: torch.Tensor, amount: float) -> torch.Tensor:
        """Apply color temperature adjustment."""
        if amount == 0:
            return img
        result = img.clone()
        result[..., 0] = (result[..., 0] + amount * 0.3).clamp(0, 1)  # Red
        result[..., 2] = (result[..., 2] - amount * 0.3).clamp(0, 1)  # Blue
        return result

    def _apply_tint(self, img: torch.Tensor, amount: float) -> torch.Tensor:
        """Apply tint adjustment (green-magenta)."""
        if amount == 0:
            return img
        result = img.clone()
        result[..., 1] = (result[..., 1] + amount * 0.3).clamp(0, 1)  # Green
        result[..., 0] = (result[..., 0] - amount * 0.15).clamp(0, 1)  # Red
        result[..., 2] = (result[..., 2] - amount * 0.15).clamp(0, 1)  # Blue
        return result

    def _apply_vibrance(self, img: torch.Tensor, amount: float) -> torch.Tensor:
        """Apply vibrance (selective saturation for muted colors)."""
        if amount == 0:
            return img
        max_c, _ = img.max(dim=-1)
        min_c, _ = img.min(dim=-1)
        sat = max_c - min_c
        amt = amount * (1.0 - sat)

        weights = torch.tensor(
            [0.299, 0.587, 0.114], device=img.device, dtype=img.dtype
        )
        gray = (img * weights).sum(dim=-1, keepdim=True)

        return (gray + (img - gray) * (1.0 + amt.unsqueeze(-1))).clamp(0, 1)

    def _apply_highlights(self, img: torch.Tensor, amount: float) -> torch.Tensor:
        """Adjust highlight tones."""
        if amount == 0:
            return img
        weights = torch.tensor(
            [0.299, 0.587, 0.114], device=img.device, dtype=img.dtype
        )
        lum = (img * weights).sum(dim=-1, keepdim=True)
        mask = ((lum - 0.5) * 2).clamp(0, 1)
        return (img + amount * mask).clamp(0, 1)

    def _apply_shadows(self, img: torch.Tensor, amount: float) -> torch.Tensor:
        """Adjust shadow tones."""
        if amount == 0:
            return img
        weights = torch.tensor(
            [0.299, 0.587, 0.114], device=img.device, dtype=img.dtype
        )
        lum = (img * weights).sum(dim=-1, keepdim=True)
        mask = (1 - lum * 2).clamp(0, 1)
        return (img + amount * mask).clamp(0, 1)

    def _apply_sharpen(self, img: torch.Tensor, amount: float) -> torch.Tensor:
        """Apply unsharp mask sharpening."""
        if amount == 0:
            return img

        # Simple 3x3 blur kernel
        T, H, W, C = img.shape
        # Reshape for conv2d: (T*C, 1, H, W)
        img_conv = img.permute(0, 3, 1, 2).reshape(T * C, 1, H, W)

        # Box blur kernel
        kernel = torch.ones(1, 1, 3, 3, device=img.device, dtype=img.dtype) / 9.0

        # Apply blur with padding
        blurred = torch.nn.functional.conv2d(img_conv, kernel, padding=1)
        blurred = blurred.reshape(T, C, H, W).permute(0, 2, 3, 1)

        # Unsharp mask
        diff = img - blurred
        return (img + diff * amount).clamp(0, 1)

    def _apply_blur(self, img: torch.Tensor, radius: float) -> torch.Tensor:
        """Apply Gaussian blur."""
        if radius <= 0:
            return img

        T, H, W, C = img.shape
        # Reshape for conv2d: (T*C, 1, H, W)
        img_conv = img.permute(0, 3, 1, 2).reshape(T * C, 1, H, W)

        # Create Gaussian kernel
        kernel_size = int(radius * 2) * 2 + 1  # Ensure odd
        kernel_size = max(3, min(kernel_size, 31))  # Clamp to reasonable size

        # Create 1D Gaussian
        sigma = radius
        x = (
            torch.arange(kernel_size, device=img.device, dtype=img.dtype)
            - kernel_size // 2
        )
        gauss_1d = torch.exp(-(x**2) / (2 * sigma**2 + 1e-8))
        gauss_1d = gauss_1d / gauss_1d.sum()

        # Create 2D kernel
        kernel = gauss_1d.view(-1, 1) @ gauss_1d.view(1, -1)
        kernel = kernel.view(1, 1, kernel_size, kernel_size)

        # Apply blur
        padding = kernel_size // 2
        blurred = torch.nn.functional.conv2d(img_conv, kernel, padding=padding)
        return blurred.reshape(T, C, H, W).permute(0, 2, 3, 1)

    def _apply_vignette(self, img: torch.Tensor, amount: float) -> torch.Tensor:
        """Apply vignette effect."""
        if amount == 0:
            return img

        T, H, W, C = img.shape

        # Create coordinate grids
        y = torch.linspace(-1, 1, H, device=img.device, dtype=img.dtype)
        x = torch.linspace(-1, 1, W, device=img.device, dtype=img.dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")

        # Calculate distance from center
        dist = torch.sqrt(xx**2 + yy**2)

        # Vignette falloff
        softness = 0.3
        vig = 1 - ((dist - (1 - softness)) / softness * amount).clamp(0, 1)
        vig = vig.unsqueeze(0).unsqueeze(-1)  # (1, H, W, 1)

        return img * vig

    def _apply_grain(self, img: torch.Tensor, amount: float) -> torch.Tensor:
        """Apply film grain effect."""
        if amount == 0:
            return img

        noise = torch.rand_like(img) * 2 - 1  # Range [-1, 1]
        return (img + noise * amount).clamp(0, 1)

    def _apply_sepia(self, img: torch.Tensor, amount: float) -> torch.Tensor:
        """Apply sepia tone effect."""
        if amount == 0:
            return img

        r, g, b = img[..., 0], img[..., 1], img[..., 2]

        sepia_r = r * 0.393 + g * 0.769 + b * 0.189
        sepia_g = r * 0.349 + g * 0.686 + b * 0.168
        sepia_b = r * 0.272 + g * 0.534 + b * 0.131

        sepia = torch.stack([sepia_r, sepia_g, sepia_b], dim=-1).clamp(0, 1)

        return img * (1 - amount) + sepia * amount

    def _apply_invert(self, img: torch.Tensor, amount: float) -> torch.Tensor:
        """Apply color inversion."""
        if amount == 0:
            return img

        inverted = 1.0 - img
        return img * (1 - amount) + inverted * amount

    def _apply_posterize(self, img: torch.Tensor, levels: int) -> torch.Tensor:
        """Apply posterization effect."""
        if levels <= 1:
            return img

        return (img * levels).floor() / (levels - 1)

    def _apply_chromatic_aberration(
        self, img: torch.Tensor, amount: float
    ) -> torch.Tensor:
        """Apply chromatic aberration (RGB channel separation)."""
        if amount == 0:
            return img

        T, H, W, C = img.shape
        shift = int(amount)
        if shift == 0:
            return img

        result = img.clone()
        # Shift red channel right, blue channel left
        result[..., 0] = torch.roll(img[..., 0], shift, dims=2)
        result[..., 2] = torch.roll(img[..., 2], -shift, dims=2)
        return result

    def _apply_glitch(self, img: torch.Tensor, amount: float) -> torch.Tensor:
        """Apply glitch effect with horizontal band displacement."""
        if amount == 0:
            return img

        T, H, W, C = img.shape
        result = img.clone()

        # Create horizontal bands with random displacement
        num_bands = 20
        band_height = H // num_bands

        for i in range(num_bands):
            # Deterministic "random" based on band index
            rnd = ((i * 12.9898 + 78.233) * 43758.5453) % 1.0

            if rnd > 0.7:  # Only affect some bands
                y_start = i * band_height
                y_end = min((i + 1) * band_height, H)

                shift = int((rnd - 0.5) * amount * W * 0.2)
                if rnd > 0.9:
                    shift *= 3

                if shift != 0:
                    # Chromatic aberration within the band
                    result[:, y_start:y_end, :, 0] = torch.roll(
                        img[:, y_start:y_end, :, 0], shift, dims=2
                    )
                    result[:, y_start:y_end, :, 2] = torch.roll(
                        img[:, y_start:y_end, :, 2], -shift, dims=2
                    )

        return result

    def _apply_pixelate(self, img: torch.Tensor, block_size: int) -> torch.Tensor:
        """Apply pixelation effect."""
        if block_size <= 1:
            return img

        T, H, W, C = img.shape

        # Downsample then upsample
        new_h = max(1, H // block_size)
        new_w = max(1, W // block_size)

        # Reshape for interpolation: (T, C, H, W)
        img_nchw = img.permute(0, 3, 1, 2)

        # Downsample
        small = torch.nn.functional.interpolate(
            img_nchw, size=(new_h, new_w), mode="nearest"
        )
        # Upsample back
        pixelated = torch.nn.functional.interpolate(small, size=(H, W), mode="nearest")

        return pixelated.permute(0, 2, 3, 1)

    def _apply_emboss(self, img: torch.Tensor, amount: float) -> torch.Tensor:
        """Apply emboss/relief effect."""
        if amount == 0:
            return img

        T, H, W, C = img.shape
        img_conv = img.permute(0, 3, 1, 2).reshape(T * C, 1, H, W)

        # Emboss kernel
        kernel = torch.tensor(
            [[-2, -1, 0], [-1, 1, 1], [0, 1, 2]],
            device=img.device,
            dtype=img.dtype,
        ).view(1, 1, 3, 3)

        embossed = torch.nn.functional.conv2d(img_conv, kernel, padding=1)
        embossed = embossed.reshape(T, C, H, W).permute(0, 2, 3, 1)

        # Normalize and blend
        embossed = (embossed + 0.5).clamp(0, 1)
        return img * (1 - amount) + embossed * amount

    def _apply_edge_detect(self, img: torch.Tensor, amount: float) -> torch.Tensor:
        """Apply edge detection (Sobel)."""
        if amount == 0:
            return img

        T, H, W, C = img.shape

        # Convert to grayscale for edge detection
        weights = torch.tensor(
            [0.299, 0.587, 0.114], device=img.device, dtype=img.dtype
        )
        gray = (img * weights).sum(dim=-1, keepdim=True)
        gray_conv = gray.permute(0, 3, 1, 2)  # (T, 1, H, W)

        # Sobel kernels
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            device=img.device,
            dtype=img.dtype,
        ).view(1, 1, 3, 3)

        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            device=img.device,
            dtype=img.dtype,
        ).view(1, 1, 3, 3)

        edge_x = torch.nn.functional.conv2d(gray_conv, sobel_x, padding=1)
        edge_y = torch.nn.functional.conv2d(gray_conv, sobel_y, padding=1)

        edges = torch.sqrt(edge_x**2 + edge_y**2)
        edges = edges.permute(0, 2, 3, 1).expand(-1, -1, -1, 3)
        edges = edges.clamp(0, 1)

        return img * (1 - amount) + edges * amount

    def _apply_halftone(self, img: torch.Tensor, dot_size: int) -> torch.Tensor:
        """Apply halftone dot pattern effect."""
        if dot_size <= 1:
            return img

        T, H, W, C = img.shape

        # Convert to grayscale
        weights = torch.tensor(
            [0.299, 0.587, 0.114], device=img.device, dtype=img.dtype
        )
        gray = (img * weights).sum(dim=-1)  # (T, H, W)

        # Create halftone pattern
        halftone = torch.zeros_like(gray)

        for y in range(0, H, dot_size):
            for x in range(0, W, dot_size):
                # Get average luminance in this cell
                y_end = min(y + dot_size, H)
                x_end = min(x + dot_size, W)
                cell = gray[:, y:y_end, x:x_end]
                avg_lum = cell.mean(dim=(1, 2), keepdim=True)

                # Draw dot based on luminance (inverted: dark = big dot)
                radius = ((1 - avg_lum) * dot_size / 2).squeeze()
                cy, cx = dot_size // 2, dot_size // 2

                for dy in range(y_end - y):
                    for dx in range(x_end - x):
                        dist = ((dy - cy) ** 2 + (dx - cx) ** 2) ** 0.5
                        if dist <= radius.max():
                            halftone[:, y + dy, x + dx] = (dist <= radius).float()

        return halftone.unsqueeze(-1).expand(-1, -1, -1, 3)

    def _apply_scanlines(self, img: torch.Tensor, amount: float) -> torch.Tensor:
        """Apply CRT scanline effect."""
        if amount == 0:
            return img

        T, H, W, C = img.shape

        # Create scanline pattern (darken every other row)
        scanlines = torch.ones(H, device=img.device, dtype=img.dtype)
        scanlines[1::2] = 1 - amount * 0.5  # Darken odd rows

        scanlines = scanlines.view(1, H, 1, 1).expand(T, -1, W, C)
        return img * scanlines

    def _apply_rgb_shift(self, img: torch.Tensor, amount: float) -> torch.Tensor:
        """Apply vertical RGB channel shift."""
        if amount == 0:
            return img

        T, H, W, C = img.shape
        shift = int(amount * H * 0.05)  # Max 5% of height
        if shift == 0:
            return img

        result = img.clone()
        result[..., 0] = torch.roll(img[..., 0], shift, dims=1)  # Red up/down
        result[..., 2] = torch.roll(img[..., 2], -shift, dims=1)  # Blue opposite
        return result

    def _apply_mirror(self, img: torch.Tensor, amount: float) -> torch.Tensor:
        """Apply horizontal mirror/kaleidoscope effect."""
        if amount == 0:
            return img

        T, H, W, C = img.shape
        mid = W // 2

        # Mirror left half to right
        mirrored = img.clone()
        mirrored[:, :, mid:, :] = torch.flip(img[:, :, :mid, :], dims=[2])

        return img * (1 - amount) + mirrored * amount

    def _apply_noise(self, img: torch.Tensor, amount: float) -> torch.Tensor:
        """Apply color noise overlay."""
        if amount == 0:
            return img

        # Color noise (different noise per channel)
        noise = torch.rand_like(img) * 2 - 1
        return (img + noise * amount * 0.3).clamp(0, 1)

    def _get_params(self, **kwargs) -> dict[str, float | int]:
        """Get filter parameters, applying preset if selected.

        Args:
            **kwargs: Runtime parameter overrides

        Returns:
            Dictionary of all filter parameters with values
        """
        # Check for preset in effect_layers JSON (from web component)
        effect_layers_str = kwargs.get("effect_layers", self.config.effect_layers)
        preset_from_layers = None
        if effect_layers_str and effect_layers_str not in ("[]", ""):
            try:
                import json

                effect_data = json.loads(effect_layers_str)
                if isinstance(effect_data, dict) and "preset" in effect_data:
                    preset_from_layers = effect_data.get("preset")
            except (json.JSONDecodeError, TypeError):
                pass

        # Use preset from effect_layers if available, otherwise fall back to preset field
        preset_value = preset_from_layers or kwargs.get("preset", self.config.preset)

        # Handle both enum and string values
        if isinstance(preset_value, str):
            try:
                preset = FilterPreset(preset_value)
            except ValueError:
                preset = FilterPreset.CUSTOM
        else:
            preset = preset_value

        # Default neutral values
        defaults = {
            "brightness": 0.0,
            "contrast": 0.0,
            "saturation": 0.0,
            "exposure": 0.0,
            "gamma": 1.0,
            "hue_shift": 0.0,
            "temperature": 0.0,
            "tint": 0.0,
            "vibrance": 0.0,
            "highlights": 0.0,
            "shadows": 0.0,
            "sharpen": 0.0,
            "blur": 0.0,
            "vignette": 0.0,
            "grain": 0.0,
            "sepia": 0.0,
            "invert": 0.0,
            "posterize_levels": 0,
            # Creative effects
            "chromatic_aberration": 0.0,
            "glitch": 0.0,
            "pixelate": 0,
            "emboss": 0.0,
            "edge_detect": 0.0,
            "halftone": 0,
            "scanlines": 0.0,
            "rgb_shift": 0.0,
            "mirror": 0.0,
            "noise": 0.0,
        }

        if preset != FilterPreset.CUSTOM and preset in PRESETS:
            # Use preset values, filling in defaults for unspecified params
            preset_params = PRESETS[preset]
            params = {**defaults, **preset_params}
        else:
            # Custom mode: use config values or runtime overrides
            params = {
                "brightness": kwargs.get("brightness", self.config.brightness),
                "contrast": kwargs.get("contrast", self.config.contrast),
                "saturation": kwargs.get("saturation", self.config.saturation),
                "exposure": kwargs.get("exposure", self.config.exposure),
                "gamma": kwargs.get("gamma", self.config.gamma),
                "hue_shift": kwargs.get("hue_shift", self.config.hue_shift),
                "temperature": kwargs.get("temperature", self.config.temperature),
                "tint": kwargs.get("tint", self.config.tint),
                "vibrance": kwargs.get("vibrance", self.config.vibrance),
                "highlights": kwargs.get("highlights", self.config.highlights),
                "shadows": kwargs.get("shadows", self.config.shadows),
                "sharpen": kwargs.get("sharpen", self.config.sharpen),
                "blur": kwargs.get("blur", self.config.blur),
                "vignette": kwargs.get("vignette", self.config.vignette),
                "grain": kwargs.get("grain", self.config.grain),
                "sepia": kwargs.get("sepia", self.config.sepia),
                "invert": kwargs.get("invert", self.config.invert),
                "posterize_levels": kwargs.get(
                    "posterize_levels", self.config.posterize_levels
                ),
                # Creative effects
                "chromatic_aberration": kwargs.get(
                    "chromatic_aberration", self.config.chromatic_aberration
                ),
                "glitch": kwargs.get("glitch", self.config.glitch),
                "pixelate": kwargs.get("pixelate", self.config.pixelate),
                "emboss": kwargs.get("emboss", self.config.emboss),
                "edge_detect": kwargs.get("edge_detect", self.config.edge_detect),
                "halftone": kwargs.get("halftone", self.config.halftone),
                "scanlines": kwargs.get("scanlines", self.config.scanlines),
                "rgb_shift": kwargs.get("rgb_shift", self.config.rgb_shift),
                "mirror": kwargs.get("mirror", self.config.mirror),
                "noise": kwargs.get("noise", self.config.noise),
            }

        return params

    def _apply_filters(self, img: torch.Tensor, **kwargs) -> torch.Tensor:
        """Apply all filters to the image.

        Args:
            img: Input tensor in THWC format with values in [0, 1]
            **kwargs: Filter parameters from config or runtime

        Returns:
            Filtered tensor in same format
        """
        # Get parameters (with preset applied if selected)
        p = self._get_params(**kwargs)

        # Apply filters in order (similar to photo editing workflow)
        # 1. Exposure/brightness/contrast first
        img = self._apply_exposure(img, p["exposure"])
        img = self._apply_brightness(img, p["brightness"])
        img = self._apply_contrast(img, p["contrast"])

        # 2. Tone adjustments
        img = self._apply_shadows(img, p["shadows"])
        img = self._apply_highlights(img, p["highlights"])
        img = self._apply_gamma(img, p["gamma"])

        # 3. Color adjustments
        img = self._apply_saturation(img, p["saturation"])
        img = self._apply_vibrance(img, p["vibrance"])
        img = self._apply_hue_shift(img, p["hue_shift"])
        img = self._apply_temperature(img, p["temperature"])
        img = self._apply_tint(img, p["tint"])

        # 4. Detail
        img = self._apply_blur(img, p["blur"])
        img = self._apply_sharpen(img, p["sharpen"])

        # 5. Stylistic effects
        img = self._apply_sepia(img, p["sepia"])
        img = self._apply_invert(img, p["invert"])
        img = self._apply_posterize(img, int(p["posterize_levels"]))

        # 6. Creative effects (applied after color grading)
        img = self._apply_pixelate(img, int(p["pixelate"]))
        img = self._apply_halftone(img, int(p["halftone"]))
        img = self._apply_emboss(img, p["emboss"])
        img = self._apply_edge_detect(img, p["edge_detect"])
        img = self._apply_mirror(img, p["mirror"])
        img = self._apply_chromatic_aberration(img, p["chromatic_aberration"])
        img = self._apply_rgb_shift(img, p["rgb_shift"])
        img = self._apply_glitch(img, p["glitch"])
        img = self._apply_scanlines(img, p["scanlines"])

        # 7. Final overlays
        img = self._apply_vignette(img, p["vignette"])
        img = self._apply_grain(img, p["grain"])
        img = self._apply_noise(img, p["noise"])

        return img

    def _apply_effect_layers(self, img: torch.Tensor, layers_json: str) -> torch.Tensor:
        """Apply effects from the effect_layers JSON configuration.

        Args:
            img: Input tensor in THWC format with values in [0, 1]
            layers_json: JSON string containing layer configuration
                Can be either:
                - Old format: array of layers directly
                - New format: {"preset": "...", "layers": [...]}

        Returns:
            Filtered tensor in same format
        """
        import json

        try:
            parsed = json.loads(layers_json) if layers_json else []
            # Handle new format with preset and layers
            if isinstance(parsed, dict):
                layers = parsed.get("layers", [])
            else:
                # Old format: direct array
                layers = parsed
        except json.JSONDecodeError:
            logger.warning("Invalid effect_layers JSON, skipping layer effects")
            return img

        if not layers or not isinstance(layers, list):
            return img

        # Map effect names to apply methods
        effect_map = {
            # Basic adjustments
            "brightness": lambda img, p: self._apply_brightness(
                img, p.get("amount", 0)
            ),
            "contrast": lambda img, p: self._apply_contrast(img, p.get("amount", 0)),
            "saturation": lambda img, p: self._apply_saturation(
                img, p.get("amount", 0)
            ),
            "exposure": lambda img, p: self._apply_exposure(img, p.get("amount", 0)),
            "gamma": lambda img, p: self._apply_gamma(img, p.get("amount", 1.0)),
            # Color
            "hue_shift": lambda img, p: self._apply_hue_shift(img, p.get("amount", 0)),
            "temperature": lambda img, p: self._apply_temperature(
                img, p.get("amount", 0)
            ),
            "tint": lambda img, p: self._apply_tint(img, p.get("amount", 0)),
            "vibrance": lambda img, p: self._apply_vibrance(img, p.get("amount", 0)),
            # Tone
            "highlights": lambda img, p: self._apply_highlights(
                img, p.get("amount", 0)
            ),
            "shadows": lambda img, p: self._apply_shadows(img, p.get("amount", 0)),
            # Detail
            "sharpen": lambda img, p: self._apply_sharpen(img, p.get("amount", 0)),
            "blur": lambda img, p: self._apply_blur(
                img, p.get("amount", p.get("radius", 0))
            ),
            # Stylistic
            "vignette": lambda img, p: self._apply_vignette(img, p.get("amount", 0)),
            "grain": lambda img, p: self._apply_grain(img, p.get("amount", 0)),
            "sepia": lambda img, p: self._apply_sepia(img, p.get("amount", 0)),
            "invert": lambda img, p: self._apply_invert(img, p.get("amount", 0)),
            "posterize_levels": lambda img, p: self._apply_posterize(
                img, int(p.get("amount", 0))
            ),
            # Creative
            "chromatic_aberration": lambda img, p: self._apply_chromatic_aberration(
                img, p.get("amount", 0)
            ),
            "glitch": lambda img, p: self._apply_glitch(img, p.get("amount", 0)),
            "pixelate": lambda img, p: self._apply_pixelate(
                img, int(p.get("amount", p.get("size", 0)))
            ),
            "edge_detect": lambda img, p: self._apply_edge_detect(
                img, p.get("amount", 0)
            ),
            "emboss": lambda img, p: self._apply_emboss(img, p.get("amount", 0)),
            "scanlines": lambda img, p: self._apply_scanlines(img, p.get("amount", 0)),
            "rgb_shift": lambda img, p: self._apply_rgb_shift(img, p.get("amount", 0)),
            "noise": lambda img, p: self._apply_noise(img, p.get("amount", 0)),
        }

        # Apply each enabled layer in order
        for layer in layers:
            if not layer.get("enabled", True):
                continue

            effect_id = layer.get("effect")
            params = layer.get("params", {})

            apply_fn = effect_map.get(effect_id)
            if apply_fn:
                img = apply_fn(img, params)
            else:
                logger.debug(f"Unknown effect in layer: {effect_id}")

        return img

    def __call__(self, **kwargs) -> dict:
        """Process video frames with image filters.

        Args:
            **kwargs: Pipeline parameters. Input video is passed with the "video" key.
                Additional filter parameters can override config values.

        Returns:
            Dictionary containing processed video tensor under "video" key.
        """
        input_video = kwargs.get("video")

        if input_video is None:
            raise ValueError("Input cannot be None for ImageFilterPipeline")

        if isinstance(input_video, list):
            # Normalize frame sizes to handle resolution changes
            input_video = normalize_frame_sizes(input_video)
            # Preprocess: convert list of frames to BCTHW tensor in [-1, 1] range
            input_video = preprocess_chunk(input_video, self.device, self.dtype)

        # Convert from BCTHW to THWC format
        # First convert to BTCHW, then use postprocess_chunk to get THWC [0, 1]
        input_btchw = rearrange(input_video, "B C T H W -> B T C H W")
        input_thwc = postprocess_chunk(input_btchw)  # Now in THWC [0, 1] range

        # Ensure we're in float32 for quality processing
        input_thwc = input_thwc.to(dtype=torch.float32)

        # Apply preset/slider filters first
        output = self._apply_filters(input_thwc, **kwargs)

        # Apply effect layers (from custom Web Component UI)
        effect_layers = kwargs.get("effect_layers", self.config.effect_layers)
        if effect_layers and effect_layers != "[]":
            output = self._apply_effect_layers(output, effect_layers)

        # Return THWC [0, 1] float format (same as postprocess_chunk output)
        return {"video": output}
