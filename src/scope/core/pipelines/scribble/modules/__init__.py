"""Contour inference model for scribble/line art extraction.

Based on VACE's ContourInference model optimized for realtime use.
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Residual block with instance normalization."""

    def __init__(self, in_features: int):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv_block(x)


class ContourInference(nn.Module):
    """Neural network for extracting contour/scribble from images.

    Architecture: Encoder-Decoder with residual blocks.
    - Initial 7x7 conv to 64 channels
    - 2x downsampling (64 -> 128 -> 256)
    - N residual blocks at 256 channels
    - 2x upsampling (256 -> 128 -> 64)
    - Output 7x7 conv to 1 channel with sigmoid

    Args:
        input_nc: Number of input channels (default: 3 for RGB)
        output_nc: Number of output channels (default: 1 for grayscale contour)
        n_residual_blocks: Number of residual blocks (default: 3 for faster inference)
        sigmoid: Whether to apply sigmoid to output (default: True)
    """

    def __init__(
        self,
        input_nc: int = 3,
        output_nc: int = 1,
        n_residual_blocks: int = 3,
        sigmoid: bool = True,
    ):
        super().__init__()

        # Initial convolution block: 3 -> 64 channels
        self.model0 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Downsampling: 64 -> 128 -> 256
        self.model1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Residual blocks at 256 channels
        self.model2 = nn.Sequential(
            *[ResidualBlock(256) for _ in range(n_residual_blocks)]
        )

        # Upsampling: 256 -> 128 -> 64
        self.model3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Output layer: 64 -> output_nc
        layers = [nn.ReflectionPad2d(3), nn.Conv2d(64, output_nc, 7)]
        if sigmoid:
            layers.append(nn.Sigmoid())
        self.model4 = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W) with values in [0, 1]

        Returns:
            Contour map of shape (B, 1, H, W) with values in [0, 1]
        """
        out = self.model0(x)
        out = self.model1(out)
        out = self.model2(out)
        out = self.model3(out)
        out = self.model4(out)
        return out


__all__ = ["ContourInference"]
