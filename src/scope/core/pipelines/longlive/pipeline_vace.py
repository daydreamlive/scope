"""LongLive VACE Pipeline - Reference Image Conditioning."""

import logging
from typing import TYPE_CHECKING

import torch

from ..schema import LongLiveVaceConfig
from .pipeline import LongLivePipeline

if TYPE_CHECKING:
    from ..schema import BasePipelineConfig

logger = logging.getLogger(__name__)


class LongLiveVacePipeline(LongLivePipeline):
    """LongLive pipeline with VACE support for reference image conditioning.

    This pipeline extends LongLivePipeline to always load VACE weights and
    handle reference image conditioning.
    """

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return LongLiveVaceConfig

    def __init__(
        self,
        config,
        quantization=None,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """Initialize LongLive VACE pipeline.

        Args:
            config: LongLiveVaceConfig instance
            quantization: Optional quantization mode
            device: Target device
            dtype: Model dtype
        """
        # Ensure VACE path is set from config
        if not hasattr(config, "vace_path") or config.vace_path is None:
            # Try to auto-detect VACE checkpoint
            from scope.core.config import get_models_dir

            vace_model_path = get_models_dir() / "Wan2.1-VACE-1.3B"
            vace_checkpoint = vace_model_path / "diffusion_pytorch_model.safetensors"

            if vace_checkpoint.exists():
                config.vace_path = str(vace_checkpoint)
                logger.info(
                    f"LongLiveVacePipeline: Auto-detected VACE checkpoint at {vace_checkpoint}"
                )
            else:
                logger.warning(
                    f"LongLiveVacePipeline: VACE checkpoint not found at {vace_checkpoint}. "
                    "Pipeline will initialize but VACE conditioning will not be available. "
                    "Please download VACE weights to enable reference image conditioning."
                )

        # Initialize parent with VACE path
        super().__init__(config, quantization, device, dtype)

        logger.info("LongLiveVacePipeline: VACE pipeline initialized successfully")

    def __call__(self, **kwargs) -> torch.Tensor:
        """Generate video with optional reference image conditioning.

        Args:
            ref_images: Optional list of reference image paths
            vace_context_scale: Optional VACE context scaling factor (default: 1.0)
            **kwargs: Other generation parameters

        Returns:
            Generated video tensor
        """
        # Handle reference images if provided
        ref_images = kwargs.get("ref_images")
        vace_context_scale = kwargs.get("vace_context_scale", 1.0)

        logger.info(
            f"LongLiveVacePipeline.__call__: Received kwargs keys: {list(kwargs.keys())}"
        )

        if ref_images is not None:
            logger.info(
                f"LongLiveVacePipeline.__call__: Generating with {len(ref_images)} reference image(s), "
                f"context_scale={vace_context_scale}, paths={ref_images}"
            )
        else:
            logger.warning(
                "LongLiveVacePipeline.__call__: No ref_images provided in kwargs. "
                "VACE conditioning will not be applied."
            )

        return super().__call__(**kwargs)
