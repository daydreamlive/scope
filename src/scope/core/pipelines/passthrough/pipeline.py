from typing import TYPE_CHECKING

import torch
from einops import rearrange

from ..interface import Pipeline, Requirements
from ..process import postprocess_chunk, preprocess_chunk
from .schema import PassthroughConfig

if TYPE_CHECKING:
    from ..schema import BasePipelineConfig


class PassthroughPipeline(Pipeline):
    """Passthrough pipeline for testing"""

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return PassthroughConfig

    def __init__(
        self,
        height: int = 512,
        width: int = 512,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.height = height
        self.width = width
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.dtype = dtype
        self.prompts = None

    def prepare(self, **kwargs) -> Requirements:
        return Requirements(input_size=4)

    def __call__(
        self,
        **kwargs,
    ) -> torch.Tensor:
        input = kwargs.get("video")

        if input is None:
            raise ValueError("Input cannot be None for PassthroughPipeline")

        if isinstance(input, list):
            # Don't resize for passthrough - preserve original input resolution
            input = preprocess_chunk(input, self.device, self.dtype)

        input = rearrange(input, "B C T H W -> B T C H W")

        return postprocess_chunk(input)
