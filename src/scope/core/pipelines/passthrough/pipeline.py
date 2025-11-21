import torch
from einops import rearrange

from ..defaults import GENERATION_MODE_VIDEO
from ..interface import Pipeline, PipelineDefaults, Requirements
from ..process import postprocess_chunk, preprocess_chunk


class PassthroughPipeline(Pipeline):
    """Passthrough pipeline for testing"""

    NATIVE_GENERATION_MODE = GENERATION_MODE_VIDEO

    @classmethod
    def get_defaults(cls) -> PipelineDefaults:
        """Return default parameters for Passthrough pipeline."""
        shared = {
            "denoising_steps": None,
            "resolution": {"height": 512, "width": 512},
            "manage_cache": False,
            "base_seed": 42,
            "noise_scale": None,
            "noise_controller": None,
        }
        return cls._build_defaults(shared=shared)

    def __init__(
        self,
        height: int = 512,
        width: int = 512,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.height = height
        self.width = width
        self.device = device if device is not None else torch.device("cuda")
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
            input = preprocess_chunk(
                input, self.device, self.dtype, height=self.height, width=self.width
            )

        input = rearrange(input, "B C T H W -> B T C H W")

        return postprocess_chunk(input)
