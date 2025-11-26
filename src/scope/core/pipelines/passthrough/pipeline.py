import torch
from einops import rearrange

from ..defaults import INPUT_MODE_VIDEO
from ..helpers import build_pipeline_schema
from ..interface import Pipeline, Requirements
from ..process import postprocess_chunk, preprocess_chunk


class PassthroughPipeline(Pipeline):
    """Passthrough pipeline for testing"""

    @classmethod
    def get_schema(cls) -> dict:
        """Return schema for Passthrough pipeline."""
        return build_pipeline_schema(
            pipeline_id="passthrough",
            name="Passthrough",
            description="Simple passthrough pipeline for testing and debugging",
            native_mode=INPUT_MODE_VIDEO,
            shared={
                "denoising_steps": None,
                "resolution": {"height": 512, "width": 512},
                "manage_cache": False,
                "base_seed": 42,
                "noise_scale": None,
                "noise_controller": None,
            },
        )

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
