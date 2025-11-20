import torch
from einops import rearrange

from ..interface import Pipeline, Requirements
from ..process import postprocess_chunk, preprocess_chunk


class PassthroughPipeline(Pipeline):
    """Passthrough pipeline for testing"""

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
