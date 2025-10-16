"""Base interface for all pipelines."""

from abc import ABC, abstractmethod

import torch
from pydantic import BaseModel

# Parameters that are only used in prepare() and should not be passed to __call__()
PREPARE_ONLY_PARAMS = frozenset(
    {
        "prompt_interpolation_method",
        "reset_cache",
        "manage_cache",
    }
)


class Requirements(BaseModel):
    """Requirements for pipeline configuration."""

    input_size: int


class Pipeline(ABC):
    """Abstract base class for all pipelines."""

    @abstractmethod
    def prepare(self, should_prepare: bool = False, **kwargs) -> Requirements | None:
        """
        Prepare the pipeline and get requirements for the next processing chunk.

        Args:
            should_prepare: Whether to trigger preparation logic
            **kwargs: Additional parameters such as:
                - prompts: list[dict] | None - List of prompts with weights for blending
                - prompt_interpolation_method: str - Interpolation method (linear/slerp)
                - reset_cache: bool - Whether to reset internal caches
        """
        pass

    @abstractmethod
    def __call__(
        self, input: torch.Tensor | list[torch.Tensor] | None = None, **kwargs
    ) -> torch.Tensor:
        """
        Process a chunk of video frames.

        Args:
            input: A tensor in BCTHW format OR a list of frame tensors in THWC format (in [0, 255] range), or None
            **kwargs: Additional parameters

        Returns:
            A processed chunk tensor in THWC format and [0, 1] range
        """
        pass
