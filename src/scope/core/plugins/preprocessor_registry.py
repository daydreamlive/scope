"""Registry for video preprocessors.

This module provides:
- PreprocessorContext: Unified context object for all preprocessors
- PreprocessorRegistry: Registry for managing available preprocessors

Preprocessors can be either:
- Video preprocessors: Transform video frames (e.g., optical flow, pose)
- Generator preprocessors: Generate frames from other inputs (e.g., controller input)

All preprocessors receive PreprocessorContext and return VACE-compatible tensors.
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from scope.core.pipelines.controller import CtrlInput

logger = logging.getLogger(__name__)


@dataclass
class PreprocessorContext:
    """Unified context for all preprocessors.

    Preprocessors receive this context and extract what they need:
    - Video preprocessors use video_frames
    - Generator preprocessors use ctrl_input
    - Hybrid preprocessors can use both

    All preprocessors must return tensors in VACE format:
    [1, C, F, H, W] in [-1, 1] range.

    Attributes:
        video_frames: Input video frames [B, C, F, H, W] in [0, 1] range.
                     None if no video input available.
        ctrl_input: Controller input (keyboard/mouse state).
                   None if no controller input available.
        target_height: Target output height in pixels.
        target_width: Target output width in pixels.
        num_frames: Number of frames to generate (for generators).
    """

    video_frames: torch.Tensor | None = None
    ctrl_input: "CtrlInput | None" = None
    target_height: int = 512
    target_width: int = 512
    num_frames: int = 12


@dataclass
class PreprocessorInfo:
    """Information about a registered preprocessor."""

    id: str
    name: str
    preprocessor_class: type
    instance: Any = None


class PreprocessorRegistry:
    """Registry for managing available video preprocessors."""

    _preprocessors: dict[str, PreprocessorInfo] = {}

    @classmethod
    def register(
        cls, preprocessor_id: str, name: str, preprocessor_class: type
    ) -> None:
        """Register a preprocessor class.

        Args:
            preprocessor_id: Unique identifier (e.g., "pose", "depth")
            name: Display name for UI (e.g., "Pose", "Depth")
            preprocessor_class: The preprocessor class to instantiate
        """
        if preprocessor_id in cls._preprocessors:
            logger.warning(
                f"Preprocessor '{preprocessor_id}' already registered, overwriting"
            )

        cls._preprocessors[preprocessor_id] = PreprocessorInfo(
            id=preprocessor_id,
            name=name,
            preprocessor_class=preprocessor_class,
        )
        logger.info(f"Registered preprocessor: {preprocessor_id} ({name})")

    @classmethod
    def get(cls, preprocessor_id: str) -> PreprocessorInfo | None:
        """Get preprocessor info by ID."""
        return cls._preprocessors.get(preprocessor_id)

    @classmethod
    def get_instance(cls, preprocessor_id: str) -> Any | None:
        """Get or create a preprocessor instance (lazy initialization)."""
        info = cls._preprocessors.get(preprocessor_id)
        if info is None:
            return None

        if info.instance is None:
            logger.info(f"Initializing preprocessor: {preprocessor_id}")
            info.instance = info.preprocessor_class()

        return info.instance

    @classmethod
    def list_preprocessors(cls) -> list[dict]:
        """Get list of all registered preprocessors for API."""
        return [
            {"id": info.id, "name": info.name} for info in cls._preprocessors.values()
        ]

    @classmethod
    def clear(cls) -> None:
        """Clear all registered preprocessors (for testing)."""
        cls._preprocessors = {}
