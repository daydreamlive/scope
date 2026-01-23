"""Event processing framework for discrete, async operations.

This module provides a base class for event-driven processors that
handle discrete operations asynchronously without blocking the main
pipeline.

Unlike frame processors (continuous, every frame), event processors
are triggered by discrete events like prompt changes or image uploads.
"""

from .processor import (
    EventProcessor,
    ProcessorConfig,
    ProcessorResult,
    ProcessorState,
)

__all__ = [
    "EventProcessor",
    "ProcessorConfig",
    "ProcessorResult",
    "ProcessorState",
]
