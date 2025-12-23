"""Pipeline interfaces for Scope worker plugins.

This module provides the base classes needed for creating Scope pipeline plugins.
"""

from .interface import Pipeline, Requirements
from .schema import BasePipelineConfig, InputMode, ModeDefaults

__all__ = [
    "Pipeline",
    "Requirements",
    "BasePipelineConfig",
    "InputMode",
    "ModeDefaults",
]

