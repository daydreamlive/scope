"""PersonaLive plugin for Daydream Scope.

This plugin provides the PersonaLive portrait animation pipeline for real-time
portrait animation from reference images and driving video.

Based on: https://github.com/GVCLab/PersonaLive
"""

import scope.core
from .pipeline import PersonaLivePipeline


@scope.core.hookimpl
def register_pipelines(register):
    """Register the PersonaLive pipeline."""
    register(PersonaLivePipeline)


__all__ = ["PersonaLivePipeline"]
