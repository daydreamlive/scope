"""Wrapper to expose CausalStreamInferencePipeline as a ModularPipeline for Modular Diffusers."""

import torch
from diffusers.modular_pipelines import ModularPipeline


class StreamDiffusionV2ModularPipeline(ModularPipeline):
    """Wrapper that exposes CausalStreamInferencePipeline as a ModularPipeline."""

    def __init__(self, stream):
        """
        Initialize the wrapper.

        Args:
            stream: CausalStreamInferencePipeline instance
        """
        self.stream = stream
        self._execution_device_val = next(stream.generator.parameters()).device

    @property
    def _execution_device(self):
        """Return the execution device."""
        return self._execution_device_val

    @_execution_device.setter
    def _execution_device(self, value):
        """Set the execution device."""
        self._execution_device_val = value
