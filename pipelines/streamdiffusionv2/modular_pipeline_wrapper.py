"""Wrapper to expose CausalStreamInferencePipeline as a ModularPipeline for Modular Diffusers."""

import os
import time

import torch
from diffusers.modular_pipelines import ModularPipeline

from .vendor.causvid.models.wan.causal_stream_inference import (
    CausalStreamInferencePipeline,
)


class StreamDiffusionV2ModularPipeline(ModularPipeline):
    """Wrapper that exposes CausalStreamInferencePipeline as a ModularPipeline."""

    def __init__(self, config, device, dtype, model_dir):
        """
        Initialize the wrapper.

        Args:
            config: Configuration dictionary for the pipeline
            device: Device to run the pipeline on
            dtype: Data type for the pipeline
            model_dir: Directory containing the model files
        """
        # Create and initialize the stream pipeline
        self.stream = CausalStreamInferencePipeline(config, device).to(
            device=device, dtype=dtype
        )

        # Load the generator state dict
        start = time.time()
        state_dict = torch.load(
            os.path.join(model_dir, "StreamDiffusionV2/model.pt"),
            map_location="cpu",
        )["generator"]
        self.stream.generator.load_state_dict(state_dict, strict=True)
        print(f"Loaded diffusion state dict in {time.time() - start:.3f}s")

        self._execution_device_val = next(self.stream.generator.parameters()).device

    @property
    def _execution_device(self):
        """Return the execution device."""
        return self._execution_device_val

    @_execution_device.setter
    def _execution_device(self, value):
        """Set the execution device."""
        self._execution_device_val = value
