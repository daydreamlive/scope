"""Helper functions to load components using ComponentsManager."""

import os
import time

import torch
from diffusers.modular_pipelines.components_manager import ComponentsManager

from .vendor.causvid.models.wan.causal_stream_inference import (
    CausalStreamInferencePipeline,
)


class ComponentProvider:
    """Simple wrapper to provide component access from ComponentsManager to blocks."""

    def __init__(self, components_manager: ComponentsManager, component_name: str, collection: str = "streamdiffusionv2"):
        """
        Initialize the component provider.

        Args:
            components_manager: The ComponentsManager instance
            component_name: Name of the component to provide
            collection: Collection name for retrieving the component
        """
        self.components_manager = components_manager
        self.component_name = component_name
        self.collection = collection
        # Cache the component to avoid repeated lookups
        self._component = None

    @property
    def stream(self):
        """Provide access to the stream component."""
        if self._component is None:
            self._component = self.components_manager.get_one(
                name=self.component_name, collection=self.collection
            )
        return self._component


def load_stream_component(
    config,
    device,
    dtype,
    model_dir,
    components_manager: ComponentsManager,
    collection: str = "streamdiffusionv2",
) -> ComponentProvider:
    """
    Load the CausalStreamInferencePipeline and add it to ComponentsManager.

    Args:
        config: Configuration dictionary for the pipeline
        device: Device to run the pipeline on
        dtype: Data type for the pipeline
        model_dir: Directory containing the model files
        components_manager: ComponentsManager instance to add component to
        collection: Collection name for organizing components

    Returns:
        ComponentProvider: A provider that gives access to the stream component
    """
    # Check if component already exists in ComponentsManager
    try:
        existing = components_manager.get_one(name="stream", collection=collection)
        # Component exists, create provider for it
        print(f"Reusing existing stream component from collection '{collection}'")
        return ComponentProvider(components_manager, "stream", collection)
    except Exception:
        # Component doesn't exist, create and add it
        pass

    # Create and initialize the stream pipeline
    stream = CausalStreamInferencePipeline(config, device).to(
        device=device, dtype=dtype
    )

    # Load the generator state dict
    start = time.time()
    model_path = os.path.join(model_dir, "StreamDiffusionV2/model.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at {model_path}. "
            "Please ensure StreamDiffusionV2/model.pt exists in the model directory."
        )

    state_dict_data = torch.load(model_path, map_location="cpu")

    # Handle both dict with "generator" key and direct state dict
    if isinstance(state_dict_data, dict) and "generator" in state_dict_data:
        state_dict = state_dict_data["generator"]
    else:
        state_dict = state_dict_data

    stream.generator.load_state_dict(state_dict, strict=True)
    print(f"Loaded diffusion state dict in {time.time() - start:.3f}s")

    # Add component to ComponentsManager
    component_id = components_manager.add(
        "stream",
        stream,
        collection=collection,
    )
    print(f"Added stream component to ComponentsManager with ID: {component_id}")

    # Create and return provider
    return ComponentProvider(components_manager, "stream", collection)
