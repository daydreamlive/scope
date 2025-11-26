"""Multi-mode pipeline base class for declarative pipeline definitions.

This module provides the MultiModePipeline base class that centralizes
orchestration logic for pipelines supporting multiple input modes (text, video, etc.).
Pipelines declare their capabilities via class methods, and the base class handles
all execution logic.
"""

import logging
from abc import abstractmethod
from typing import Any

import torch
from diffusers.modular_pipelines import PipelineState

from .blending import EmbeddingBlender
from .components import ComponentsManager
from .defaults import (
    apply_mode_defaults_to_state,
    get_mode_config,
    resolve_input_mode,
)
from .helpers import initialize_state_from_config
from .interface import Pipeline, Requirements
from .process import postprocess_chunk

logger = logging.getLogger(__name__)


class MultiModePipeline(Pipeline):
    """Base class for multi-mode pipelines with declarative configuration.

    This class centralizes all orchestration logic that was previously duplicated
    across pipeline implementations. Pipelines inherit from this class and declare
    their capabilities via three class methods:

    - get_blocks(): Returns block graph (can be SequentialPipelineBlocks or AutoPipelineBlocks)
    - get_components(): Declares component requirements (generator, VAE, etc.)
    - get_defaults(): Specifies mode-specific default parameters

    The base class automatically handles:
    - Component initialization
    - Mode inference and detection
    - Block graph execution
    - State management
    - Auto-generation of prepare() method

    Block Graph Design:
    Pipelines can use either:
    1. Single unified workflow (SequentialPipelineBlocks) where blocks conditionally
       execute based on input presence (recommended, aligns with diffusers philosophy)
    2. Multiple workflows with routing (AutoPipelineBlocks) for more complex scenarios

    Example:
        class MyPipeline(MultiModePipeline):
            @classmethod
            def get_blocks(cls):
                return MyUnifiedWorkflow()

            @classmethod
            def get_components(cls) -> dict:
                return {
                    "generator": MyGenerator,
                    "text_encoder": MyTextEncoder,
                    "vae": {
                        "text": {"strategy": "text_vae"},
                        "video": {"strategy": "video_vae"},
                    },
                }

            @classmethod
            def get_defaults(cls) -> dict:
                return {
                    "text": {"denoising_steps": [1000, 750]},
                    "video": {"denoising_steps": [1000, 750], "input_size": 4},
                }
    """

    @classmethod
    @abstractmethod
    def get_blocks(cls):
        """Return block graph for pipeline execution.

        Can return either:
        - SequentialPipelineBlocks: Single unified workflow with conditional execution
        - AutoPipelineBlocks: Multiple workflows with automatic routing

        Returns:
            Block graph instance (SequentialPipelineBlocks or AutoPipelineBlocks)
        """
        pass

    @classmethod
    @abstractmethod
    def get_components(cls) -> dict:
        """Declare component requirements for this pipeline.

        Returns:
            Dictionary declaring components. Simple components use direct class
            references, mode-specific components use nested dicts with strategies.

        Example:
            {
                "generator": WanDiffusionWrapper,
                "text_encoder": WanTextEncoderWrapper,
                "vae": {
                    "text": {"strategy": "longlive"},
                    "video": {"strategy": "streamdiffusionv2_longlive_scaled"},
                },
            }
        """
        pass

    @classmethod
    @abstractmethod
    def get_defaults(cls) -> dict:
        """Specify mode-specific default parameters.

        Returns:
            Dictionary mapping mode names to their default configurations.
            Include "input_size" in a mode's config to signal video input requirement.

        Example:
            {
                "text": {
                    "denoising_steps": [1000, 750, 500, 250],
                    "resolution": {"height": 320, "width": 576},
                },
                "video": {
                    "denoising_steps": [1000, 750],
                    "resolution": {"height": 512, "width": 512},
                    "noise_scale": 0.7,
                    "input_size": 4,
                },
            }
        """
        pass

    def __init__(
        self,
        config: Any,
        generator: Any,
        text_encoder: Any,
        model_config: dict,
        device: torch.device,
        dtype: torch.dtype,
        vae_init_kwargs: dict | None = None,
    ):
        """Initialize multi-mode pipeline.

        Args:
            config: Configuration object with pipeline parameters
            generator: Diffusion model wrapper
            text_encoder: Text encoder wrapper
            model_config: Model configuration dictionary
            device: Target device for computation
            dtype: Target dtype for tensors
            vae_init_kwargs: Optional kwargs for VAE initialization
        """
        self.device = device
        self.dtype = dtype

        # Create components manager
        components_config = {}
        components_config.update(model_config)
        components_config["device"] = device
        components_config["dtype"] = dtype
        components_config["pipeline_class"] = self.__class__
        components_config["pipeline_name"] = self.__class__.get_schema()["id"]
        components_config["vae_init_kwargs"] = vae_init_kwargs or {}

        self.components = ComponentsManager(components_config)
        self.components.add("generator", generator)
        self.components.add("scheduler", generator.get_scheduler())
        self.components.add("text_encoder", text_encoder)

        embedding_blender = EmbeddingBlender(device=device, dtype=dtype)
        self.components.add("embedding_blender", embedding_blender)

        # Create unified block graph from AutoPipelineBlocks
        self.blocks = self.__class__.get_blocks()

        # Initialize state with native mode defaults
        self.state = PipelineState()
        native_mode_config = get_mode_config(self.__class__)
        initialize_state_from_config(self.state, config, native_mode_config)

        self.first_call = True
        self.last_mode = None  # Track last mode to detect mode changes

    def prepare(self, **kwargs) -> Requirements | None:
        """Determine input requirements for this generation call.

        Auto-generated from mode defaults. Infers which mode would be triggered
        based on kwargs, then checks if that mode requires video input.

        Args:
            **kwargs: Generation parameters that may include input_mode

        Returns:
            Requirements object if video input is needed, None otherwise
        """
        mode = self._infer_mode_from_kwargs(kwargs)
        defaults = self.__class__.get_defaults().get(mode, {})
        input_size = defaults.get("input_size")

        if input_size:
            return Requirements(input_size=input_size)
        return None

    def _infer_mode_from_kwargs(self, kwargs: dict) -> str:
        """Infer which mode would be triggered from kwargs.

        This checks for explicit input_mode parameter first, then falls back
        to the native mode from schema.

        Args:
            kwargs: Generation parameters

        Returns:
            Inferred mode name (e.g., "text", "video")
        """
        return resolve_input_mode(kwargs.get("input_mode"), kwargs, self.__class__)

    def __call__(self, **kwargs) -> torch.Tensor:
        """Execute pipeline with given parameters.

        Args:
            **kwargs: Generation parameters including prompts, mode, etc.

        Returns:
            Post-processed output tensor
        """
        # Detect mode to check for mode changes
        current_mode = resolve_input_mode(
            kwargs.get("input_mode"), kwargs, self.__class__
        )

        # Initialize cache on first call or when mode changes
        mode_changed = self.last_mode is not None and self.last_mode != current_mode
        if self.first_call or mode_changed:
            self.state.set("init_cache", True)
            self.first_call = False
        else:
            self.state.set("init_cache", False)

        self.last_mode = current_mode

        # Update state with kwargs
        for k, v in kwargs.items():
            self.state.set(k, v)

        # Clear transition from state if not provided
        if "transition" not in kwargs:
            self.state.set("transition", None)

        return self._execute(**kwargs)

    def _execute(self, **kwargs) -> torch.Tensor:
        """Execute pipeline blocks based on detected mode.

        This method:
        1. Resolves input mode
        2. Applies mode-specific defaults to state
        3. Stores detected mode in state for self-configuring blocks
        4. Executes block graph (blocks self-configure based on state)
        5. Post-processes output

        Args:
            **kwargs: Generation parameters

        Returns:
            Post-processed output tensor
        """
        mode = resolve_input_mode(self.state.get("input_mode"), kwargs, self.__class__)

        apply_mode_defaults_to_state(self.state, self.__class__, mode, kwargs)

        # Store detected mode in state for self-configuring blocks
        self.state.set("_detected_mode", mode)

        # Execute unified block graph (blocks conditionally execute based on input presence)
        _, state = self.blocks(self.components, self.state)

        return postprocess_chunk(state.values["output_video"])
