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
    get_pipeline_config,
    resolve_input_mode,
)
from .helpers import initialize_state_from_config
from .interface import Pipeline, Requirements
from .process import postprocess_chunk

logger = logging.getLogger(__name__)


class MultiModePipeline(Pipeline):
    """Base class for multi-mode pipelines with declarative configuration.

    This class centralizes orchestration logic for pipelines supporting multiple
    input modes (text-to-video, video-to-video). Pipelines declare their
    capabilities via class methods:

    - get_config_class(): Returns Pydantic config model class
    - get_blocks(): Returns single workflow with nested AutoPipelineBlocks
    - get_components(): Declares component requirements

    Architecture Pattern:
    Uses nested AutoPipelineBlocks for input-based routing (e.g., AutoPrepareLatentsBlock
    routes based on presence of 'video' input). MultiModePipeline complements this by:
    - Resolving mode from inputs and applying mode-specific defaults
    - Handling mode transitions and cache initialization
    - Providing configuration and schema management

    This eliminates duplicate workflows while maintaining clear separation of concerns.

    The unified WanVAE handles both text-to-video and video-to-video modes via
    the use_cache parameter at encode/decode time, eliminating the need for
    mode-specific VAE strategies.

    Example:
        class MyPipeline(MultiModePipeline):
            @classmethod
            def get_config_class(cls) -> type[BasePipelineConfig]:
                return MyPipelineConfig

            @classmethod
            def get_blocks(cls):
                return MyWorkflow()

            @classmethod
            def get_components(cls) -> dict:
                return {
                    "generator": MyGenerator,
                    "text_encoder": MyTextEncoder,
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
            Dictionary declaring components using direct class references.
            Pipelines typically declare generator and text_encoder.

        Example:
            {
                "generator": WanDiffusionWrapper,
                "text_encoder": WanTextEncoderWrapper,
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

        # Get pipeline config with defaults
        pipeline_config = get_pipeline_config(self.__class__)

        # Create components manager
        components_config = {}
        components_config.update(model_config)
        components_config["device"] = device
        components_config["dtype"] = dtype
        components_config["pipeline_class"] = self.__class__
        components_config["pipeline_name"] = pipeline_config.pipeline_id
        components_config["vae_init_kwargs"] = vae_init_kwargs or {}

        self.components = ComponentsManager(components_config)
        self.components.add("generator", generator)
        self.components.add("scheduler", generator.get_scheduler())
        self.components.add("text_encoder", text_encoder)

        embedding_blender = EmbeddingBlender(device=device, dtype=dtype)
        self.components.add("embedding_blender", embedding_blender)

        # Create unified block graph from AutoPipelineBlocks
        self.blocks = self.__class__.get_blocks()

        # Initialize state with pipeline defaults
        self.state = PipelineState()
        initialize_state_from_config(self.state, config, pipeline_config)

        self.first_call = True
        self.last_mode = None  # Track last mode to detect mode changes

    def prepare(self, **kwargs) -> Requirements | None:
        """Determine input requirements for this generation call.

        Checks if the pipeline requires video input by examining the config.

        Args:
            **kwargs: Generation parameters that may include input_mode

        Returns:
            Requirements object if video input is needed, None otherwise
        """
        pipeline_config = get_pipeline_config(self.__class__)

        if pipeline_config.input_size:
            return Requirements(input_size=pipeline_config.input_size)
        return None

    def _infer_mode_from_kwargs(self, kwargs: dict) -> str:
        """Infer which mode would be triggered from kwargs.

        Mode is inferred from presence of 'video' in kwargs.

        Args:
            kwargs: Generation parameters

        Returns:
            Inferred mode name (e.g., "text", "video")
        """
        return resolve_input_mode(kwargs)

    def __call__(self, **kwargs) -> torch.Tensor:
        """Execute pipeline with given parameters.

        Args:
            **kwargs: Generation parameters including prompts, mode, etc.

        Returns:
            Post-processed output tensor
        """
        # Detect mode based on video presence to check for mode changes
        current_mode = resolve_input_mode(kwargs)

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
        """Execute pipeline blocks based on resolved mode.

        This method:
        1. Resolves input mode from presence of 'video' in kwargs
        2. Applies mode-specific defaults to state
        3. Executes block graph (AutoBlocks route based on input presence)
        4. Post-processes output

        Args:
            **kwargs: Generation parameters

        Returns:
            Post-processed output tensor
        """
        mode = resolve_input_mode(kwargs)

        apply_mode_defaults_to_state(self.state, self.__class__, mode, kwargs)

        # Execute unified block graph (AutoBlocks route based on input presence)
        _, state = self.blocks(self.components, self.state)

        return postprocess_chunk(state.values["output_video"])
