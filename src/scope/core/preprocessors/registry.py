"""Preprocessor registry for centralized preprocessor management.

This module provides a registry pattern similar to pipelines for managing
available preprocessors. Preprocessors are pipelines that can be used to
preprocess video input before it enters the main diffusion pipeline.
"""

import importlib
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..pipelines.interface import Pipeline

logger = logging.getLogger(__name__)


class PreprocessorRegistry:
    """Registry for managing available preprocessors.

    Preprocessors are pipelines that implement the Pipeline interface
    and can be used to preprocess video input.
    """

    _preprocessors: dict[str, type["Pipeline"]] = {}

    @classmethod
    def register(cls, preprocessor_id: str, preprocessor_class: type["Pipeline"]) -> None:
        """Register a preprocessor class with its ID.

        Args:
            preprocessor_id: Unique identifier for the preprocessor
            preprocessor_class: Preprocessor class (must implement Pipeline interface)
        """
        cls._preprocessors[preprocessor_id] = preprocessor_class

    @classmethod
    def get(cls, preprocessor_id: str) -> type["Pipeline"] | None:
        """Get a preprocessor class by its ID.

        Args:
            preprocessor_id: Preprocessor identifier

        Returns:
            Preprocessor class if found, None otherwise
        """
        return cls._preprocessors.get(preprocessor_id)

    @classmethod
    def list_preprocessors(cls) -> list[str]:
        """Get list of all registered preprocessor IDs.

        Returns:
            List of preprocessor IDs
        """
        return list(cls._preprocessors.keys())


# Register all available preprocessors
def _register_preprocessors():
    """Register built-in preprocessors."""
    # Define preprocessor imports with their module paths and class names
    # Note: Preprocessors are actually pipelines that can be used as preprocessors
    # They need to implement the Pipeline interface
    preprocessor_configs = [
        ("depthanything", "scope.core.pipelines.depthanything.pipeline", "DepthAnythingPipeline"),
        ("passthrough", "scope.core.pipelines.passthrough.pipeline", "PassthroughPipeline"),
    ]

    # Try to import and register each preprocessor
    for preprocessor_id, module_path, class_name in preprocessor_configs:
        try:
            module = importlib.import_module(module_path)
            preprocessor_class = getattr(module, class_name)
            PreprocessorRegistry.register(preprocessor_id, preprocessor_class)
            logger.debug(f"Registered preprocessor: {preprocessor_id}")
        except ImportError as e:
            logger.warning(
                f"Failed to import preprocessor {preprocessor_id} from {module_path}: {e}. "
                f"This preprocessor will not be available."
            )
        except AttributeError as e:
            logger.warning(
                f"Preprocessor {preprocessor_id} does not have class {class_name} in {module_path}: {e}. "
                f"This preprocessor will not be available."
            )


def _initialize_registry():
    """Initialize registry with built-in preprocessors and sync with PipelineRegistry."""
    # Register built-in preprocessors first
    _register_preprocessors()

    # Sync plugin preprocessors from PipelineRegistry
    # Preprocessors are registered as pipelines via register_pipelines hook,
    # so we check PipelineRegistry for any pipelines that are also preprocessors
    try:
        from scope.core.pipelines.registry import PipelineRegistry

        # Get all registered pipelines
        all_pipelines = PipelineRegistry.list_pipelines()

        # Known preprocessor IDs (built-in + any plugin preprocessors)
        # Plugin preprocessors will be in PipelineRegistry if they were registered
        # We can check if they're already in our registry, and if not but they're
        # in PipelineRegistry, we can add them (though for now we only track built-ins)
        builtin_preprocessor_ids = {"depthanything", "passthrough"}

        # Check if any plugin pipelines are preprocessors
        # For now, we only auto-register built-ins. Plugin preprocessors would need
        # to be explicitly identified, or we could check PipelineRegistry.get() when needed
        for pipeline_id in all_pipelines:
            if pipeline_id in builtin_preprocessor_ids:
                # Already registered as built-in
                continue
            # Plugin preprocessors: could check PipelineRegistry here if needed
            # For now, PreprocessorRegistry only tracks built-ins explicitly

    except ImportError:
        # PipelineRegistry might not be available in all contexts
        logger.debug("PipelineRegistry not available, skipping sync")

    preprocessor_count = len(PreprocessorRegistry.list_preprocessors())
    logger.info(f"Preprocessor registry initialized with {preprocessor_count} preprocessor(s)")


# Auto-register preprocessors on module import
_initialize_registry()
