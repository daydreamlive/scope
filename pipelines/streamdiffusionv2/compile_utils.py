"""
Utility module for applying torch.compile optimization to StreamDiffusion V2.

This module provides a simple interface to compile the transformer blocks
with sensible default settings based on the ComfyUI WanVideo implementation.
"""

import torch
from typing import Optional, Dict, Any


class CompileConfig:
    """Configuration for torch.compile optimization."""

    # Preset configurations
    PRESETS = {
        "default": {
            "backend": "inductor",
            "fullgraph": False,
            "mode": "default",
            "dynamic": True,
            "dynamo_cache_size_limit": 512,
            "dynamo_recompile_limit": 1024,  # Increased: reduces cache eviction
            "compile_transformer_blocks_only": True,
        },
        "fast": {
            "backend": "inductor",
            "fullgraph": False,
            "mode": "reduce-overhead",
            "dynamic": False,
            "dynamo_cache_size_limit": 256,  # Increased from 64
            "dynamo_recompile_limit": 512,  # Increased from 128
            "compile_transformer_blocks_only": True,
        },
        "aggressive": {
            "backend": "inductor",
            "fullgraph": False,
            "mode": "max-autotune",
            "dynamic": False,
            "dynamo_cache_size_limit": 512,  # Increased for better caching
            "dynamo_recompile_limit": 1024,  # Increased for better caching
            "compile_transformer_blocks_only": True,
        },
        "cudagraphs": {
            "backend": "cudagraphs",
            "fullgraph": False,
            "mode": "default",
            "dynamic": False,
            "dynamo_cache_size_limit": 256,  # Increased from 64
            "dynamo_recompile_limit": 512,  # Increased from 128
            "compile_transformer_blocks_only": True,
        },
        "streaming": {
            # Optimized for streaming workloads with varying chunk sizes
            "backend": "inductor",
            "fullgraph": False,
            "mode": "default",
            "dynamic": True,  # Handle varying chunk sizes without recompilation
            "dynamo_cache_size_limit": 512,
            "dynamo_recompile_limit": 1024,
            "compile_transformer_blocks_only": True,
        },
    }

    def __init__(self, preset: str = "default", **overrides):
        """
        Initialize compile configuration.

        Args:
            preset: One of "default", "fast", "aggressive", "cudagraphs", or "streaming"
            **overrides: Any settings to override from the preset
        """
        if preset not in self.PRESETS:
            raise ValueError(
                f"Unknown preset '{preset}'. Available: {list(self.PRESETS.keys())}"
            )

        self.config = self.PRESETS[preset].copy()
        self.config.update(overrides)

    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return self.config.copy()


def check_requirements(verbose: bool = True) -> bool:
    """
    Check if requirements for torch.compile are met.

    Args:
        verbose: Whether to print status messages

    Returns:
        True if all requirements are met, False otherwise
    """
    all_good = True

    # Check CUDA
    if not torch.cuda.is_available():
        if verbose:
            print("✗ CUDA not available")
        all_good = False
    elif verbose:
        print(f"✓ CUDA available")

    # Check PyTorch version
    torch_version = tuple(map(int, torch.__version__.split('.')[:2]))
    if torch_version < (2, 0):
        if verbose:
            print(f"✗ PyTorch {torch.__version__} < 2.0 (torch.compile not available)")
        all_good = False
    else:
        if verbose:
            print(f"✓ PyTorch {torch.__version__}")
            if torch_version < (2, 7):
                print("  ⚠ PyTorch >= 2.7.0 recommended for best stability")

    # Check Triton (optional but recommended for inductor backend)
    try:
        import triton
        if verbose:
            print(f"✓ Triton {triton.__version__}")
    except ImportError:
        if verbose:
            print("⚠ Triton not found (recommended for 'inductor' backend)")
            print("  Install with: pip install triton")
            print("  Or use 'cudagraphs' backend instead")

    return all_good


def compile_model(
    transformer: torch.nn.Module,
    config: Optional[CompileConfig] = None,
    verbose: bool = True
) -> torch.nn.Module:
    """
    Apply torch.compile to transformer model.

    Args:
        transformer: The CausalWanModel to compile
        config: CompileConfig instance (uses "default" preset if None)
        verbose: Whether to print progress messages

    Returns:
        The transformer with compiled blocks

    Example:
        >>> from pipelines.streamdiffusionv2.pipeline import StreamDiffusionV2Pipeline
        >>> from pipelines.streamdiffusionv2.compile_utils import compile_model
        >>>
        >>> pipeline = StreamDiffusionV2Pipeline(config)
        >>> pipeline.stream.generator.model = compile_model(
        ...     pipeline.stream.generator.model
        ... )
    """
    if config is None:
        config = CompileConfig("default")

    compile_args = config.to_dict()

    # Configure dynamo
    torch._dynamo.config.cache_size_limit = compile_args["dynamo_cache_size_limit"]

    try:
        if hasattr(torch, '_dynamo') and hasattr(torch._dynamo, 'config'):
            torch._dynamo.config.recompile_limit = compile_args["dynamo_recompile_limit"]
    except Exception as e:
        if verbose:
            print(f"Warning: Could not set recompile_limit: {e}")

    compile_transformer_blocks_only = compile_args["compile_transformer_blocks_only"]

    if compile_transformer_blocks_only:
        # Compile individual blocks
        num_blocks = len(transformer.blocks)
        if verbose:
            print(f"Compiling {num_blocks} transformer blocks...")
            print(f"  Backend: {compile_args['backend']}")
            print(f"  Mode: {compile_args['mode']}")

        for i, block in enumerate(transformer.blocks):
            # Handle already compiled blocks
            if hasattr(block, "_orig_mod"):
                block = block._orig_mod

            if verbose and (i % 5 == 0 or i == num_blocks - 1):
                print(f"  Compiling block {i+1}/{num_blocks}...")

            transformer.blocks[i] = torch.compile(
                block,
                fullgraph=compile_args["fullgraph"],
                dynamic=compile_args["dynamic"],
                backend=compile_args["backend"],
                mode=compile_args["mode"]
            )

        if verbose:
            print(f"✓ All {num_blocks} blocks compiled successfully!")
    else:
        # Compile entire transformer
        if verbose:
            print("Compiling entire transformer model...")
            print(f"  Backend: {compile_args['backend']}")
            print(f"  Mode: {compile_args['mode']}")

        transformer = torch.compile(
            transformer,
            fullgraph=compile_args["fullgraph"],
            dynamic=compile_args["dynamic"],
            backend=compile_args["backend"],
            mode=compile_args["mode"]
        )

        if verbose:
            print("✓ Transformer compiled successfully!")

    return transformer


def apply_to_pipeline(
    pipeline,
    preset: str = "default",
    verbose: bool = True,
    **overrides
):
    """
    Apply torch.compile to a StreamDiffusionV2Pipeline.

    This is a convenience function that compiles the transformer model
    in-place.

    Args:
        pipeline: StreamDiffusionV2Pipeline instance
        preset: Compilation preset ("default", "fast", "aggressive", or "cudagraphs")
        verbose: Whether to print progress messages
        **overrides: Any config settings to override

    Example:
        >>> from pipelines.streamdiffusionv2.pipeline import StreamDiffusionV2Pipeline
        >>> from pipelines.streamdiffusionv2.compile_utils import apply_to_pipeline
        >>>
        >>> pipeline = StreamDiffusionV2Pipeline(config)
        >>> apply_to_pipeline(pipeline, preset="fast")
        >>>
        >>> # Now use pipeline normally
        >>> output = pipeline(input_frames)
    """
    if verbose:
        print("="*80)
        print(f"Applying torch.compile to StreamDiffusion V2 pipeline")
        print(f"Preset: {preset}")
        if overrides:
            print(f"Overrides: {overrides}")
        print("="*80 + "\n")

        # Check requirements
        if not check_requirements(verbose=True):
            print("\n⚠ Warning: Some requirements not met. Compilation may fail.")
        print()

    # Create config and compile
    config = CompileConfig(preset, **overrides)
    pipeline.stream.generator.model = compile_model(
        pipeline.stream.generator.model,
        config,
        verbose=verbose
    )

    if verbose:
        print("\n" + "="*80)
        print("✓ Pipeline compilation complete!")
        print("Note: Actual compilation happens on first inference (lazy)")
        print("="*80)


# Convenience presets as functions
def compile_default(transformer, verbose: bool = True):
    """Apply default compilation settings."""
    return compile_model(transformer, CompileConfig("default"), verbose)


def compile_fast(transformer, verbose: bool = True):
    """Apply fast compilation settings (reduce-overhead mode)."""
    return compile_model(transformer, CompileConfig("fast"), verbose)


def compile_aggressive(transformer, verbose: bool = True):
    """Apply aggressive compilation settings (max-autotune mode)."""
    return compile_model(transformer, CompileConfig("aggressive"), verbose)


def compile_cudagraphs(transformer, verbose: bool = True):
    """Apply CUDA graphs compilation (no Triton required)."""
    return compile_model(transformer, CompileConfig("cudagraphs"), verbose)


def compile_streaming(transformer, verbose: bool = True):
    """Apply streaming-optimized compilation (handles varying chunk sizes)."""
    return compile_model(transformer, CompileConfig("streaming"), verbose)


def enable_recompilation_logging():
    """
    Enable verbose logging for torch.compile recompilations.

    This helps diagnose performance issues caused by excessive recompilation.
    Call this before running your pipeline to see when and why recompilations occur.

    Example:
        >>> from pipelines.streamdiffusionv2.compile_utils import enable_recompilation_logging
        >>> enable_recompilation_logging()
        >>> # Now run your pipeline and watch for recompilation messages
    """
    import logging

    # Enable dynamo verbose logging
    torch._dynamo.config.verbose = True
    torch._logging.set_logs(dynamo=logging.DEBUG)

    print("✓ Recompilation logging enabled")
    print("  You will see messages when torch.compile recompiles due to:")
    print("  - Shape changes")
    print("  - Cache limit hits")
    print("  - Control flow variations")
    print()


def get_recompilation_stats():
    """
    Get statistics about torch.compile cache usage and recompilations.

    Returns:
        dict: Statistics about dynamo cache usage
    """
    try:
        stats = {
            "cache_size_limit": torch._dynamo.config.cache_size_limit,
            "current_cache_size": len(torch._dynamo.convert_frame.compile_cache) if hasattr(torch._dynamo.convert_frame, 'compile_cache') else "N/A",
        }

        if hasattr(torch._dynamo, 'config') and hasattr(torch._dynamo.config, 'recompile_limit'):
            stats["recompile_limit"] = torch._dynamo.config.recompile_limit

        return stats
    except Exception as e:
        return {"error": str(e)}
