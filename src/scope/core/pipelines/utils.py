import json
import logging
import os
from pathlib import Path
from typing import Any, Callable

import torch
from omegaconf import OmegaConf
from safetensors.torch import load_file as load_safetensors

logger = logging.getLogger(__name__)

# Re-export enums for backwards compatibility
from .enums import Quantization as Quantization  # noqa: PLC0414
from .enums import VaeType as VaeType  # noqa: PLC0414


def compile_with_inductor_fallback(
    fn: Callable, *args: Any, eager_fallback: Callable | None = None, **kwargs: Any
) -> Callable:
    """Wrap ``torch.compile`` with a graceful eager fallback on ``InductorError``.

    ``torch.inductor`` uses a subprocess worker pool for kernel compilation.  On
    some hosting environments (e.g. fal.ai H100 workers) these subprocesses can
    fail due to resource limits (FD exhaustion, memory pressure, or a locked
    ``torch_compile_debug`` cache directory), raising ``SubprocException`` wrapped
    in ``InductorError``.

    This helper attempts ``torch.compile(fn, ...)`` and, if an ``InductorError``
    is raised *at call time* (i.e. during JIT compilation on the first forward
    pass), falls back to eager execution.  The actual inner exception message is
    logged so the underlying cause can be diagnosed.

    Args:
        fn: The callable to compile.
        *args: Positional arguments forwarded to ``torch.compile``.
        eager_fallback: Optional replacement to use when compilation fails.
            Defaults to the original ``fn`` (uncompiled eager execution).
        **kwargs: Keyword arguments forwarded to ``torch.compile``.

    Returns:
        A compiled function that transparently falls back to eager execution on
        ``InductorError`` / ``SubprocException``.
    """
    compiled_fn = torch.compile(fn, *args, **kwargs)
    _fallback = eager_fallback if eager_fallback is not None else fn
    _active: list[Callable] = [compiled_fn]  # mutable so the closure can swap it

    def _wrapper(*call_args: Any, **call_kwargs: Any) -> Any:
        try:
            return _active[0](*call_args, **call_kwargs)
        except Exception as exc:
            # torch._inductor.exc.InductorError wraps SubprocException.
            # Check by class name to avoid a hard import of the inductor module.
            exc_type = type(exc).__name__
            inner_exc = getattr(exc, "inner_exception", None)
            if exc_type == "InductorError" or (
                inner_exc is not None
                and "SubprocException" in type(inner_exc).__name__
            ):
                inner_msg = str(inner_exc) if inner_exc is not None else str(exc)
                logger.warning(
                    "torch.inductor SubprocException during compilation of %s — "
                    "falling back to eager execution.  Inner exception: %s",
                    getattr(fn, "__name__", repr(fn)),
                    inner_msg,
                )
                # Permanently swap to eager so subsequent calls skip compile entirely.
                _active[0] = _fallback
                return _fallback(*call_args, **call_kwargs)
            raise  # re-raise unrelated errors unchanged

    return _wrapper  # type: ignore[return-value]


def load_state_dict(weights_path: str) -> dict:
    """Load weights with automatic format detection."""
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights not found at: {weights_path}")

    if weights_path.endswith(".safetensors"):
        # Load from safetensors and convert keys
        state_dict = load_safetensors(weights_path)

    elif weights_path.endswith(".pth") or weights_path.endswith(".pt"):
        # Load from PyTorch format (assume already in correct format)
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)

    else:
        raise ValueError(
            f"Unsupported file format. Expected .safetensors, .pth, or .pt, got: {weights_path}"
        )

    return state_dict


def load_model_config(config, pipeline_file_path: str | Path) -> OmegaConf:
    """
    Load model configuration from config or auto-load from model.yaml.
    Args:
        config: Configuration object that may contain a model_config attribute
        pipeline_file_path: Path to the pipeline's __file__ (used to locate model.yaml)
    Returns:
        OmegaConf: The model configuration, either from config or loaded from model.yaml
    """
    model_config = getattr(config, "model_config", None)
    if not model_config:
        model_yaml_path = Path(pipeline_file_path).parent / "model.yaml"
        model_config = OmegaConf.load(model_yaml_path)
    return model_config


def validate_resolution(
    height: int,
    width: int,
    scale_factor: int,
) -> None:
    """
    Validate that resolution dimensions are divisible by the required scale factor.

    Args:
        height: Height of the resolution
        width: Width of the resolution
        scale_factor: The factor that both dimensions must be divisible by

    Raises:
        ValueError: If height or width is not divisible by scale_factor
    """
    if height % scale_factor != 0 or width % scale_factor != 0:
        adjusted_width = (width // scale_factor) * scale_factor
        adjusted_height = (height // scale_factor) * scale_factor
        raise ValueError(
            f"Invalid resolution {width}×{height}. "
            f"Both width and height must be divisible by {scale_factor} "
            f"Please adjust to a valid resolution, e.g., {adjusted_width}×{adjusted_height}."
        )


def parse_jsonl_prompts(file_path: str) -> list[list[str]]:
    """Parse and validate a JSONL file containing prompt sequences.

    Args:
        file_path: Path to the JSONL file

    Returns:
        List of prompt sequences (each sequence is a list of prompt strings)

    Raises:
        ValueError: If the file is invalid JSONL or doesn't follow the expected format
    """
    prompt_sequences = []
    path = Path(file_path)

    if not path.exists():
        raise ValueError(f"File not found: {file_path}")

    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            # Parse JSON
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at line {line_num}: {e}") from e

            # Validate structure
            if "prompts" not in data:
                raise ValueError(
                    f"Invalid format at line {line_num}: missing 'prompts' key"
                )

            prompts = data["prompts"]
            if not isinstance(prompts, list):
                raise ValueError(
                    f"Invalid format at line {line_num}: 'prompts' must be a list of strings"
                )

            for i, prompt in enumerate(prompts):
                if not isinstance(prompt, str):
                    raise ValueError(
                        f"Invalid format at line {line_num}: prompt at index {i} is not a string"
                    )

            prompt_sequences.append(prompts)

    if not prompt_sequences:
        raise ValueError(f"No valid prompt sequences found in {file_path}")

    return prompt_sequences


def print_statistics(latency_measures: list[float], fps_measures: list[float]) -> None:
    """Print performance statistics."""
    print("\n=== Performance Statistics ===")
    print(
        f"Latency - Avg: {sum(latency_measures) / len(latency_measures):.2f}s, "
        f"Max: {max(latency_measures):.2f}s, Min: {min(latency_measures):.2f}s"
    )
    print(
        f"FPS - Avg: {sum(fps_measures) / len(fps_measures):.2f}, "
        f"Max: {max(fps_measures):.2f}, Min: {min(fps_measures):.2f}"
    )
