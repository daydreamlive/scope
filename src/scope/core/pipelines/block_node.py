"""Block node adapter for exposing diffusers ModularPipelineBlocks to the graph system.

Wraps individual pipeline blocks with typed port metadata so they can be
visualized as nodes in the Workflow Builder's subgraph view.
"""

from __future__ import annotations

import types
from typing import Any, Union, get_args, get_origin

from pydantic import BaseModel


class TypedPort(BaseModel):
    """A typed input or output port on a block node."""

    name: str
    type_hint: str
    required: bool = False
    description: str = ""


class BlockParameter(BaseModel):
    """A configurable parameter on a block node."""

    name: str
    type_hint: str
    default: Any = None
    description: str = ""
    required: bool = False
    min_value: float | None = None
    max_value: float | None = None
    step: float | None = None
    choices: list[str] | None = None


class BlockNodeSchema(BaseModel):
    """Schema for a single block node, exposed to the frontend."""

    block_id: str
    block_name: str
    description: str
    inputs: list[TypedPort]
    outputs: list[TypedPort]
    parameters: list[BlockParameter] = []
    components: list[str]
    parent_pipeline_id: str


def _serialize_type_hint(hint: Any) -> str:
    """Convert a Python type annotation to a readable string.

    Examples:
        torch.Tensor -> "Tensor"
        int -> "int"
        str | list[str] -> "str | list[str]"
        list[dict] -> "list[dict]"
        None -> "Any"
    """
    if hint is None:
        return "Any"

    # Handle UnionType (Python 3.10+ X | Y syntax) and typing.Union
    origin = get_origin(hint)
    if origin is Union or isinstance(hint, types.UnionType):
        args = get_args(hint)
        # Filter out NoneType for Optional
        non_none = [a for a in args if a is not type(None)]
        if not non_none:
            return "None"
        parts = [_serialize_type_hint(a) for a in non_none]
        return " | ".join(parts)

    # Handle generic types like list[str], dict[str, Any]
    if origin is not None:
        args = get_args(hint)
        origin_name = getattr(origin, "__name__", str(origin))
        if args:
            arg_strs = [_serialize_type_hint(a) for a in args]
            return f"{origin_name}[{', '.join(arg_strs)}]"
        return origin_name

    # Handle regular types
    name = getattr(hint, "__name__", None)
    if name:
        # Shorten common qualified names
        if name == "Tensor":
            return "Tensor"
        return name

    return str(hint)


_TENSOR_TYPE_NAMES = {"Tensor", "FloatTensor", "LongTensor"}

# InputParam names that are internal state, not user-facing parameters
_INTERNAL_PARAM_NAMES = {
    "init_cache",
    "conditioning_embeds_updated",
    "conditioning_embeds",
    "kv_cache",
    "crossattn_cache",
    "kv_bank",
    "recache_buffer",
    "start_frame",
    "generator",
    "latents",
    "video",
    "output_video",
    "vace_context",
    "vace_input_frames",
    "vace_input_masks",
    "vace_ref_images",
    "first_frame_image",
    "last_frame_image",
    "embeds_list",
    "embeds_weights",
}


def _is_tensor_type(type_str: str) -> bool:
    """Check if a type hint represents a pure tensor type.

    For union types like ``list[int] | Tensor``, returns False because
    the user-facing part (list[int]) is not a tensor.
    """
    parts = [p.strip() for p in type_str.split("|")]
    return all(any(t in part for t in _TENSOR_TYPE_NAMES) for part in parts)


def _is_user_parameter(inp: Any) -> bool:
    """Determine if an InputParam should be exposed as a configurable parameter."""
    type_str = _serialize_type_hint(inp.type_hint)

    if _is_tensor_type(type_str):
        return False

    name = inp.name
    if name in _INTERNAL_PARAM_NAMES:
        return False
    if name.startswith("current_") or name.startswith("_"):
        return False

    has_default = getattr(inp, "default", None) is not None
    is_required = getattr(inp, "required", False)
    return has_default or is_required


def _extract_default(inp: Any) -> Any:
    """Safely extract the default value from an InputParam."""
    default = getattr(inp, "default", None)
    if default is None:
        return None
    # Convert non-JSON-serializable types to their string representation
    if isinstance(default, (int, float, str, bool, list)):
        return default
    return str(default)


class BlockNode:
    """Adapter wrapping a diffusers ModularPipelineBlocks for graph visibility."""

    def __init__(
        self,
        block_id: str,
        block_name: str,
        block_class: type,
        parent_pipeline_id: str,
        parameter_overrides: dict[str, dict[str, Any]] | None = None,
        additional_parameters: list[BlockParameter] | None = None,
    ):
        self.block_id = block_id
        self.block_name = block_name
        self.block_class = block_class
        self.parent_pipeline_id = parent_pipeline_id
        self.parameter_overrides = parameter_overrides or {}
        self.additional_parameters = additional_parameters or []
        self._instance = block_class()

    def _extract_parameters(self) -> list[BlockParameter]:
        """Extract configurable parameters from block InputParams."""
        params = []
        for inp in self._instance.inputs:
            if not _is_user_parameter(inp):
                continue
            type_str = _serialize_type_hint(inp.type_hint)
            param = BlockParameter(
                name=inp.name,
                type_hint=type_str,
                default=_extract_default(inp),
                description=getattr(inp, "description", ""),
                required=getattr(inp, "required", False),
            )
            # Apply overrides (min/max/step/choices) from pipeline
            overrides = self.parameter_overrides.get(inp.name, {})
            if overrides:
                param = param.model_copy(update=overrides)
            params.append(param)

        # Add pipeline-declared additional parameters (e.g., VAE type)
        params.extend(self.additional_parameters)
        return params

    def get_schema(self) -> BlockNodeSchema:
        """Extract typed port information from the diffusers block."""
        inputs = []
        for inp in self._instance.inputs:
            inputs.append(
                TypedPort(
                    name=inp.name,
                    type_hint=_serialize_type_hint(inp.type_hint),
                    required=getattr(inp, "required", False),
                    description=getattr(inp, "description", ""),
                )
            )

        outputs = []
        for out in self._instance.intermediate_outputs:
            outputs.append(
                TypedPort(
                    name=out.name,
                    type_hint=_serialize_type_hint(out.type_hint),
                    description=getattr(out, "description", ""),
                )
            )

        components = []
        if hasattr(self._instance, "expected_components"):
            components = [c.name for c in self._instance.expected_components]

        description = ""
        if hasattr(self._instance, "description"):
            desc = self._instance.description
            if isinstance(desc, str):
                description = desc

        return BlockNodeSchema(
            block_id=self.block_id,
            block_name=self.block_name,
            description=description,
            inputs=inputs,
            outputs=outputs,
            parameters=self._extract_parameters(),
            components=components,
            parent_pipeline_id=self.parent_pipeline_id,
        )
