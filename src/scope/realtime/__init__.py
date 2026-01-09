"""Realtime control plane for video generation.

This module provides the control layer for the realtime video generation system,
separating control semantics from the underlying pipeline implementation.

Key components:
- ControlState: Immediate control surface for the generator
- ControlBus: Event queue with chunk-boundary semantics
- PipelineAdapter: Maps ControlState to pipeline kwargs, handles continuity
- GeneratorDriver: Tick loop that owns the pipeline and applies control events
"""

from scope.realtime.control_state import (
    CompiledPrompt,
    ControlState,
    GenerationMode,
)
from scope.realtime.control_bus import (
    ApplyMode,
    ControlBus,
    ControlEvent,
    EventType,
    pause_event,
    prompt_event,
    world_state_event,
)
from scope.realtime.pipeline_adapter import PipelineAdapter
from scope.realtime.generator_driver import (
    DriverState,
    GenerationResult,
    GeneratorDriver,
)
from scope.realtime.style_manifest import StyleManifest, StyleRegistry
from scope.realtime.world_state import (
    BeatType,
    CameraIntent,
    CharacterState,
    PropState,
    WorldState,
    create_simple_world,
    create_character_scene,
)
from scope.realtime.prompt_compiler import (
    CompiledPrompt as StyleCompiledPrompt,
    PromptCompiler,
    TemplateCompiler,
    LLMCompiler,
    CachedCompiler,
    InstructionSheet,
    create_compiler,
)
from scope.realtime.prompt_playlist import PromptPlaylist

__all__ = [
    # control_state
    "CompiledPrompt",
    "ControlState",
    "GenerationMode",
    # control_bus
    "ApplyMode",
    "ControlBus",
    "ControlEvent",
    "EventType",
    "pause_event",
    "prompt_event",
    "world_state_event",
    # pipeline_adapter
    "PipelineAdapter",
    # generator_driver
    "DriverState",
    "GenerationResult",
    "GeneratorDriver",
    # style_manifest
    "StyleManifest",
    "StyleRegistry",
    # world_state
    "BeatType",
    "CameraIntent",
    "CharacterState",
    "PropState",
    "WorldState",
    "create_simple_world",
    "create_character_scene",
    # prompt_compiler
    "StyleCompiledPrompt",
    "PromptCompiler",
    "TemplateCompiler",
    "LLMCompiler",
    "CachedCompiler",
    "InstructionSheet",
    "create_compiler",
    # prompt_playlist
    "PromptPlaylist",
]
