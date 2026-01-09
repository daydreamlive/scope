"""Control state dataclasses for the realtime control plane.

ControlState is the immediate control surface for the generator. It can be
populated by the PromptCompiler (from WorldState + StyleManifest) or directly
via the Dev Console.

CompiledPrompt is the output of the PromptCompiler - what gets sent to the
pipeline after translation through a StyleManifest.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class GenerationMode(Enum):
    """Video generation input mode."""

    T2V = "text_to_video"
    V2V = "video_to_video"


@dataclass
class CompiledPrompt:
    """Output of PromptCompiler - ready for pipeline consumption.

    Attributes:
        positive: List of prompt dicts [{"text": "...", "weight": 1.0}]
        negative: Negative prompt (NOTE: not consumed by Scope/KREA pipeline)
        lora_scales: List of LoRA scale dicts [{"path": "...", "scale": 0.8}]
    """

    positive: list[dict] = field(default_factory=list)
    negative: str = ""
    lora_scales: list[dict] = field(default_factory=list)


@dataclass
class ControlState:
    """Immediate control surface for the generator.

    This is the state that directly maps to pipeline kwargs. It can be:
    - Populated by PromptCompiler from WorldState + StyleManifest
    - Directly overridden via Dev Console for prompt iteration
    """

    # Prompts (output of PromptCompiler, or direct override)
    # Shape: [{"text": "...", "weight": 1.0}]
    prompts: list[dict] = field(default_factory=list)

    # NOTE: The current Scope/KREA realtime pipeline does not consume negative prompts.
    # Keep this field for forward-compatibility with other backends.
    negative_prompt: str = ""

    # LoRA configuration (runtime updates via lora_scales; edge-triggered)
    # Shape: [{"path": "...", "scale": 0.8}]
    lora_scales: list[dict] = field(default_factory=list)

    # Generation parameters
    mode: GenerationMode = GenerationMode.T2V
    num_frame_per_block: int = 3  # Must match pipeline config
    denoising_step_list: list[int] = field(
        default_factory=lambda: [1000, 750, 500, 250]
    )

    # Determinism
    base_seed: int = 42
    branch_seed_offset: int = 0  # For deterministic branching

    # KV cache behavior (0.3 is KREA default - higher = more stable, less responsive)
    kv_cache_attention_bias: float = 0.3

    # Prompt transitions (pipeline-native; optional)
    # Shape matches Scope's `transition` contract:
    # {"target_prompts": [...], "num_steps": 4, "temporal_interpolation_method": "linear"}
    transition: Optional[dict] = None

    # Pipeline state tracking
    current_start_frame: int = 0

    def effective_seed(self) -> int:
        """Compute the effective seed including branch offset."""
        return self.base_seed + self.branch_seed_offset

    def to_pipeline_kwargs(self) -> dict:
        """Convert to kwargs for pipeline call.

        NOTE: This produces the BASE kwargs. The PipelineAdapter is responsible
        for:
        - Adding `init_cache` (driver decides)
        - Edge-triggering `lora_scales` (only when changed)
        - NOT including `negative_prompt` (not consumed by Scope/KREA)
        """
        kwargs = {
            "prompts": self.prompts,
            "num_frame_per_block": self.num_frame_per_block,
            "denoising_step_list": self.denoising_step_list,
            "base_seed": self.effective_seed(),
            "kv_cache_attention_bias": self.kv_cache_attention_bias,
        }
        # Include transition if present
        if self.transition is not None:
            kwargs["transition"] = self.transition
        return kwargs
