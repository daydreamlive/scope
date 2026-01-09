"""
PromptCompiler - Translates WorldState + StyleManifest into prompt strings.

The compiler is pluggable:
- LLMCompiler: Uses an LLM (Gemini Flash, etc.) with instruction sheets
- TemplateCompiler: Deterministic vocab substitution (for testing/fallback)
- CachedCompiler: Wraps another compiler with memoization

Factory function:
- create_compiler(): Creates the appropriate compiler based on config
"""

import hashlib
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .style_manifest import StyleManifest
from .world_state import WorldState

logger = logging.getLogger(__name__)


@dataclass
class PromptEntry:
    """A single prompt entry with text and weight."""

    text: str
    weight: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "weight": self.weight}


@dataclass
class LoRAScaleUpdate:
    """A single LoRA scale update in Scope runtime format."""

    path: str
    scale: float

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "scale": self.scale}


@dataclass
class CompiledPrompt:
    """Result of prompt compilation - pipeline-ready format."""

    # Pipeline-ready format: list of {"text": ..., "weight": ...}
    prompts: list[PromptEntry] = field(default_factory=list)
    negative_prompt: str = ""

    # LoRA configuration from style
    lora_scales: list[LoRAScaleUpdate] = field(default_factory=list)

    # Metadata about the compilation
    style_name: str = ""
    compiler_type: str = ""
    token_count: int | None = None
    truncated: bool = False

    # For debugging/iteration
    world_state_hash: str = ""
    manifest_hash: str = ""  # For cache invalidation
    raw_llm_response: str | None = None

    @property
    def prompt(self) -> str:
        """Convenience: return first prompt text (for simple cases)."""
        if self.prompts:
            return self.prompts[0].text
        return ""

    def to_pipeline_kwargs(self) -> dict[str, Any]:
        """Convert to kwargs suitable for pipeline call."""
        kwargs = {
            "prompts": [p.to_dict() for p in self.prompts],
        }
        if self.negative_prompt:
            kwargs["negative_prompt"] = self.negative_prompt
        if self.lora_scales:
            kwargs["lora_scales"] = [u.to_dict() for u in self.lora_scales]
        return kwargs


class PromptCompiler(ABC):
    """
    Abstract base class for prompt compilers.

    Subclasses implement the actual compilation logic.
    """

    @abstractmethod
    def compile(
        self,
        world_state: WorldState,
        style: StyleManifest,
        **kwargs: Any,
    ) -> CompiledPrompt:
        """
        Compile a WorldState into a prompt using the given style.

        Args:
            world_state: The world state to compile
            style: The style manifest with vocab and constraints
            **kwargs: Additional compiler-specific options

        Returns:
            CompiledPrompt with the generated prompt string
        """
        pass

    def _compute_state_hash(self, world_state: WorldState, style: StyleManifest) -> str:
        """Compute a hash for cache keys (WorldState only)."""
        state_str = world_state.model_dump_json()
        return hashlib.md5(state_str.encode()).hexdigest()[:12]

    def _compute_manifest_hash(self, style: StyleManifest) -> str:
        """Compute a hash for manifest content (for cache invalidation)."""
        # Include all vocab content, not just name
        manifest_str = style.model_dump_json()
        return hashlib.md5(manifest_str.encode()).hexdigest()[:12]

    def _compute_cache_key(self, world_state: WorldState, style: StyleManifest) -> str:
        """Compute full cache key including manifest content."""
        state_hash = self._compute_state_hash(world_state, style)
        manifest_hash = self._compute_manifest_hash(style)
        return f"{style.name}:{manifest_hash}:{state_hash}"


class TemplateCompiler(PromptCompiler):
    """
    Deterministic template-based compiler.

    Uses simple vocab substitution without LLM calls.
    Good for testing and as a fallback.
    """

    # Common action normalizations (user can extend via style.custom_vocab["action_aliases"])
    ACTION_ALIASES = {
        "walking": "walk",
        "running": "run",
        "standing": "idle",
        "sitting": "sit",
        "jumping": "jump",
    }

    def _normalize_action(self, action: str, style: StyleManifest) -> str:
        """Normalize action key to canonical form."""
        # Check style-specific aliases first
        aliases = style.custom_vocab.get("action_aliases", {})
        if action in aliases:
            return aliases[action]
        # Fall back to built-in aliases
        return self.ACTION_ALIASES.get(action, action)

    def compile(
        self,
        world_state: WorldState,
        style: StyleManifest,
        **kwargs: Any,
    ) -> CompiledPrompt:
        """Compile using template substitution."""
        parts = []

        # 1. Trigger words (always first)
        if style.trigger_words:
            parts.extend(style.trigger_words)

        # 2. Action/motion (with normalization)
        if world_state.action:
            action_key = self._normalize_action(world_state.action, style)
            motion = style.get_vocab("motion", action_key, world_state.action)
            parts.append(motion)

        # 3. Character emotions
        for char in world_state.characters:
            if char.emotion and char.emotion != "neutral":
                emotion = style.get_vocab("emotion", char.emotion, char.emotion)
                parts.append(f"{char.name} {emotion}")

        # 4. Camera
        camera_key = world_state.camera.value
        camera = style.get_vocab("camera", camera_key, camera_key)
        parts.append(camera)

        # 5. Lighting (based on time of day or mood)
        if world_state.time_of_day:
            lighting = style.get_vocab("lighting", world_state.time_of_day)
            if lighting != world_state.time_of_day:
                parts.append(lighting)

        # 6. Beat modifier
        beat_key = world_state.beat.value
        beat = style.get_vocab("beat", beat_key)
        if beat != beat_key:
            parts.append(beat)

        # 7. Scene description (truncated if needed)
        if world_state.scene_description:
            parts.append(world_state.scene_description)

        # Join and basic cleanup
        prompt_text = ", ".join(p for p in parts if p)

        # Build lora_scales from style if lora_path is set
        lora_scales: list[LoRAScaleUpdate] = []
        if style.lora_path:
            lora_scales.append(
                LoRAScaleUpdate(path=style.lora_path, scale=style.lora_default_scale)
            )

        return CompiledPrompt(
            prompts=[PromptEntry(text=prompt_text, weight=1.0)],
            negative_prompt=style.default_negative,
            lora_scales=lora_scales,
            style_name=style.name,
            compiler_type="template",
            world_state_hash=self._compute_state_hash(world_state, style),
            manifest_hash=self._compute_manifest_hash(style),
        )


@dataclass
class InstructionSheet:
    """
    An instruction sheet for LLM-based compilation.

    Contains the system prompt and few-shot examples.
    """

    name: str
    system_prompt: str
    examples: list[dict[str, str]] = field(default_factory=list)
    # Each example: {"world_state": "...", "output": "..."}

    @classmethod
    def from_markdown(cls, path: str | Path) -> "InstructionSheet":
        """
        Load an instruction sheet from a markdown file.

        Expected format:
        ```
        # Sheet Name

        ## System Prompt
        Your instructions here...

        ## Examples

        ### Example 1
        **Input:**
        WorldState description...

        **Output:**
        Expected prompt output...
        ```
        """
        path = Path(path)
        content = path.read_text()

        # Parse markdown (simplified parser)
        lines = content.split("\n")
        name = ""
        system_prompt = ""
        examples = []

        current_section = None
        current_example = {}
        buffer = []

        for line in lines:
            if line.startswith("# ") and not name:
                name = line[2:].strip()
            elif line.startswith("## System Prompt"):
                current_section = "system"
                buffer = []
            elif line.startswith("## Examples"):
                if buffer and current_section == "system":
                    system_prompt = "\n".join(buffer).strip()
                current_section = "examples"
                buffer = []
            elif line.startswith("### Example"):
                # Save previous example if complete
                if current_example and buffer and current_section == "output":
                    current_example["output"] = "\n".join(buffer).strip()
                if current_example and "world_state" in current_example and "output" in current_example:
                    examples.append(current_example)
                current_example = {}
                buffer = []
                current_section = "example_start"
            elif line.startswith("**Input:**"):
                buffer = []
                current_section = "input"
            elif line.startswith("**Output:**"):
                current_example["world_state"] = "\n".join(buffer).strip()
                buffer = []
                current_section = "output"
            elif current_section in ("system", "input", "output"):
                buffer.append(line)

        # Capture final content
        if buffer:
            if current_section == "system":
                system_prompt = "\n".join(buffer).strip()
            elif current_section == "output":
                current_example["output"] = "\n".join(buffer).strip()

        if current_example and "world_state" in current_example and "output" in current_example:
            examples.append(current_example)

        return cls(
            name=name or path.stem,
            system_prompt=system_prompt,
            examples=examples,
        )


class LLMCompiler(PromptCompiler):
    """
    LLM-based prompt compiler.

    Uses an LLM (via a callable) to translate WorldState into prompts,
    guided by instruction sheets and style vocab.
    """

    def __init__(
        self,
        llm_callable: callable,
        instruction_sheet: InstructionSheet | None = None,
    ):
        """
        Args:
            llm_callable: A function that takes (system_prompt, user_message) -> str
            instruction_sheet: Optional instruction sheet with system prompt and examples
        """
        self.llm_callable = llm_callable
        self.instruction_sheet = instruction_sheet

    def compile(
        self,
        world_state: WorldState,
        style: StyleManifest,
        **kwargs: Any,
    ) -> CompiledPrompt:
        """Compile using LLM."""
        # Build system prompt
        system_prompt = self._build_system_prompt(style)

        # Build user message with WorldState
        user_message = self._build_user_message(world_state, style)

        # Call LLM
        try:
            raw_response = self.llm_callable(system_prompt, user_message)
            prompt_text = self._parse_response(raw_response)
        except Exception as e:
            logger.error(f"LLM compilation failed: {e}")
            # Fall back to template compilation
            fallback = TemplateCompiler()
            result = fallback.compile(world_state, style, **kwargs)
            result.compiler_type = "template_fallback"
            return result

        # Build lora_scales from style
        lora_scales: list[LoRAScaleUpdate] = []
        if style.lora_path:
            lora_scales.append(
                LoRAScaleUpdate(path=style.lora_path, scale=style.lora_default_scale)
            )

        return CompiledPrompt(
            prompts=[PromptEntry(text=prompt_text, weight=1.0)],
            negative_prompt=style.default_negative,
            lora_scales=lora_scales,
            style_name=style.name,
            compiler_type="llm",
            world_state_hash=self._compute_state_hash(world_state, style),
            manifest_hash=self._compute_manifest_hash(style),
            raw_llm_response=raw_response,
        )

    def _build_system_prompt(self, style: StyleManifest) -> str:
        """Build the system prompt with vocab context and few-shot examples."""
        parts = []

        # Base instruction sheet
        if self.instruction_sheet:
            parts.append(self.instruction_sheet.system_prompt)
        else:
            parts.append(
                "You are a prompt compiler for video generation. "
                "Translate the given world state into an effective prompt."
            )

        # Add vocab context
        parts.append("\n\n## Available Vocabulary\n")
        for category, vocab in style.get_all_vocab().items():
            if vocab:
                parts.append(f"\n### {category.title()}")
                for key, value in vocab.items():
                    parts.append(f"- {key}: {value}")

        # Add constraints
        parts.append(f"\n\n## Constraints")
        parts.append(f"- Trigger words (must include): {', '.join(style.trigger_words)}")
        parts.append(f"- Max tokens: {style.max_prompt_tokens}")
        parts.append(f"- Priority order: {', '.join(style.priority_order)}")

        # Add few-shot examples from instruction sheet
        if self.instruction_sheet and self.instruction_sheet.examples:
            parts.append("\n\n## Examples\n")
            for i, example in enumerate(self.instruction_sheet.examples, 1):
                parts.append(f"\n### Example {i}")
                if "world_state" in example:
                    parts.append(f"**Input:**\n{example['world_state']}")
                if "output" in example:
                    parts.append(f"\n**Output:**\n{example['output']}")

        return "\n".join(parts)

    def _build_user_message(self, world_state: WorldState, style: StyleManifest) -> str:
        """Build the user message with WorldState context."""
        context = world_state.to_context_dict()

        parts = ["## World State\n"]
        for key, value in context.items():
            # Include all values except None and empty strings/lists
            # This preserves visible=False, tension=0.0, etc.
            if value is not None and value != "" and value != []:
                parts.append(f"- {key}: {value}")

        parts.append("\n\nGenerate the prompt:")

        return "\n".join(parts)

    def _parse_response(self, response: str) -> str:
        """Parse the LLM response to extract the prompt."""
        # Simple extraction - assume the response IS the prompt
        # Could be enhanced to handle structured output
        return response.strip()


class CachedCompiler(PromptCompiler):
    """
    Caching wrapper for any PromptCompiler.

    Memoizes results based on WorldState + Style manifest content hash.
    """

    def __init__(
        self,
        inner_compiler: PromptCompiler,
        max_cache_size: int = 100,
    ):
        self.inner = inner_compiler
        self.max_cache_size = max_cache_size
        self._cache: dict[str, CompiledPrompt] = {}
        self._cache_order: list[str] = []  # For LRU eviction

    def compile(
        self,
        world_state: WorldState,
        style: StyleManifest,
        **kwargs: Any,
    ) -> CompiledPrompt:
        """Compile with caching."""
        # Use full cache key that includes manifest content hash
        cache_key = self._compute_cache_key(world_state, style)

        if cache_key in self._cache:
            # Move to end of LRU order
            self._cache_order.remove(cache_key)
            self._cache_order.append(cache_key)
            return self._cache[cache_key]

        # Compile and cache
        result = self.inner.compile(world_state, style, **kwargs)

        # LRU eviction
        if len(self._cache) >= self.max_cache_size:
            oldest_key = self._cache_order.pop(0)
            del self._cache[oldest_key]

        self._cache[cache_key] = result
        self._cache_order.append(cache_key)

        return result

    def clear_cache(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._cache_order.clear()


def _load_instruction_sheet(style: StyleManifest) -> InstructionSheet | None:
    """
    Load instruction sheet for a style.

    Looks for instructions.md in the style directory.
    """
    if not style.name:
        return None

    from .style_manifest import get_style_dirs

    candidates: list[Path] = []

    raw = (style.instruction_sheet_path or "").strip()
    if raw:
        p = Path(raw).expanduser()
        if p.is_absolute():
            candidates.append(p)
        else:
            # Try both "instructions.md" and "style/instructions.md" style paths.
            for base in get_style_dirs():
                candidates.append(base / style.name / p)
                candidates.append(base / p)
    else:
        for base in get_style_dirs():
            candidates.append(base / style.name / "instructions.md")

    for instruction_path in candidates:
        if not instruction_path.exists():
            continue
        try:
            return InstructionSheet.from_markdown(instruction_path)
        except Exception as e:
            logger.warning(
                "Failed to load instruction sheet %s: %s", instruction_path, e
            )
            return None

    return None


def create_compiler(
    style: StyleManifest,
    mode: str = "auto",
) -> PromptCompiler:
    """
    Create a PromptCompiler for the given style.

    Args:
        style: The style manifest to compile prompts for
        mode: Compiler mode - "gemini", "template", or "auto"
              "auto" uses Gemini if GEMINI_API_KEY is set, else template

    Returns:
        A PromptCompiler instance (possibly wrapped in CachedCompiler)

    Environment variables:
        SCOPE_LLM_COMPILER: Override mode ("gemini", "template", "auto")
        GEMINI_API_KEY: Required for Gemini mode
    """
    # Check env override
    env_mode = os.getenv("SCOPE_LLM_COMPILER", "auto")
    if env_mode in ("gemini", "template"):
        mode = env_mode

    # Template mode - simple and fast
    if mode == "template":
        logger.info(f"Using TemplateCompiler for style '{style.name}'")
        return TemplateCompiler()

    # Gemini mode - check availability
    if mode == "gemini":
        from .gemini_client import GeminiCompiler, is_gemini_available

        if not is_gemini_available():
            logger.warning(
                "SCOPE_LLM_COMPILER=gemini but GEMINI_API_KEY not set, "
                "falling back to TemplateCompiler"
            )
            return TemplateCompiler()

        instruction_sheet = _load_instruction_sheet(style)
        inner = LLMCompiler(
            llm_callable=GeminiCompiler(),
            instruction_sheet=instruction_sheet,
        )
        logger.info(
            f"Using LLMCompiler (Gemini) for style '{style.name}' "
            f"with instruction_sheet={instruction_sheet.name if instruction_sheet else 'None'}"
        )
        return CachedCompiler(inner)

    # Auto mode - use Gemini if available
    from .gemini_client import is_gemini_available

    if is_gemini_available():
        return create_compiler(style, mode="gemini")

    logger.info(f"Using TemplateCompiler for style '{style.name}' (auto mode, no API key)")
    return TemplateCompiler()
