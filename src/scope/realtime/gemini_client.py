"""
Gemini integration for LLMCompiler.

Provides:
- GeminiCompiler: Implements llm_callable signature for prompt compilation
- GeminiWorldChanger: Natural language WorldState updates
- GeminiPromptJiggler: Prompt variation generator
"""

from __future__ import annotations

import logging
import os
import time
import hashlib
import threading
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from scope.realtime.world_state import WorldState

logger = logging.getLogger(__name__)

# Default model - Gemini 3 Flash Preview
DEFAULT_MODEL = "gemini-3-flash-preview"

# Rate limiting
_last_call_time = 0.0
_min_call_interval = 0.1  # 10 calls/sec max
_rate_limit_lock = threading.Lock()


def _rate_limit():
    """Simple rate limiter to avoid hitting API limits."""
    global _last_call_time
    with _rate_limit_lock:
        now = time.time()
        elapsed = now - _last_call_time
        if elapsed < _min_call_interval:
            time.sleep(_min_call_interval - elapsed)
        _last_call_time = time.time()


# Prompt jiggle cache (short TTL to reduce redundant LLM calls in live usage)
_JIGGLE_CACHE_TTL_SEC = 10.0
_JIGGLE_CACHE_MAX_SIZE = 128
_JIGGLE_SYSTEM_PROMPT_VERSION = "v1"
_jiggle_cache: dict[str, tuple[float, list[str]]] = {}
_jiggle_cache_lock = threading.Lock()


def _jiggle_cache_key(
    prompt: str,
    model: str,
    mode: str,
    direction: str | None,
    intensity: float,
    count: int,
) -> str:
    dir_norm = (direction or "").strip().lower()
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]
    return (
        f"{_JIGGLE_SYSTEM_PROMPT_VERSION}:{model}:{mode}:{dir_norm}:{intensity:.2f}:{count}:{prompt_hash}"
    )


def _get_jiggle_cached(key: str) -> list[str] | None:
    now = time.time()
    with _jiggle_cache_lock:
        entry = _jiggle_cache.get(key)
        if entry is None:
            return None
        ts, variations = entry
        if now - ts > _JIGGLE_CACHE_TTL_SEC:
            _jiggle_cache.pop(key, None)
            return None
        return variations


def _set_jiggle_cached(key: str, variations: list[str]) -> None:
    now = time.time()
    with _jiggle_cache_lock:
        if len(_jiggle_cache) >= _JIGGLE_CACHE_MAX_SIZE:
            oldest_key = min(_jiggle_cache, key=lambda k: _jiggle_cache[k][0])
            _jiggle_cache.pop(oldest_key, None)
        _jiggle_cache[key] = (now, variations)


def init_client():
    """
    Initialize the Gemini client.

    Reads GEMINI_API_KEY from environment.

    Returns:
        google.genai.Client or None if no API key available
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.warning("GEMINI_API_KEY not set - Gemini features will be unavailable")
        return None

    try:
        from google import genai

        return genai.Client(api_key=api_key)
    except ImportError:
        logger.error("google-genai package not installed")
        return None
    except Exception as e:
        logger.error(f"Failed to initialize Gemini client: {e}")
        return None


class GeminiCompiler:
    """
    Gemini-based LLM compiler.

    Implements the llm_callable signature: (system_prompt, user_message) -> str
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.4,
        max_output_tokens: int = 256,
    ):
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self._client = None

    @property
    def client(self):
        """Lazy-initialize client on first use."""
        if self._client is None:
            self._client = init_client()
        return self._client

    def __call__(self, system_prompt: str, user_message: str) -> str:
        """
        Call Gemini to generate a prompt.

        Matches LLMCompiler's llm_callable signature.

        Args:
            system_prompt: System instructions with vocab, examples, etc.
            user_message: The WorldState context and request

        Returns:
            Generated prompt text

        Raises:
            RuntimeError: If no API key or client unavailable
        """
        if self.client is None:
            raise RuntimeError("Gemini client not available - check GEMINI_API_KEY")

        _rate_limit()

        try:
            from google.genai import types

            response = self.client.models.generate_content(
                model=self.model,
                contents=[user_message],
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=self.temperature,
                    max_output_tokens=self.max_output_tokens,
                    response_mime_type="text/plain",
                ),
            )
            return response.text.strip()

        except Exception as e:
            logger.error(f"Gemini compilation failed: {e}")
            raise


class GeminiWorldChanger:
    """
    Natural language WorldState editor.

    Takes an instruction like "make Rooster angry" and returns updated WorldState.
    """

    SYSTEM_PROMPT = """You are a WorldState editor for an animation system.

Given the current WorldState as JSON and a natural language instruction,
output ONLY valid JSON representing the updated WorldState.

Rules:
- Make minimal changes - only modify what the instruction specifies
- Preserve all fields not mentioned in the instruction
- Valid emotions: happy, sad, angry, frustrated, shocked, neutral, determined, confused, surprised
- Valid beat types: setup, escalation, climax, payoff, reset, transition
- Valid camera intents: establishing, close_up, medium, wide, low_angle, high_angle, tracking, static
- Character actions should be short action verbs or phrases

Output ONLY the JSON - no markdown, no explanation, no comments."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.3,
    ):
        self.model = model
        self.temperature = temperature
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = init_client()
        return self._client

    def change(self, world_state: WorldState, instruction: str) -> WorldState:
        """
        Apply a natural language instruction to update WorldState.

        Args:
            world_state: Current world state
            instruction: Natural language instruction (e.g., "make Rooster angry")

        Returns:
            Updated WorldState

        Raises:
            ValueError: If LLM returns invalid JSON
            RuntimeError: If Gemini unavailable
        """
        from .world_state import WorldState

        if self.client is None:
            raise RuntimeError("Gemini client not available - check GEMINI_API_KEY")

        _rate_limit()

        current_json = world_state.model_dump_json(indent=2)
        user_message = f"""Current WorldState:
{current_json}

Instruction: {instruction}

Output the updated WorldState JSON:"""

        try:
            from google.genai import types

            response = self.client.models.generate_content(
                model=self.model,
                contents=[user_message],
                config=types.GenerateContentConfig(
                    system_instruction=self.SYSTEM_PROMPT,
                    temperature=self.temperature,
                    max_output_tokens=2048,
                    response_mime_type="application/json",
                ),
            )

            response_text = response.text.strip()

            # Parse the JSON response
            return WorldState.model_validate_json(response_text)

        except Exception as e:
            logger.error(f"WorldState change failed: {e}")
            raise ValueError(f"Failed to parse LLM response: {e}") from e


class GeminiPromptJiggler:
    """
    Generates variations of a prompt while preserving meaning.

    Useful for adding visual variety without changing the scene.
    """

    SYSTEM_PROMPT = """You are a prompt variation generator for video generation.

Given a prompt, create a subtle variation that:
- Preserves all core elements (characters, action, camera, style triggers)
- Adjusts word order, synonyms, or emphasis
- Stays within the same token budget
- Maintains the same visual intent

    Output ONLY the varied prompt - no explanation, no quotes."""

    ATTENTIONAL_SYSTEM_PROMPT = """You are a prompt variation generator for video generation.

Given a prompt, create EXACTLY {count} distinct variations that:
- Preserve ALL semantic content (characters, actions, setting, mood, style triggers)
- Adjust word order, emphasis, or phrasing
- Shift what the model will attend to first
- Stay within a similar token budget
- Each variation must be meaningfully different
{direction_clause}

Output ONLY the varied prompts, one per line. No numbering, bullets, quotes, or explanation."""

    SEMANTIC_SYSTEM_PROMPT = """You are a prompt variation generator for video generation.

Given a prompt and a direction, create EXACTLY {count} distinct variations that:
- Shift the meaning/mood/camera in the direction specified
- Preserve core characters and setting unless the direction says otherwise
- Make meaningful semantic changes, not just word swaps
- Each variation must be meaningfully different

Direction: {direction}

Output ONLY the varied prompts, one per line. No numbering, bullets, quotes, or explanation."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.7,
    ):
        self.model = model
        self.temperature = temperature
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = init_client()
        return self._client

    def jiggle(self, prompt: str, intensity: float = 0.3) -> str:
        """
        Generate a variation of the prompt.

        Args:
            prompt: Original prompt text
            intensity: How different the variation should be (0-1)

        Returns:
            Varied prompt text
        """
        if self.client is None:
            # Graceful fallback - return original
            logger.warning("Gemini unavailable, returning original prompt")
            return prompt

        _rate_limit()

        # Scale temperature with intensity
        adjusted_temp = 0.3 + (intensity * 0.7)  # Range: 0.3 to 1.0

        user_message = f"""Original prompt:
{prompt}

Generate a subtle variation (intensity: {intensity:.1f}):"""

        try:
            from google.genai import types

            response = self.client.models.generate_content(
                model=self.model,
                contents=[user_message],
                config=types.GenerateContentConfig(
                    system_instruction=self.SYSTEM_PROMPT,
                    temperature=adjusted_temp,
                    max_output_tokens=256,
                    response_mime_type="text/plain",
                ),
            )
            return response.text.strip()

        except Exception as e:
            logger.warning(f"Prompt jiggle failed, returning original: {e}")
            return prompt

    def _parse_variations(self, raw: str, count: int, original: str) -> list[str]:
        lines = raw.splitlines()
        variations: list[str] = []
        original_stripped = original.strip()

        prefixes = [
            "1.",
            "2.",
            "3.",
            "4.",
            "5.",
            "6.",
            "7.",
            "8.",
            "9.",
            "10.",
            "-",
            "•",
            "*",
            "—",
        ]

        for line in lines:
            candidate = line.strip()
            if not candidate:
                continue

            for prefix in prefixes:
                if candidate.startswith(prefix):
                    candidate = candidate[len(prefix) :].strip()
                    break

            candidate = candidate.strip().strip('"').strip("'").strip()
            if not candidate:
                continue
            if candidate == original_stripped:
                continue
            variations.append(candidate)

        # Dedupe while preserving order
        seen: set[str] = set()
        unique: list[str] = []
        for variation in variations:
            if variation in seen:
                continue
            seen.add(variation)
            unique.append(variation)

        if len(unique) < count:
            unique.extend([original_stripped] * (count - len(unique)))
        return unique[:count]

    def jiggle_multi(
        self,
        prompt: str,
        count: int = 3,
        intensity: float = 0.3,
        direction: str | None = None,
        mode: Literal["attentional", "semantic"] = "attentional",
    ) -> list[str]:
        """Generate multiple variations of the prompt in a single LLM call."""
        if count < 1:
            raise ValueError("count must be >= 1")

        direction_clean = direction.strip() if direction else None
        if mode == "semantic" and not direction_clean:
            raise ValueError("direction is required when mode='semantic'")

        if self.client is None:
            logger.warning("Gemini unavailable, returning original prompt")
            return [prompt] * count

        cache_key = _jiggle_cache_key(
            prompt=prompt,
            model=self.model,
            mode=mode,
            direction=direction_clean,
            intensity=float(intensity),
            count=int(count),
        )
        cached = _get_jiggle_cached(cache_key)
        if cached is not None:
            if len(cached) >= count:
                return cached[:count]
            return cached + [prompt] * (count - len(cached))

        _rate_limit()

        adjusted_temp = 0.3 + (intensity * 0.7)  # Range: 0.3 to 1.0
        max_output_tokens = min(2048, 256 * count)

        if mode == "semantic":
            system_prompt = self.SEMANTIC_SYSTEM_PROMPT.format(
                count=count,
                direction=direction_clean,
            )
        else:
            direction_clause = f"\nDirection: {direction_clean}" if direction_clean else ""
            system_prompt = self.ATTENTIONAL_SYSTEM_PROMPT.format(
                count=count,
                direction_clause=direction_clause,
            )

        user_message = f"""Original prompt:
{prompt}

Generate EXACTLY {count} variations (intensity: {intensity:.1f}):"""

        try:
            from google.genai import types

            response = self.client.models.generate_content(
                model=self.model,
                contents=[user_message],
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=adjusted_temp,
                    max_output_tokens=max_output_tokens,
                    response_mime_type="text/plain",
                ),
            )
            raw = (response.text or "").strip()
            variations = self._parse_variations(raw, count=count, original=prompt)

            if any(v != prompt for v in variations):
                _set_jiggle_cached(cache_key, variations)

            return variations
        except Exception as e:
            logger.warning(f"Prompt jiggle_multi failed, returning original: {e}")
            return [prompt] * count


def is_gemini_available() -> bool:
    """Check if Gemini is configured and available."""
    return bool(os.environ.get("GEMINI_API_KEY"))
