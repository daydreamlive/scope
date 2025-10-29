import logging
from collections import OrderedDict
from enum import Enum

import torch

logger = logging.getLogger(__name__)


class BlenderState(Enum):
    """State of the PromptBlender for explicit transition management."""

    IDLE = "idle"
    TRANSITIONING = "transitioning"


# Numerical stability constants
EPSILON = 1e-8  # Small value to prevent division by zero
SLERP_PARALLEL_THRESHOLD = 1e-4  # Threshold for detecting parallel embeddings in SLERP

# Cache configuration
DEFAULT_MAX_CACHE_SIZE = 10  # Maximum number of prompts to cache

# Logging configuration
LOG_PROMPT_PREVIEW_LENGTH = 50  # Characters to show in log messages for prompt preview

# Prompt defaults
DEFAULT_PROMPT_WEIGHT = 1.0  # Default weight for prompt blending

# Minimum embedding difference threshold for skipping transitions
MIN_EMBEDDING_DIFF_THRESHOLD = 0.01


def normalize_weights(weights, dtype, device) -> torch.Tensor:
    """Normalize weights to sum to 1.0"""
    weights_tensor = torch.tensor(weights, dtype=dtype, device=device)
    total = weights_tensor.sum()
    if total > 0:
        weights_tensor = weights_tensor / total
    else:
        # Fallback: equal weights for all inputs
        weights_tensor = torch.ones_like(weights_tensor) / len(weights_tensor)
        logger.warning(
            "normalize_weights: All weights zero or negative, using equal weights"
        )
    return weights_tensor


def slerp(embed1, embed2, t) -> torch.Tensor:
    """Spherical linear interpolation between two embeddings"""
    # Normalize embeddings
    embed1_norm = embed1 / (embed1.norm(dim=-1, keepdim=True) + EPSILON)
    embed2_norm = embed2 / (embed2.norm(dim=-1, keepdim=True) + EPSILON)

    # Compute angle between embeddings
    dot_product = (embed1_norm * embed2_norm).sum(dim=-1, keepdim=True)
    # Clamp to avoid numerical issues with acos
    dot_product = torch.clamp(dot_product, -1.0, 1.0)
    omega = torch.acos(dot_product)

    # Fall back to linear interpolation when embeddings are nearly parallel
    if omega.abs().max() < SLERP_PARALLEL_THRESHOLD:
        return (1.0 - t) * embed1 + t * embed2

    sin_omega = torch.sin(omega)

    # Compute interpolation coefficients
    coeff1 = torch.sin((1.0 - t) * omega) / (sin_omega + EPSILON)
    coeff2 = torch.sin(t * omega) / (sin_omega + EPSILON)

    # Interpolate
    result = coeff1 * embed1 + coeff2 * embed2
    return result


def blend_embeddings(embeddings, weights, method, dtype, device) -> torch.Tensor | None:
    """Blend multiple embeddings using linear or slerp interpolation"""
    if not embeddings:
        logger.warning("blend_embeddings: No embeddings provided")
        return None

    # Normalize weights
    normalized_weights = normalize_weights(weights, dtype, device)

    # Apply interpolation
    if method == "slerp" and len(embeddings) == 2:
        # Spherical linear interpolation for 2 prompts
        t = normalized_weights[1].item()
        combined_embeds = slerp(embeddings[0], embeddings[1], t)
    else:
        # Linear interpolation (weighted average) with normalization
        # Compute weighted average of norms to preserve magnitude
        target_norm = sum(
            embed.norm() * weight
            for embed, weight in zip(embeddings, normalized_weights, strict=False)
        )

        # Compute linear blend
        combined_embeds = torch.zeros_like(embeddings[0])
        for embed, weight in zip(embeddings, normalized_weights, strict=False):
            combined_embeds += weight * embed

        # Normalize to preserve embedding magnitude and prevent artifacts
        current_norm = combined_embeds.norm()
        if current_norm > EPSILON:
            combined_embeds = combined_embeds * (target_norm / current_norm)

    return combined_embeds


def parse_and_start_transition(transition, prompt_blender, text_encoder):
    """Parse transition dict and start it via PromptBlender.

    Args:
        transition: Transition config dict (from WebRTC parameters)
        prompt_blender: PromptBlender instance
        text_encoder: Text encoder for encoding prompts

    Returns:
        tuple: (target_prompts, should_apply_immediately)
            - target_prompts: Target prompts list from transition
            - should_apply_immediately: True if num_steps=0 (instant), False if smooth
    """
    if transition is None:
        return None, False

    # Extract from dict (Pydantic models already converted to dict at API boundary)
    target_prompts = transition["target_prompts"]
    num_steps = transition.get("num_steps", 0)
    temporal_method = transition.get("temporal_interpolation_method", "linear")

    # Validate target prompts
    if not target_prompts:
        logger.warning(
            "parse_and_start_transition: Empty target_prompts, ignoring transition"
        )
        return None, False

    # Check if at least one prompt has non-empty text
    has_valid_prompt = any(p.get("text", "").strip() for p in target_prompts)
    if not has_valid_prompt:
        logger.warning(
            "parse_and_start_transition: All target prompts are empty, ignoring transition"
        )
        return None, False

    # If num_steps is 0, caller should apply immediately
    if num_steps <= 0:
        logger.debug(
            "parse_and_start_transition: num_steps=0, returning for immediate application"
        )
        return target_prompts, True

    # Start the smooth transition
    prompt_blender.start_transition(
        target_prompts=target_prompts,
        num_steps=num_steps,
        temporal_interpolation_method=temporal_method,
        text_encoder=text_encoder,
    )

    return target_prompts, False


class PromptBlender:
    """Manages prompt caching and blending for pipelines"""

    def __init__(
        self,
        device,
        dtype,
        max_cache_size: int = DEFAULT_MAX_CACHE_SIZE,
        cache_reset_callback=None,
    ) -> None:
        self.device = device
        self.dtype = dtype
        self.max_cache_size = max_cache_size
        self._prompt_cache = OrderedDict()  # LRU cache using OrderedDict
        self._current_prompts = []
        self._interpolation_method = "linear"

        # State management for transitions
        self._state = BlenderState.IDLE

        # Temporal interpolation state (prompt transitions)
        self._transition_queue = []  # Queue of pre-computed interpolated embeddings
        self._current_blend_embedding = None  # Cached current blend for transitions

        # Pipeline-specific cache reset callback invoked during transitions
        self._cache_reset_callback = cache_reset_callback

    def should_update(self, prompts, interpolation_method) -> bool:
        """Check if prompts or interpolation method changed"""
        if prompts is None:
            return False

        # Compare as tuples for simple equality check
        new_comparable = [
            (p.get("text", ""), p.get("weight", DEFAULT_PROMPT_WEIGHT)) for p in prompts
        ]
        old_comparable = [
            (p.get("text", ""), p.get("weight", DEFAULT_PROMPT_WEIGHT))
            for p in self._current_prompts
        ]

        prompts_changed = (
            new_comparable != old_comparable
            or interpolation_method != self._interpolation_method
        )

        # If prompts changed while transitioning, cancel the transition
        if prompts_changed and self._state == BlenderState.TRANSITIONING:
            logger.info(
                "should_update: Prompts changed during transition, cancelling transition"
            )
            self.cancel_transition()

        return prompts_changed

    def blend(self, prompts, interpolation_method, text_encoder) -> torch.Tensor | None:
        """Update state and return blended embeddings.

        If a transition is active, this returns None to signal that the pipeline
        should skip re-blending (transition queue will provide embeddings via get_next_embedding).
        """
        # If transitioning, skip blend - get_next_embedding() handles it
        if self._state == BlenderState.TRANSITIONING:
            logger.debug("blend: Transition active, skipping blend request")
            return None

        self._current_prompts = prompts if prompts else []
        self._interpolation_method = interpolation_method

        result = self._encode_and_blend(text_encoder)
        # Cache the current blend for potential transitions
        if result is not None:
            self._current_blend_embedding = result.detach()
        return result

    def _encode_and_blend(self, text_encoder) -> torch.Tensor | None:
        """Encode prompts (with caching) and blend them"""
        if not self._current_prompts:
            logger.warning("PromptBlender: No prompts set, using empty prompt")
            self._current_prompts = [{"text": "", "weight": DEFAULT_PROMPT_WEIGHT}]

        embeddings = []
        weights = []

        # Encode and cache prompts
        for prompt in self._current_prompts:
            prompt_text = prompt.get("text", "")
            weight = prompt.get("weight", DEFAULT_PROMPT_WEIGHT)

            if prompt_text not in self._prompt_cache:
                # Evict oldest entry if cache is full (LRU eviction)
                if len(self._prompt_cache) >= self.max_cache_size:
                    oldest_key = next(iter(self._prompt_cache))
                    self._prompt_cache.pop(oldest_key)
                    logger.info(
                        f"PromptBlender: Evicted oldest cache entry: {oldest_key[:LOG_PROMPT_PREVIEW_LENGTH]}..."
                    )

                logger.info(
                    f"PromptBlender: Encoding and caching prompt: {prompt_text[:LOG_PROMPT_PREVIEW_LENGTH]}..."
                )
                encoded = text_encoder(text_prompts=[prompt_text])
                # Detach from computation graph to prevent memory leak
                self._prompt_cache[prompt_text] = encoded["prompt_embeds"].detach()
            else:
                # Move to end (mark as recently used)
                self._prompt_cache.move_to_end(prompt_text)

            embeddings.append(self._prompt_cache[prompt_text])
            weights.append(weight)

        if not embeddings:
            logger.warning("PromptBlender: No cached embeddings found")
            return None

        # Use the utility function for actual blending
        return blend_embeddings(
            embeddings, weights, self._interpolation_method, self.dtype, self.device
        )

    def start_transition(
        self,
        target_prompts,
        num_steps: int,
        temporal_interpolation_method: str,
        text_encoder,
    ) -> None:
        """Start a temporal transition from current blend to target blend.

        This pre-computes interpolated embeddings.

        Args:
            target_prompts: List of prompt dicts for target blend
            num_steps: Number of generation calls to transition over
            temporal_interpolation_method: Method for temporal interpolation (linear or slerp)
            text_encoder: Text encoder to use for encoding target prompts
        """

        if self._current_blend_embedding is None:
            logger.warning(
                "start_transition: No current blend cached, cannot start transition"
            )
            return

        # Encode and blend target prompts
        old_prompts = self._current_prompts
        old_method = self._interpolation_method

        # Temporarily set target prompts to encode them
        self._current_prompts = target_prompts
        target_blend = self._encode_and_blend(text_encoder)

        # Restore original prompts
        self._current_prompts = old_prompts
        self._interpolation_method = old_method

        if target_blend is None:
            logger.warning(
                "start_transition: Failed to encode target blend, cannot start transition"
            )
            return

        # Check if embeddings are actually different (skip if too similar to save computation)
        diff_norm = (target_blend - self._current_blend_embedding).norm()
        if diff_norm < MIN_EMBEDDING_DIFF_THRESHOLD:
            logger.info(
                f"start_transition: Embeddings are very similar (diff_norm={diff_norm.item():.6f}), skipping transition"
            )
            return

        # Pre-compute interpolation steps
        # Generate num_steps embeddings from current to target
        t_values = torch.linspace(0, 1, steps=num_steps).to(self.device)

        interpolated_embeddings = []
        for _i, t in enumerate(t_values):
            if temporal_interpolation_method == "slerp":
                interpolated = slerp(
                    self._current_blend_embedding, target_blend, t.item()
                )
            else:
                # Linear interpolation
                interpolated = torch.lerp(
                    self._current_blend_embedding, target_blend, t
                )
            interpolated_embeddings.append(interpolated.detach())

        # Store interpolated embeddings in queue and update state
        self._transition_queue = interpolated_embeddings
        self._state = BlenderState.TRANSITIONING

        logger.info(
            f"start_transition: Started transition over {num_steps} steps using {temporal_interpolation_method}. "
            f"Queue length: {len(self._transition_queue)}, State: {self._state.value}"
        )

    def get_next_embedding(self, text_encoder) -> torch.Tensor | None:
        """Get the next embedding, either from transition queue or current blend.

        This should be called on each generation call. If a transition is active,
        it will return and pop the next interpolated embedding. Otherwise, it returns
        the current blend.

        Args:
            text_encoder: Text encoder to use if encoding is needed

        Returns:
            Blended or interpolated embedding, or None if no prompts set
        """
        # If we have a transition in progress, pop from queue
        if self._state == BlenderState.TRANSITIONING and self._transition_queue:
            next_embedding = self._transition_queue.pop(0)
            logger.debug(
                f"get_next_embedding: Popping from transition queue ({len(self._transition_queue)} remaining)"
            )

            # Invoke cache reset callback if provided (critical for model to respond to new embedding)
            if self._cache_reset_callback:
                logger.debug("get_next_embedding: Invoking cache reset callback")
                self._cache_reset_callback()

            # Update cached current blend as we progress
            self._current_blend_embedding = next_embedding

            if not self._transition_queue:
                self._state = BlenderState.IDLE
                logger.info(
                    f"get_next_embedding: Transition completed, State: {self._state.value}"
                )

            return next_embedding

        # Otherwise, return current blend (no logging needed for normal path)
        return self._encode_and_blend(text_encoder)

    def is_transitioning(self) -> bool:
        """Check if a transition is currently in progress."""
        return self._state == BlenderState.TRANSITIONING

    def cancel_transition(self) -> None:
        """Cancel any active transition and clear the queue."""
        if self._state == BlenderState.TRANSITIONING:
            logger.info(
                f"cancel_transition: Cancelling transition with {len(self._transition_queue)} steps remaining, State: {self._state.value} -> {BlenderState.IDLE.value}"
            )
            self._transition_queue.clear()
            self._state = BlenderState.IDLE
