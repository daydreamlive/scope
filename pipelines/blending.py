import logging
from enum import Enum

import torch

logger = logging.getLogger(__name__)


class BlenderState(Enum):
    """State of the EmbeddingBlender for explicit transition management."""

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


def parse_transition_config(transition):
    """Parse and validate transition configuration.

    Args:
        transition: Transition config dict (from WebRTC parameters)

    Returns:
        tuple: (target_prompts, num_steps, temporal_method, is_immediate)
            - target_prompts: Target prompts list from transition
            - num_steps: Number of steps for transition
            - temporal_method: Interpolation method (linear or slerp)
            - is_immediate: True if num_steps=0 (instant), False if smooth
    """
    if transition is None:
        return None, 0, "linear", False

    # Extract from dict (Pydantic models already converted to dict at API boundary)
    target_prompts = transition["target_prompts"]
    num_steps = transition.get("num_steps", 0)
    temporal_method = transition.get("temporal_interpolation_method", "linear")

    # Validate target prompts
    if not target_prompts:
        logger.warning(
            "parse_transition_config: Empty target_prompts, ignoring transition"
        )
        return None, 0, temporal_method, False

    # Check if at least one prompt has non-empty text
    has_valid_prompt = any(p.get("text", "").strip() for p in target_prompts)
    if not has_valid_prompt:
        logger.warning(
            "parse_transition_config: All target prompts are empty, ignoring transition"
        )
        return None, 0, temporal_method, False

    # If num_steps is 0, this is an immediate transition
    is_immediate = num_steps <= 0
    if is_immediate:
        logger.debug("parse_transition_config: num_steps=0, immediate transition")

    return target_prompts, num_steps, temporal_method, is_immediate


class EmbeddingBlender:
    """Manages embedding blending for pipelines

    This class handles the core business logic for embedding blending:
    - Spatial blending: Combining multiple weighted embeddings into a single embedding
    - Temporal blending: Smooth transitions between embeddings over time
    - State management: Transition state machine (IDLE → TRANSITIONING → IDLE)

    Architecture Notes:
    - This class operates ONLY on pre-encoded embeddings (no text encoding)
    - Text encoding happens upstream in TextConditioningBlock
    - This separation allows EmbeddingBlender to be generic and reusable
    - Intentionally separate from EmbeddingBlendingBlock to maintain
      separation between business logic (this class) and pipeline integration (the block)
    - Cache management is handled by pipeline state flags (prompt_embeds_updated)
    """

    def __init__(
        self,
        device,
        dtype,
    ) -> None:
        self.device = device
        self.dtype = dtype

        # State management for transitions
        self._state = BlenderState.IDLE

        # Temporal interpolation state (prompt transitions)
        self._transition_queue = []  # Queue of pre-computed interpolated embeddings
        self._current_blend_embedding = None  # Cached current blend for transitions

    def blend(
        self, embeddings, weights, interpolation_method, cache_result=True
    ) -> torch.Tensor | None:
        """Blend pre-encoded embeddings using specified interpolation method.

        Args:
            embeddings: List of pre-encoded embedding tensors
            weights: List of weights corresponding to each embedding
            interpolation_method: Method for spatial interpolation ('linear' or 'slerp')
            cache_result: Whether to cache the result as current blend (default True)

        Returns:
            Blended embedding tensor, or None if inputs are invalid
        """
        if not embeddings:
            logger.warning("blend: No embeddings provided")
            return None

        if len(embeddings) != len(weights):
            logger.warning(
                f"blend: Mismatch between embeddings ({len(embeddings)}) and weights ({len(weights)})"
            )
            return None

        # Use the utility function for actual blending
        result = blend_embeddings(
            embeddings, weights, interpolation_method, self.dtype, self.device
        )

        # Cache the current blend for potential transitions (unless explicitly disabled)
        if result is not None and cache_result:
            self._current_blend_embedding = result.detach()

        return result

    def start_transition(
        self,
        target_embedding,
        num_steps: int,
        temporal_interpolation_method: str,
    ) -> None:
        """Start a temporal transition from current blend to target blend.

        This pre-computes interpolated embeddings.

        Args:
            target_embedding: Pre-encoded and blended target embedding tensor
            num_steps: Number of generation calls to transition over
            temporal_interpolation_method: Method for temporal interpolation (linear or slerp)
        """

        if self._current_blend_embedding is None:
            logger.warning(
                "start_transition: No current blend cached, cannot start transition"
            )
            return

        if target_embedding is None:
            logger.warning(
                "start_transition: No target embedding provided, cannot start transition"
            )
            return

        # Check if embeddings are actually different (skip if too similar to save computation)
        diff_norm = (target_embedding - self._current_blend_embedding).norm()
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
                    self._current_blend_embedding, target_embedding, t.item()
                )
            else:
                # Linear interpolation
                interpolated = torch.lerp(
                    self._current_blend_embedding, target_embedding, t
                )
            interpolated_embeddings.append(interpolated.detach())

        # Store interpolated embeddings in queue and update state
        self._transition_queue = interpolated_embeddings
        self._state = BlenderState.TRANSITIONING

        logger.info(
            f"start_transition: Started transition over {num_steps} steps using {temporal_interpolation_method}. "
            f"Queue length: {len(self._transition_queue)}, State: {self._state.value}"
        )

    def get_next_embedding(self) -> torch.Tensor | None:
        """Get the next interpolated embedding during a transition.

        This should be called on each generation call. If a transition is active,
        it will return and pop the next interpolated embedding from the queue.
        Otherwise, it returns None.

        Returns:
            Next interpolated embedding during transition, or None if not transitioning
        """
        # If we have a transition in progress, pop from queue
        if self._state == BlenderState.TRANSITIONING and self._transition_queue:
            next_embedding = self._transition_queue.pop(0)
            logger.debug(
                f"get_next_embedding: Popping from transition queue ({len(self._transition_queue)} remaining)"
            )

            # Update cached current blend as we progress
            self._current_blend_embedding = next_embedding

            if not self._transition_queue:
                self._state = BlenderState.IDLE
                logger.info(
                    f"get_next_embedding: Transition completed, State: {self._state.value}"
                )

            return next_embedding

        # Not transitioning - return None (block handles normal blending)
        return None

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
