import logging
from dataclasses import dataclass
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


@dataclass(frozen=True, slots=True)
class TransitionConfig:
    """Normalized transition configuration.

    This value object describes *how* to transition (policy), independent of prompts.

    The semantics are:
        - num_steps > 0 → smooth transition over that many steps
        - num_steps <= 0 → no smooth transition (snap to target)
    """

    num_steps: int
    temporal_interpolation_method: str


def parse_transition_config(transition: dict | None) -> TransitionConfig:
    """Parse and normalize transition configuration.

    Args:
        transition: Transition config dict (from WebRTC parameters)

    Returns:
        TransitionConfig: Normalized transition policy with:
            - num_steps: Number of steps for transition
            - temporal_interpolation_method: Interpolation method (linear or slerp)
    """
    if transition is None:
        return TransitionConfig(num_steps=0, temporal_interpolation_method="linear")

    # Extract from dict (Pydantic models already converted to dict at API boundary)
    raw_num_steps = transition.get("num_steps", 0)
    try:
        num_steps = int(raw_num_steps)
    except (TypeError, ValueError):
        logger.warning(
            "parse_transition_config: Invalid num_steps %r, defaulting to 0",
            raw_num_steps,
        )
        num_steps = 0

    temporal_interpolation_method = transition.get(
        "temporal_interpolation_method", "linear"
    )

    return TransitionConfig(
        num_steps=num_steps,
        temporal_interpolation_method=temporal_interpolation_method,
    )


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
    - Cache management is handled by pipeline state flags (conditioning_embeds_updated)
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

        # Temporal interpolation state (embedding transitions)
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

    def set_current_embedding(self, embedding: torch.Tensor) -> None:
        """Manually set the current blend embedding used as the source for transitions.

        This is useful when the caller manages spatial blending separately and wants
        to drive temporal transitions from the last used embedding.
        """
        if embedding is None:
            self._current_blend_embedding = None
            return

        self._current_blend_embedding = embedding.detach()

    def start_transition(
        self,
        source_embedding,
        target_embedding,
        num_steps: int,
        temporal_interpolation_method: str,
    ) -> None:
        """Start a temporal transition from source embedding to target embedding.

        This pre-computes interpolated embeddings.

        Args:
            source_embedding: Pre-encoded current embedding tensor to transition from
            target_embedding: Pre-encoded and blended target embedding tensor
            num_steps: Number of generation calls to transition over
            temporal_interpolation_method: Method for temporal interpolation (linear or slerp)
        """

        if source_embedding is None:
            logger.warning(
                "start_transition: No source embedding provided, cannot start transition"
            )
            return

        if target_embedding is None:
            logger.warning(
                "start_transition: No target embedding provided, cannot start transition"
            )
            return

        if num_steps <= 0:
            logger.warning(
                "start_transition: num_steps=%s, expected > 0 for smooth transition",
                num_steps,
            )
            return

        # Cache the starting embedding for this transition
        self._current_blend_embedding = source_embedding.detach()

        # Check if embeddings are actually different (skip if too similar to save computation)
        diff_norm = (target_embedding - self._current_blend_embedding).norm()
        if diff_norm < MIN_EMBEDDING_DIFF_THRESHOLD:
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

            # Update cached current blend as we progress
            self._current_blend_embedding = next_embedding

            if not self._transition_queue:
                self._state = BlenderState.IDLE

            return next_embedding

        # Not transitioning - return None (block handles normal blending)
        return None

    def is_transitioning(self) -> bool:
        """Check if a transition is currently in progress."""
        return self._state == BlenderState.TRANSITIONING

    def cancel_transition(self) -> None:
        """Cancel any active transition and clear the queue."""
        if self._state == BlenderState.TRANSITIONING:
            self._transition_queue.clear()
            self._state = BlenderState.IDLE

    def reset(self) -> None:
        """Fully reset temporal state for fresh sessions (e.g., after cache reset)."""
        self._transition_queue.clear()
        self._current_blend_embedding = None
        self._state = BlenderState.IDLE
