import logging
from collections import OrderedDict

import torch

logger = logging.getLogger(__name__)

# Numerical stability constants
EPSILON = 1e-8  # Small value to prevent division by zero
SLERP_PARALLEL_THRESHOLD = 1e-4  # Threshold for detecting parallel embeddings in SLERP

# Cache configuration
DEFAULT_MAX_CACHE_SIZE = 10  # Maximum number of prompts to cache

# Logging configuration
LOG_PROMPT_PREVIEW_LENGTH = 50  # Characters to show in log messages for prompt preview

# Prompt defaults
DEFAULT_PROMPT_WEIGHT = 1.0  # Default weight for prompt blending
SLERP_MIN_EMBEDDINGS = 2  # SLERP requires exactly 2 embeddings


def normalize_weights(weights, dtype, device) -> torch.Tensor:
    """Normalize weights to sum to 1.0"""
    weights_tensor = torch.tensor(weights, dtype=dtype, device=device)
    total = weights_tensor.sum()
    if total > 0:
        weights_tensor = weights_tensor / total
    else:
        # Fallback: equal weights for all inputs
        weights_tensor = torch.ones_like(weights_tensor) / len(weights_tensor)
        logger.warning("normalize_weights: All weights zero or negative, using equal weights")
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
    epsilon = 1e-8

    # Compute interpolation coefficients
    coeff1 = torch.sin((1.0 - t) * omega) / (sin_omega + epsilon)
    coeff2 = torch.sin(t * omega) / (sin_omega + epsilon)

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
    if method == "slerp" and len(embeddings) == SLERP_MIN_EMBEDDINGS:
        # Spherical linear interpolation for 2 prompts
        t = normalized_weights[1].item()
        combined_embeds = slerp(embeddings[0], embeddings[1], t)
    else:
        # Linear interpolation (weighted average) with normalization
        # Compute weighted average of norms to preserve magnitude
        target_norm = sum(embed.norm() * weight for embed, weight in zip(embeddings, normalized_weights))

        # Compute linear blend
        combined_embeds = torch.zeros_like(embeddings[0])
        for embed, weight in zip(embeddings, normalized_weights):
            combined_embeds += weight * embed

        # Normalize to preserve embedding magnitude and prevent artifacts
        current_norm = combined_embeds.norm()
        if current_norm > EPSILON:
            combined_embeds = combined_embeds * (target_norm / current_norm)

    return combined_embeds


class PromptBlender:
    """Manages prompt caching and blending for pipelines"""

    def __init__(self, device, dtype, max_cache_size: int = DEFAULT_MAX_CACHE_SIZE) -> None:
        self.device = device
        self.dtype = dtype
        self.max_cache_size = max_cache_size
        self._prompt_cache = OrderedDict()  # LRU cache using OrderedDict
        self._current_prompts = []
        self._interpolation_method = "linear"

    def should_update(self, prompts, interpolation_method) -> bool:
        """Check if prompts or interpolation method changed"""
        if prompts is None:
            return False

        # Compare as tuples for simple equality check
        new_comparable = [(p.get("text", ""), p.get("weight", DEFAULT_PROMPT_WEIGHT)) for p in prompts]
        old_comparable = [(p.get("text", ""), p.get("weight", DEFAULT_PROMPT_WEIGHT)) for p in self._current_prompts]

        return (new_comparable != old_comparable or
                interpolation_method != self._interpolation_method)

    def blend(self, prompts, interpolation_method, text_encoder) -> torch.Tensor | None:
        """Update state and return blended embeddings"""
        self._current_prompts = prompts if prompts else []
        self._interpolation_method = interpolation_method

        return self._encode_and_blend(text_encoder)

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
                    logger.info(f"PromptBlender: Evicted oldest cache entry: {oldest_key[:LOG_PROMPT_PREVIEW_LENGTH]}...")

                logger.info(f"PromptBlender: Encoding and caching prompt: {prompt_text[:LOG_PROMPT_PREVIEW_LENGTH]}...")
                encoded = text_encoder(text_prompts=[prompt_text])
                # Detach from computation graph to prevent memory leak
                self._prompt_cache[prompt_text] = encoded['prompt_embeds'].detach()
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
            embeddings,
            weights,
            self._interpolation_method,
            self.dtype,
            self.device
        )
