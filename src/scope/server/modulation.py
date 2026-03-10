"""Beat-synced parameter modulation engine.

Applies continuous, per-frame parameter modulation driven by beat state.
Pipeline-agnostic: operates on the kwargs dict before it reaches any pipeline.

Usage:
    engine = ModulationEngine()
    engine.update({"noise_scale": {"enabled": True, "shape": "cosine", ...}})
    modulated = engine.apply(beat_state, beats_per_bar=4, params=call_params)
"""

import logging
import math
import threading
from typing import Any, Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

WaveShape = Literal["sine", "cosine", "triangle", "saw", "square", "exp_decay"]
ModulationRate = Literal["half_beat", "beat", "2_beat", "bar", "2_bar", "4_bar"]


class ModulationConfig(BaseModel):
    """Configuration for a single parameter's modulation."""

    enabled: bool = True
    shape: WaveShape = "cosine"
    depth: float = Field(default=0.3, ge=0.0, le=1.0)
    rate: ModulationRate = "bar"
    base_value: float | None = None


def compute_phase(
    beat_phase: float,
    bar_position: float,
    beat_count: int,
    beats_per_bar: int,
    rate: ModulationRate,
) -> float:
    """Compute a 0-1 phase value for the given rate.

    All rates produce a phase that cycles from 0 to 1 over the rate's period.
    """
    if rate == "half_beat":
        return (beat_phase * 2.0) % 1.0
    elif rate == "beat":
        return beat_phase
    elif rate == "2_beat":
        return ((beat_count % 2) + beat_phase) / 2.0
    elif rate == "bar":
        return bar_position / beats_per_bar if beats_per_bar > 0 else 0.0
    elif rate == "2_bar":
        cycle = 2 * beats_per_bar
        pos = (beat_count % cycle) + beat_phase
        return pos / cycle if cycle > 0 else 0.0
    elif rate == "4_bar":
        cycle = 4 * beats_per_bar
        pos = (beat_count % cycle) + beat_phase
        return pos / cycle if cycle > 0 else 0.0
    return 0.0


def wave(shape: WaveShape, phase: float) -> float:
    """Compute wave value in [-1, 1] for oscillators, [0, 1] for exp_decay.

    For oscillators (sine, cosine, triangle, saw, square), the output swings
    symmetrically around 0. For exp_decay, the output is a one-shot pulse
    that fires at phase=0 and decays toward 0.
    """
    phase = phase % 1.0

    if shape == "sine":
        return math.sin(2.0 * math.pi * phase)
    elif shape == "cosine":
        return math.cos(2.0 * math.pi * phase)
    elif shape == "triangle":
        if phase < 0.25:
            return 4.0 * phase
        elif phase < 0.75:
            return 2.0 - 4.0 * phase
        else:
            return -4.0 + 4.0 * phase
    elif shape == "saw":
        return 2.0 * phase - 1.0
    elif shape == "square":
        return 1.0 if phase < 0.5 else -1.0
    elif shape == "exp_decay":
        # Pulse at phase=0, exponential decay over the cycle.
        # tau controls how fast the decay happens (smaller = faster).
        tau = 0.15
        return math.exp(-phase / tau)

    return 0.0


def modulate_value(
    base: float, depth: float, wave_value: float, shape: WaveShape
) -> float:
    """Apply modulation to a base value.

    For symmetric oscillators: base + depth * wave_value
    For exp_decay: base + depth * wave_value (wave is already [0, 1])
    """
    return base + depth * wave_value


class ModulationEngine:
    """Applies beat-synced modulation to pipeline parameters.

    Thread-safe: config updates (from the parameter update path) and
    apply() calls (from the processing thread) are protected by a lock.
    """

    def __init__(self) -> None:
        self._configs: dict[str, ModulationConfig] = {}
        self._lock = threading.Lock()

    def update(self, raw_configs: dict[str, Any]) -> None:
        """Update modulation configs from a frontend-sent dict.

        Args:
            raw_configs: Mapping of parameter name to config dict, e.g.
                {"noise_scale": {"enabled": True, "shape": "cosine", ...}}
        """
        new_configs: dict[str, ModulationConfig] = {}
        for param_name, cfg in raw_configs.items():
            try:
                new_configs[param_name] = ModulationConfig.model_validate(cfg)
            except Exception:
                logger.warning(f"Invalid modulation config for '{param_name}': {cfg}")
        with self._lock:
            self._configs = new_configs

    def apply(
        self,
        beat_phase: float,
        bar_position: float,
        beat_count: int,
        beats_per_bar: int,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Apply all active modulations to the params dict.

        Modifies and returns the same dict. Only numeric params that are
        already present in the dict are modulated.
        """
        with self._lock:
            if not self._configs:
                return params
            configs = dict(self._configs)

        for param_name, config in configs.items():
            if not config.enabled:
                continue

            if param_name in params:
                base = params[param_name]
                if not isinstance(base, (int, float)):
                    continue
                base = float(base)
            elif config.base_value is not None:
                base = config.base_value
            else:
                continue

            phase = compute_phase(
                beat_phase, bar_position, beat_count, beats_per_bar, config.rate
            )
            w = wave(config.shape, phase)
            params[param_name] = modulate_value(base, config.depth, w, config.shape)

        return params

    def get_configs(self) -> dict[str, dict]:
        """Return a snapshot of active configs (for status/debugging)."""
        with self._lock:
            return {name: cfg.model_dump() for name, cfg in self._configs.items()}

    def clear(self) -> None:
        """Remove all modulations."""
        with self._lock:
            self._configs.clear()
