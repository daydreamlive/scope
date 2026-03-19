"""Control node — animated value patterns and switch-mode selection.

Ported from ``frontend/src/components/graph/utils/computePatternValue.ts``
and ``frontend/src/components/graph/nodes/ControlNode.tsx``.
"""

from __future__ import annotations

import math
import random
from typing import Any, ClassVar

from scope.server.tempo_sync import BeatState

from ..base import BaseNode, NodeDefinition, NodePort


def compute_pattern_value(
    pattern: str,
    t: float,
    speed: float,
    min_val: float,
    max_val: float,
    last_value: float,
) -> float:
    """Compute an animated value from a control-node pattern.

    Matches the frontend ``computePatternValue`` exactly.
    """
    rng = max_val - min_val
    phase = (t * speed) % 1.0

    if pattern == "sine":
        return min_val + rng * (0.5 + 0.5 * math.sin(phase * 2 * math.pi))

    if pattern == "bounce":
        triangle = phase * 2.0 if phase < 0.5 else 2.0 - phase * 2.0
        return min_val + rng * triangle

    if pattern == "random_walk":
        step = (random.random() - 0.5) * 0.1 * rng
        new_value = last_value + step
        return max(min_val, min(max_val, new_value))

    if pattern == "linear":
        return min_val + rng * phase

    if pattern == "step":
        steps = 10
        step_index = int(phase * steps)
        return min_val + (rng * step_index) / (steps - 1) if steps > 1 else min_val

    return min_val


class ControlNode(BaseNode):
    """Produces animated values (sine, bounce, random_walk, …) or switches."""

    node_type_id: ClassVar[str] = "control"

    @classmethod
    def get_definition(cls) -> NodeDefinition:
        return NodeDefinition(
            node_type_id=cls.node_type_id,
            display_name="Control",
            category="control",
            description="Animated value patterns or switch-mode string selector.",
            inputs=[
                NodePort(name="speed", port_type="number"),
                NodePort(name="min", port_type="number"),
                NodePort(name="max", port_type="number"),
            ],
            outputs=[
                NodePort(name="value", port_type="any"),
            ],
            is_animated=True,
        )

    def execute(
        self,
        inputs: dict[str, Any],
        tick_time: float,
        beat_state: BeatState | None,
    ) -> dict[str, Any]:
        mode = self.config.get("controlMode", "animated")

        if mode == "switch":
            return self._execute_switch(inputs)
        return self._execute_animated(inputs, tick_time)

    # ------------------------------------------------------------------
    # Animated mode
    # ------------------------------------------------------------------
    def _execute_animated(
        self, inputs: dict[str, Any], tick_time: float
    ) -> dict[str, Any]:
        pattern = self.config.get("controlPattern", "sine")
        speed = _num(inputs.get("speed"), self.config.get("controlSpeed", 1.0))
        min_val = _num(inputs.get("min"), self.config.get("controlMin", 0.0))
        max_val = _num(inputs.get("max"), self.config.get("controlMax", 1.0))
        last_value = self._state.get("last_value", (min_val + max_val) / 2.0)

        value = compute_pattern_value(
            pattern, tick_time, speed, min_val, max_val, last_value
        )
        self._state["last_value"] = value
        return {"value": value}

    # ------------------------------------------------------------------
    # Switch mode
    # ------------------------------------------------------------------
    def _execute_switch(self, inputs: dict[str, Any]) -> dict[str, Any]:
        items: list[str] = self.config.get("controlItems", [])
        if not items:
            return {"value": None}

        # Check for numeric input items (item_0, item_1, …) to trigger switch
        for i, _item in enumerate(items):
            key = f"item_{i}"
            val = inputs.get(key)
            if val is not None:
                try:
                    fval = float(val)
                except (TypeError, ValueError):
                    continue
                prev = self._state.get(f"prev_{key}", 0.0)
                self._state[f"prev_{key}"] = fval
                # Rising-edge detection
                if fval > 0.5 and prev <= 0.5:
                    self._state["selected_index"] = i

        idx = self._state.get("selected_index", 0)
        idx = max(0, min(idx, len(items) - 1))
        return {"value": items[idx]}


def _num(v: Any, default: float) -> float:
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default
