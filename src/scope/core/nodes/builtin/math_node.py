"""Math node — performs unary/binary arithmetic operations.

Ported from ``frontend/src/components/graph/utils/computeResult.ts``.
"""

from __future__ import annotations

import math
from typing import Any, ClassVar

from scope.server.tempo_sync import BeatState

from ..base import BaseNode, NodeDefinition, NodePort

UNARY_OPS = {"abs", "negate", "sqrt", "floor", "ceil", "round", "toInt", "toFloat"}
BINARY_OPS = {"add", "subtract", "multiply", "divide", "mod", "min", "max", "power"}


def compute_result(op: str, a: float | None, b: float | None) -> float | None:
    """Pure math evaluation matching the frontend ``computeResult``."""
    if a is None:
        return None

    # Unary ops
    if op == "abs":
        return abs(a)
    if op == "negate":
        return -a
    if op == "sqrt":
        return math.sqrt(a) if a >= 0 else None
    if op == "floor":
        return float(math.floor(a))
    if op == "ceil":
        return float(math.ceil(a))
    if op == "round":
        return float(round(a))
    if op == "toInt":
        return float(math.trunc(a))
    if op == "toFloat":
        return float(a)

    # Binary ops require b
    if b is None:
        return None

    if op == "add":
        return a + b
    if op == "subtract":
        return a - b
    if op == "multiply":
        return a * b
    if op == "divide":
        return a / b if b != 0 else None
    if op == "mod":
        return a % b if b != 0 else None
    if op == "min":
        return min(a, b)
    if op == "max":
        return max(a, b)
    if op == "power":
        return a**b

    return None


class MathNode(BaseNode):
    """Performs configurable arithmetic on one or two inputs."""

    node_type_id: ClassVar[str] = "math"

    @classmethod
    def get_definition(cls) -> NodeDefinition:
        return NodeDefinition(
            node_type_id=cls.node_type_id,
            display_name="Math",
            category="math",
            description="Performs arithmetic operations on one or two inputs.",
            inputs=[
                NodePort(name="a", port_type="number", default_value=0),
                NodePort(name="b", port_type="number", default_value=0),
            ],
            outputs=[
                NodePort(name="value", port_type="number"),
            ],
        )

    def execute(
        self,
        inputs: dict[str, Any],
        tick_time: float,
        beat_state: BeatState | None,
    ) -> dict[str, Any]:
        op = self.config.get("mathOp", "add")
        default_a = self.config.get("mathDefaultA", 0)
        default_b = self.config.get("mathDefaultB", 0)

        a = inputs.get("a", default_a)
        b = inputs.get("b", default_b)

        a = _to_float(a, default_a)
        b = _to_float(b, default_b)

        result = compute_result(op, a, b)

        output_type = self.config.get("mathOutputType", "auto")
        if result is not None and output_type == "int":
            result = float(int(result))

        return {"value": result}


def _to_float(v: Any, default: float) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return default
