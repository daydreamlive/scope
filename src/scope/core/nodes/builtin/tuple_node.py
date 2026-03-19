"""Tuple node — ordered multi-value container."""

from __future__ import annotations

from typing import Any, ClassVar

from scope.server.tempo_sync import BeatState

from ..base import BaseNode, NodeDefinition, NodePort


class TupleNode(BaseNode):
    """Holds an ordered list of numeric values with optional clamping."""

    node_type_id: ClassVar[str] = "tuple"

    @classmethod
    def get_definition(cls) -> NodeDefinition:
        return NodeDefinition(
            node_type_id=cls.node_type_id,
            display_name="Tuple",
            category="input",
            description="Multi-value container with optional ordering.",
            inputs=[],
            outputs=[
                NodePort(name="value", port_type="any"),
            ],
        )

    def execute(
        self,
        inputs: dict[str, Any],
        tick_time: float,
        beat_state: BeatState | None,
    ) -> dict[str, Any]:
        values: list[Any] = list(self.config.get("tupleValues", []))
        min_val = self.config.get("tupleMin")
        max_val = self.config.get("tupleMax")

        # Apply input overrides (input_0, input_1, …)
        for i in range(len(values)):
            key = f"input_{i}"
            if key in inputs and inputs[key] is not None:
                try:
                    values[i] = float(inputs[key])
                except (TypeError, ValueError):
                    values[i] = inputs[key]

        # Clamp numeric values if bounds are set
        if min_val is not None or max_val is not None:
            lo = float(min_val) if min_val is not None else float("-inf")
            hi = float(max_val) if max_val is not None else float("inf")
            for i, v in enumerate(values):
                try:
                    values[i] = max(lo, min(hi, float(v)))
                except (TypeError, ValueError):
                    pass

        # Enforce ordering if configured
        enforce = self.config.get("tupleEnforceOrder", False)
        if enforce and len(values) > 1:
            direction = self.config.get("tupleOrderDirection", "ascending")
            if direction == "ascending":
                for i in range(1, len(values)):
                    try:
                        if float(values[i]) < float(values[i - 1]):
                            values[i] = values[i - 1]
                    except (TypeError, ValueError):
                        pass
            elif direction == "descending":
                for i in range(1, len(values)):
                    try:
                        if float(values[i]) > float(values[i - 1]):
                            values[i] = values[i - 1]
                    except (TypeError, ValueError):
                        pass

        return {"value": values}

    def on_event(self, event_type: str, payload: dict[str, Any]) -> None:
        if event_type == "value_change" and "values" in payload:
            self.config["tupleValues"] = payload["values"]
