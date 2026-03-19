"""Bool node — gate mode (threshold) or toggle mode (stateful latch)."""

from __future__ import annotations

from typing import Any, ClassVar

from scope.server.tempo_sync import BeatState

from ..base import BaseNode, NodeDefinition, NodePort


class BoolNode(BaseNode):
    """Outputs 1 or 0 based on gate threshold or toggle state."""

    node_type_id: ClassVar[str] = "bool"

    @classmethod
    def get_definition(cls) -> NodeDefinition:
        return NodeDefinition(
            node_type_id=cls.node_type_id,
            display_name="Bool",
            category="utility",
            description="Gate (threshold) or toggle (stateful latch) boolean output.",
            inputs=[
                NodePort(name="input", port_type="number"),
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
        mode = self.config.get("boolMode", "gate")
        threshold = float(self.config.get("boolThreshold", 0.5))
        input_val = _to_float(inputs.get("input"))

        if mode == "toggle":
            return self._toggle(input_val, threshold)
        return self._gate(input_val, threshold)

    def _gate(self, input_val: float | None, threshold: float) -> dict[str, Any]:
        if input_val is None:
            return {"value": float(self.config.get("value", 0))}
        return {"value": 1.0 if input_val > threshold else 0.0}

    def _toggle(self, input_val: float | None, threshold: float) -> dict[str, Any]:
        prev = self._state.get("prev_input", 0.0)
        toggled = self._state.get("toggled", bool(self.config.get("value", False)))

        if input_val is not None:
            # Rising-edge detection
            if input_val > threshold and prev <= threshold:
                toggled = not toggled
            self._state["prev_input"] = input_val
        self._state["toggled"] = toggled
        return {"value": 1.0 if toggled else 0.0}

    def on_event(self, event_type: str, payload: dict[str, Any]) -> None:
        if event_type == "toggle":
            current = self._state.get("toggled", False)
            self._state["toggled"] = not current
        elif event_type == "value_change" and "value" in payload:
            self.config["value"] = payload["value"]


def _to_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
