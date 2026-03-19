"""Note node — annotation-only, no execution logic."""

from __future__ import annotations

from typing import Any, ClassVar

from scope.server.tempo_sync import BeatState

from ..base import BaseNode, NodeDefinition


class NoteNode(BaseNode):
    """Annotation node. Has no inputs, outputs, or execution."""

    node_type_id: ClassVar[str] = "note"

    @classmethod
    def get_definition(cls) -> NodeDefinition:
        return NodeDefinition(
            node_type_id=cls.node_type_id,
            display_name="Note",
            category="utility",
            description="Text annotation; no execution.",
        )

    def execute(
        self,
        inputs: dict[str, Any],
        tick_time: float,
        beat_state: BeatState | None,
    ) -> dict[str, Any]:
        return {}
