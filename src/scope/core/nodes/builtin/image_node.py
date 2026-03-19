"""Image node — references an image or video asset."""

from __future__ import annotations

from typing import Any, ClassVar

from scope.server.tempo_sync import BeatState

from ..base import BaseNode, NodeDefinition, NodePort


class ImageNode(BaseNode):
    """Holds a path to an image or video asset."""

    node_type_id: ClassVar[str] = "image"

    @classmethod
    def get_definition(cls) -> NodeDefinition:
        return NodeDefinition(
            node_type_id=cls.node_type_id,
            display_name="Image",
            category="input",
            description="Image or video asset reference.",
            inputs=[],
            outputs=[
                NodePort(name="value", port_type="string"),
                NodePort(name="video_value", port_type="string"),
            ],
        )

    def execute(
        self,
        inputs: dict[str, Any],
        tick_time: float,
        beat_state: BeatState | None,
    ) -> dict[str, Any]:
        path = self.config.get("imagePath", "")
        media_type = self.config.get("mediaType", "image")
        if media_type == "video":
            return {"value": path, "video_value": path}
        return {"value": path, "video_value": None}

    def on_event(self, event_type: str, payload: dict[str, Any]) -> None:
        if event_type == "value_change":
            if "imagePath" in payload:
                self.config["imagePath"] = payload["imagePath"]
            if "mediaType" in payload:
                self.config["mediaType"] = payload["mediaType"]
