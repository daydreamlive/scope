"""Graph session — ties graph engine execution to a WebRTC/headless session.

Each active session (WebRTC or headless) may have at most one ``GraphSession``
that owns a ``GraphEngine`` instance.  The session wires the engine's value
change callback to the ``NotificationSender`` so that node values stream back
to the frontend over the existing data channel.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from scope.core.nodes.engine import GraphEngine
from scope.core.nodes.schema import GraphDefinition

if TYPE_CHECKING:
    from collections.abc import Callable

    from scope.server.tempo_sync import TempoSync

logger = logging.getLogger(__name__)


class GraphSession:
    """Manages graph engine lifecycle for one session."""

    def __init__(
        self,
        session_id: str,
        tempo_sync: TempoSync | None = None,
        notification_callback: Callable[[dict[str, Any]], None] | None = None,
        pipeline_param_callback: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self._session_id = session_id
        self.engine = GraphEngine(session_id, tempo_sync)

        # Wire callbacks
        if notification_callback is not None:
            self.engine._value_change_callback = self._make_value_sender(
                notification_callback
            )
        if pipeline_param_callback is not None:
            self.engine._pipeline_param_callback = pipeline_param_callback

    def load_and_start(
        self,
        graph_def: GraphDefinition,
    ) -> None:
        """Load a graph definition and start the tick loop."""
        self.engine.load_graph(graph_def)
        self.engine.start()
        logger.info("Graph session started for %s", self._session_id)

    def update_graph(self, graph_def: GraphDefinition) -> None:
        """Hot-update: stop current engine, reload, and restart."""
        self.engine.stop()
        self.engine.load_graph(graph_def)
        self.engine.start()
        logger.info("Graph session updated for %s", self._session_id)

    def handle_event(
        self, node_id: str, event_type: str, payload: dict[str, Any]
    ) -> None:
        """Route a frontend interaction to the graph engine."""
        self.engine.handle_event(node_id, event_type, payload)

    def stop(self) -> None:
        """Stop the graph engine and clean up."""
        self.engine.stop()
        logger.info("Graph session stopped for %s", self._session_id)

    def get_all_values(self) -> dict[str, dict[str, Any]]:
        """Full snapshot for initial sync on (re)connect."""
        return self.engine.get_all_values()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_value_sender(
        callback: Callable[[dict[str, Any]], None],
    ) -> Callable[[dict[str, dict[str, Any]]], None]:
        """Wrap the notification callback to send ``node_values`` messages."""

        def sender(changed: dict[str, dict[str, Any]]) -> None:
            callback({"type": "node_values", "values": changed})

        return sender
