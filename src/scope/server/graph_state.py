"""In-memory graph configuration store.

Holds a graph config set via the API that takes priority over input.json.
Thread-safe via a threading lock.
"""

from __future__ import annotations

import logging
import threading

from .graph_schema import GraphConfig

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_graph_config: GraphConfig | None = None


def get_api_graph() -> GraphConfig | None:
    """Return the API-set graph config, or None if not set."""
    with _lock:
        return _graph_config


def set_api_graph(graph: GraphConfig) -> None:
    """Store a graph config set via the API."""
    with _lock:
        global _graph_config
        _graph_config = graph
    logger.info(
        f"API graph set with {len(graph.nodes)} nodes and {len(graph.edges)} edges"
    )


def clear_api_graph() -> None:
    """Clear the API-set graph config, reverting to fallback behavior."""
    with _lock:
        global _graph_config
        _graph_config = None
    logger.info("API graph cleared")
