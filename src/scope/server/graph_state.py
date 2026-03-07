"""Graph configuration store with file-based persistence.

Holds a graph config set via the API. The in-memory copy is the primary store
for fast reads; every mutation is also persisted to ``~/.scope/graph.json`` so
that the graph survives server restarts.

Thread-safe via a threading lock.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

from .graph_schema import GraphConfig

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_graph_config: GraphConfig | None = None

# Persistence path
_PERSIST_DIR = Path.home() / ".scope"
_PERSIST_FILE = _PERSIST_DIR / "graph.json"


def _write_to_file(graph: GraphConfig) -> None:
    """Persist graph config to disk (best-effort)."""
    try:
        _PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        _PERSIST_FILE.write_text(
            json.dumps(graph.model_dump(by_alias=True), indent=2),
            encoding="utf-8",
        )
    except Exception as exc:
        logger.warning(f"Failed to persist graph to {_PERSIST_FILE}: {exc}")


def _read_from_file() -> GraphConfig | None:
    """Load graph config from disk, returning None on any failure."""
    try:
        if not _PERSIST_FILE.exists():
            return None
        data = json.loads(_PERSIST_FILE.read_text(encoding="utf-8"))
        return GraphConfig.model_validate(data)
    except Exception as exc:
        logger.warning(f"Failed to read persisted graph from {_PERSIST_FILE}: {exc}")
        return None


def _delete_file() -> None:
    """Remove the persisted graph file (best-effort)."""
    try:
        if _PERSIST_FILE.exists():
            _PERSIST_FILE.unlink()
    except Exception as exc:
        logger.warning(f"Failed to delete persisted graph at {_PERSIST_FILE}: {exc}")


def get_api_graph() -> GraphConfig | None:
    """Return the graph config – in-memory first, then file fallback."""
    with _lock:
        global _graph_config
        if _graph_config is not None:
            return _graph_config

        # Fallback: try loading from persisted file (e.g. after server restart)
        file_graph = _read_from_file()
        if file_graph is not None:
            _graph_config = file_graph
            logger.info(
                f"Restored graph from {_PERSIST_FILE} "
                f"({len(file_graph.nodes)} nodes, {len(file_graph.edges)} edges)"
            )
        return _graph_config


def set_api_graph(graph: GraphConfig) -> None:
    """Store a graph config set via the API (memory + file)."""
    with _lock:
        global _graph_config
        _graph_config = graph
    # Persist outside the lock to avoid holding it during I/O
    _write_to_file(graph)
    logger.info(
        f"API graph set with {len(graph.nodes)} nodes and {len(graph.edges)} edges"
    )


def clear_api_graph() -> None:
    """Clear the API-set graph config, reverting to fallback behavior."""
    with _lock:
        global _graph_config
        _graph_config = None
    _delete_file()
    logger.info("API graph cleared")
