"""In-memory DAG configuration store.

Holds a DAG config set via the API that takes priority over input.json.
Thread-safe via a threading lock.
"""

from __future__ import annotations

import logging
import threading

from .dag_schema import DagConfig

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_dag_config: DagConfig | None = None


def get_api_dag() -> DagConfig | None:
    """Return the API-set DAG config, or None if not set."""
    with _lock:
        return _dag_config


def set_api_dag(dag: DagConfig) -> None:
    """Store a DAG config set via the API."""
    with _lock:
        global _dag_config
        _dag_config = dag
    logger.info(f"API DAG set with {len(dag.nodes)} nodes and {len(dag.edges)} edges")


def clear_api_dag() -> None:
    """Clear the API-set DAG config, reverting to fallback behavior."""
    with _lock:
        global _dag_config
        _dag_config = None
    logger.info("API DAG cleared")
