"""Local custom node loader — ComfyUI-style directory scanning.

Scans ``~/.daydream-scope/custom_nodes/`` (and an optional env-var
override ``SCOPE_CUSTOM_NODES_DIR``) for Python modules that export a
``NODE_CLASS_MAPPINGS`` dictionary.  This mirrors ComfyUI's discovery
pattern and provides a zero-packaging development workflow.

Directory structure expected::

    ~/.daydream-scope/custom_nodes/
        my_node_pack/
            __init__.py          # must export NODE_CLASS_MAPPINGS
            my_awesome_node.py

``NODE_CLASS_MAPPINGS`` maps node_type_id strings to BaseNode subclasses::

    from scope.core.nodes import BaseNode
    from .my_awesome_node import MyAwesomeNode

    NODE_CLASS_MAPPINGS = {
        "my_pack.awesome": MyAwesomeNode,
    }
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_DIR = Path.home() / ".daydream-scope" / "custom_nodes"


def _custom_nodes_dir() -> Path:
    override = os.environ.get("SCOPE_CUSTOM_NODES_DIR")
    if override:
        return Path(override)
    return _DEFAULT_DIR


def load_local_nodes() -> dict[str, type]:
    """Scan the custom_nodes directory and return discovered node classes.

    Returns:
        Dict mapping node_type_id → BaseNode subclass.
    """
    base_dir = _custom_nodes_dir()
    if not base_dir.is_dir():
        return {}

    discovered: dict[str, type] = {}

    # Add the custom_nodes dir to sys.path so that imports within
    # node packs resolve (e.g. from .my_module import ...).
    base_str = str(base_dir)
    if base_str not in sys.path:
        sys.path.insert(0, base_str)

    for entry in sorted(base_dir.iterdir()):
        if not entry.is_dir():
            continue
        init = entry / "__init__.py"
        if not init.exists():
            continue
        module_name = entry.name
        try:
            mod = importlib.import_module(module_name)
            mappings = getattr(mod, "NODE_CLASS_MAPPINGS", None)
            if mappings and isinstance(mappings, dict):
                for node_type_id, node_cls in mappings.items():
                    discovered[node_type_id] = node_cls
                    logger.info(
                        "Loaded local custom node: %s (from %s)",
                        node_type_id,
                        module_name,
                    )
            else:
                logger.debug("Skipped %s: no NODE_CLASS_MAPPINGS found", module_name)
        except Exception:
            logger.warning(
                "Failed to load custom node pack '%s'",
                module_name,
                exc_info=True,
            )

    return discovered
