"""Unified artifact resolution for any graph node.

Artifacts used to live only on ``BasePipelineConfig`` and were resolved
via :class:`PipelineRegistry`. After the node/pipeline unification they
live on :class:`Node` itself (with a default that delegates to the
config class), so this module resolves through :class:`NodeRegistry` —
covering both historical pipelines and plain nodes that need model
weights in a single lookup.
"""

from scope.core.nodes.registry import NodeRegistry
from scope.core.pipelines.artifacts import Artifact


def get_artifacts_for_pipeline(node_type_id: str) -> list[Artifact]:
    """Return the artifacts declared by a node type.

    Args:
        node_type_id: Registry key — either a pipeline id or a plain
            node type id. The argument name stays ``pipeline_id``-shaped
            for backward compatibility with existing callers.

    Returns:
        List of artifacts the node depends on, empty list if none.
    """
    node_class = NodeRegistry.get(node_type_id)
    if node_class is None:
        return []
    return node_class.get_artifacts()
