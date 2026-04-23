"""``Pipeline`` is an alias for :class:`scope.core.nodes.Node`.

Historically ``Pipeline`` was a separate base class that subclassed
``BaseNode``. After the node/pipeline unification, a pipeline is just
a config-driven node — one whose ``get_config_class()`` returns a
:class:`BasePipelineConfig`. The alias lives on so existing pipeline
subclasses and plugins keep working with no code change.
"""

from scope.core.nodes.base import Node, Requirements

Pipeline = Node

__all__ = ["Pipeline", "Requirements"]
