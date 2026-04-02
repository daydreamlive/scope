"""Backend node framework for stateful, event-driven graph nodes.

Nodes are lightweight processing units that produce and consume discrete values
and events. Unlike pipelines (which process video frames), nodes handle control
flow, timing, and parameter orchestration within the execution graph.

The framework is frontend-independent: the backend can execute the full graph
of nodes and pipelines without any browser or UI connected.
"""

from scope.core.nodes.base_schema import BaseNodeConfig
from scope.core.nodes.interface import BaseNode, ConnectorDef
from scope.core.nodes.registry import NodeRegistry

__all__ = ["BaseNode", "BaseNodeConfig", "ConnectorDef", "NodeRegistry"]
