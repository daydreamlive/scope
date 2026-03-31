"""Base Pydantic schema for node configuration.

Every backend node type must define a config class inheriting from
``BaseNodeConfig``. The config declares the node's identity, category,
and its static input/output ports.  The ``get_schema_for_frontend()``
method produces a JSON-serialisable dict that the frontend uses to
auto-render the node without any custom component code.
"""

from typing import Any, ClassVar

from pydantic import BaseModel

from .interface import ConnectorDef


class BaseNodeConfig(BaseModel):
    """Base configuration shared by all backend node types.

    Class variables (``ClassVar``) define the node's identity and static port
    layout.  They are not instance fields -- they describe the *type* of node,
    not a particular instance.
    """

    node_type_id: ClassVar[str] = ""
    """Unique registry key, e.g. ``"scheduler"``."""

    node_name: ClassVar[str] = ""
    """Human-readable display name, e.g. ``"Scheduler"``."""

    node_description: ClassVar[str] = ""
    """Short description of what the node does."""

    node_version: ClassVar[str] = "0.1.0"
    """Semantic version string."""

    node_category: ClassVar[str] = "general"
    """Category for grouping in the add-node menu (e.g. ``"timing"``, ``"math"``)."""

    inputs: ClassVar[list[ConnectorDef]] = []
    """Static input port definitions."""

    outputs: ClassVar[list[ConnectorDef]] = []
    """Static output port definitions."""

    dynamic_ports: ClassVar[bool] = False
    """Whether this node can add/remove ports at runtime."""

    @classmethod
    def get_schema_for_frontend(cls) -> dict[str, Any]:
        """Return a JSON-serialisable schema for frontend auto-rendering.

        The returned dict contains everything the generic ``BackendNode``
        component needs to render handles, labels, and inline controls.
        """
        return {
            "node_type_id": cls.node_type_id,
            "node_name": cls.node_name,
            "node_description": cls.node_description,
            "node_version": cls.node_version,
            "node_category": cls.node_category,
            "inputs": [c.model_dump() for c in cls.inputs],
            "outputs": [c.model_dump() for c in cls.outputs],
            "dynamic_ports": cls.dynamic_ports,
        }
