"""Schema / config for the Scheduler node."""

from typing import ClassVar

from scope.core.nodes.base_schema import BaseNodeConfig
from scope.core.nodes.interface import ConnectorDef


class SchedulerNodeConfig(BaseNodeConfig):
    """Configuration for the Scheduler node.

    The scheduler has a ``start`` / ``reset`` trigger input, a ``tick``
    trigger output that fires at each trigger point, and supports dynamic
    named trigger outputs added at runtime.
    """

    node_type_id: ClassVar[str] = "scheduler"
    node_name: ClassVar[str] = "Scheduler"
    node_description: ClassVar[str] = (
        "Time-based trigger scheduler. Add trigger points and connect them "
        "to other nodes to drive actions at specific moments."
    )
    node_version: ClassVar[str] = "0.1.0"
    node_category: ClassVar[str] = "timing"

    inputs: ClassVar[list[ConnectorDef]] = [
        ConnectorDef(
            name="start",
            type="trigger",
            direction="input",
            ui={"widget": "play_button"},
        ),
        ConnectorDef(
            name="reset",
            type="trigger",
            direction="input",
        ),
    ]

    outputs: ClassVar[list[ConnectorDef]] = [
        ConnectorDef(
            name="tick",
            type="trigger",
            direction="output",
        ),
        ConnectorDef(
            name="elapsed",
            type="float",
            direction="output",
            default=0.0,
        ),
        ConnectorDef(
            name="is_playing",
            type="bool",
            direction="output",
            default=False,
        ),
    ]

    dynamic_ports: ClassVar[bool] = True
