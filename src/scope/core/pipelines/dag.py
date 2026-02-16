"""DAG pipeline graph definition models."""

from pydantic import BaseModel


class DagNode(BaseModel):
    """A pipeline node in the DAG."""

    id: str  # unique node instance ID (e.g. "yolo_1")
    pipeline_id: str  # pipeline type from registry (e.g. "yolo_mask")


class DagEdge(BaseModel):
    """A connection between two pipeline ports."""

    source_node: str  # node id (or "__input__" for external video)
    source_port: str  # output port name (e.g. "vace_input_frames")
    target_node: str  # node id
    target_port: str  # input port name


class DagGraph(BaseModel):
    """DAG pipeline graph definition."""

    nodes: list[DagNode]
    edges: list[DagEdge]
    output_node: str  # which node produces the final output
    output_port: str = "video"  # which port on that node
