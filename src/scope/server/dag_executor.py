"""DAG executor for wiring pipeline processors into a directed acyclic graph."""

import logging
import queue
from collections import defaultdict

from scope.core.pipelines.dag import DagGraph

from .pipeline_processor import PipelineProcessor

logger = logging.getLogger(__name__)


class BroadcastQueue:
    """Copies each put_nowait() item to multiple downstream queues.

    Used for fan-out: when one source port feeds multiple target ports.
    """

    def __init__(self, targets: list[queue.Queue]):
        self.targets = targets

    def put_nowait(self, item):
        for t in self.targets:
            try:
                t.put_nowait(item)
            except queue.Full:
                pass


class DagExecutor:
    """Sets up PipelineProcessors wired as a DAG based on a DagGraph definition."""

    def setup(
        self,
        graph: DagGraph,
        pipeline_manager,
        parameters: dict,
        session_id: str | None = None,
        user_id: str | None = None,
        connection_id: str | None = None,
        connection_info: dict | None = None,
    ) -> tuple[
        list[PipelineProcessor],
        "queue.Queue | BroadcastQueue",  # external input
        queue.Queue,  # final output
    ]:
        """Wire up processors according to the DAG graph.

        Returns:
            (processors, external_input_queue, final_output_queue)
        """
        processors: dict[str, PipelineProcessor] = {}

        # 1. Create a PipelineProcessor for each node
        for node in graph.nodes:
            pipeline = pipeline_manager.get_pipeline_by_id(node.pipeline_id)
            proc = PipelineProcessor(
                pipeline=pipeline,
                pipeline_id=node.pipeline_id,
                initial_parameters=parameters.copy(),
                session_id=session_id,
                user_id=user_id,
                connection_id=connection_id,
                connection_info=connection_info,
            )
            proc._dag_mode = True
            processors[node.id] = proc

        # 2. Group edges by (source_node, source_port) to detect fan-out
        source_groups: dict[tuple[str, str], list] = defaultdict(list)
        for edge in graph.edges:
            source_groups[(edge.source_node, edge.source_port)].append(edge)

        # 3. Wire edges: create queues and connect ports
        external_queues: list[queue.Queue] = []  # queues fed by __input__
        external_input_q = None

        for (src_node, src_port), edges in source_groups.items():
            # Create a target queue for each edge and assign to target's input_queues
            target_queues: list[queue.Queue] = []
            for edge in edges:
                target_proc = processors[edge.target_node]
                q = queue.Queue(maxsize=30)
                target_proc.input_queues[edge.target_port] = q
                target_queues.append(q)

            if src_node == "__input__":
                # External input edges
                external_queues.extend(target_queues)
            else:
                # Internal edges: assign to source's output_queues
                src_proc = processors[src_node]
                if len(target_queues) == 1:
                    src_proc.output_queues[src_port] = target_queues[0]
                else:
                    # Fan-out: wrap multiple targets in BroadcastQueue
                    src_proc.output_queues[src_port] = BroadcastQueue(target_queues)

        # 4. Build external input queue (or BroadcastQueue for fan-out)
        if len(external_queues) == 1:
            external_input_q = external_queues[0]
        elif len(external_queues) > 1:
            external_input_q = BroadcastQueue(external_queues)
        else:
            # No external input edges - create a dummy queue
            external_input_q = queue.Queue(maxsize=30)

        # 5. Final output: output_node's output_queues[output_port]
        output_proc = processors[graph.output_node]
        final_output_q = queue.Queue(maxsize=8)
        # If there's already an output queue for this port (from internal edges),
        # we need to also feed the final output queue
        existing = output_proc.output_queues.get(graph.output_port)
        if existing is not None:
            # There are internal consumers AND we need final output
            if isinstance(existing, BroadcastQueue):
                existing.targets.append(final_output_q)
            else:
                # existing is a single queue for an internal edge, wrap both
                output_proc.output_queues[graph.output_port] = BroadcastQueue(
                    [existing, final_output_q]
                )
        else:
            output_proc.output_queues[graph.output_port] = final_output_q

        processor_list = list(processors.values())

        logger.info(
            f"DAG executor wired {len(processor_list)} processors, "
            f"{len(graph.edges)} edges"
        )

        return processor_list, external_input_q, final_output_q
