import type { Node, Edge } from "@xyflow/react";
import type { DagConfig, DagNode, DagEdge, PipelineSchemaInfo } from "./api";

// Layout constants
const NODE_WIDTH = 200;
const NODE_HEIGHT = 60;
const COLUMN_GAP = 300;
const ROW_GAP = 100;
const START_X = 50;
const START_Y = 50;

export interface PortInfo {
  name: string;
}

export interface FlowNodeData {
  label: string;
  pipelineId?: string | null;
  nodeType: "source" | "pipeline" | "sink";
  availablePipelineIds?: string[];
  /** Declared input ports for the selected pipeline */
  streamInputs?: string[];
  /** Declared output ports for the selected pipeline */
  streamOutputs?: string[];
  /** Pipeline schemas keyed by pipeline_id, for looking up ports on selection change */
  pipelinePortsMap?: Record<string, { inputs: string[]; outputs: string[] }>;
  [key: string]: unknown;
}

/**
 * Build a map of pipeline_id -> { inputs, outputs } from schemas.
 */
export function buildPipelinePortsMap(
  schemas: Record<string, PipelineSchemaInfo>
): Record<string, { inputs: string[]; outputs: string[] }> {
  const map: Record<string, { inputs: string[]; outputs: string[] }> = {};
  for (const [id, schema] of Object.entries(schemas)) {
    map[id] = {
      inputs: schema.stream_inputs ?? ["video"],
      outputs: schema.stream_outputs ?? ["video"],
    };
  }
  return map;
}

/**
 * Convert backend DagConfig to React Flow nodes and edges.
 * Auto-layout: sources on the left, pipelines in the middle, sinks on the right.
 */
export function dagConfigToFlow(
  dag: DagConfig,
  portsMap?: Record<string, { inputs: string[]; outputs: string[] }>
): {
  nodes: Node<FlowNodeData>[];
  edges: Edge[];
} {
  const sources = dag.nodes.filter(n => n.type === "source");
  const pipelines = dag.nodes.filter(n => n.type === "pipeline");
  const sinks = dag.nodes.filter(n => n.type === "sink");

  const nodes: Node<FlowNodeData>[] = [];

  // Layout sources (column 0)
  sources.forEach((n, i) => {
    nodes.push({
      id: n.id,
      type: "source",
      position: { x: START_X, y: START_Y + i * (NODE_HEIGHT + ROW_GAP) },
      data: { label: n.id, nodeType: "source" },
    });
  });

  // Layout pipelines (column 1)
  pipelines.forEach((n, i) => {
    const ports = n.pipeline_id && portsMap ? portsMap[n.pipeline_id] : null;
    nodes.push({
      id: n.id,
      type: "pipeline",
      position: {
        x: START_X + COLUMN_GAP,
        y: START_Y + i * (NODE_HEIGHT + ROW_GAP),
      },
      data: {
        label: n.pipeline_id || n.id,
        pipelineId: n.pipeline_id,
        nodeType: "pipeline",
        streamInputs: ports?.inputs ?? ["video"],
        streamOutputs: ports?.outputs ?? ["video"],
      },
    });
  });

  // Layout sinks (column 2)
  sinks.forEach((n, i) => {
    nodes.push({
      id: n.id,
      type: "sink",
      position: {
        x: START_X + COLUMN_GAP * 2,
        y: START_Y + i * (NODE_HEIGHT + ROW_GAP),
      },
      data: { label: n.id, nodeType: "sink" },
    });
  });

  // Convert edges
  const edges: Edge[] = dag.edges.map((e, i) => ({
    id: `e-${i}-${e.from}-${e.to_node}`,
    source: e.from,
    sourceHandle: e.from_port,
    target: e.to_node,
    targetHandle: e.to_port,
    label: e.from_port !== "video" ? e.from_port : undefined,
    animated: true,
  }));

  return { nodes, edges };
}

/**
 * Convert React Flow state back to backend DagConfig JSON.
 */
export function flowToDagConfig(
  nodes: Node<FlowNodeData>[],
  edges: Edge[]
): DagConfig {
  const dagNodes: DagNode[] = nodes.map(n => ({
    id: n.id,
    type: n.data.nodeType,
    pipeline_id:
      n.data.nodeType === "pipeline" ? (n.data.pipelineId ?? null) : undefined,
  }));

  const dagEdges: DagEdge[] = edges.map(e => ({
    from: e.source,
    from_port: e.sourceHandle || "video",
    to_node: e.target,
    to_port: e.targetHandle || "video",
    kind: "stream" as const,
  }));

  return { nodes: dagNodes, edges: dagEdges };
}

/**
 * Generate a unique node ID with a given prefix.
 */
export function generateNodeId(
  prefix: string,
  existingIds: Set<string>
): string {
  if (!existingIds.has(prefix)) return prefix;
  let i = 1;
  while (existingIds.has(`${prefix}_${i}`)) i++;
  return `${prefix}_${i}`;
}

// Default node dimensions for reference
export { NODE_WIDTH, NODE_HEIGHT };
