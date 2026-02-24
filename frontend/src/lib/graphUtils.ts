import type { Node, Edge } from "@xyflow/react";
import type {
  GraphConfig,
  GraphNode,
  GraphEdge,
  PipelineSchemaInfo,
} from "./api";

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
      inputs: schema.inputs ?? ["video"],
      outputs: schema.outputs ?? ["video"],
    };
  }
  return map;
}

/**
 * Convert backend GraphConfig to React Flow nodes and edges.
 * Auto-layout: sources on the left, pipelines in the middle, sinks on the right.
 */
export function graphConfigToFlow(
  graph: GraphConfig,
  portsMap?: Record<string, { inputs: string[]; outputs: string[] }>
): {
  nodes: Node<FlowNodeData>[];
  edges: Edge[];
} {
  const sources = graph.nodes.filter(n => n.type === "source");
  const pipelines = graph.nodes.filter(n => n.type === "pipeline");
  const sinks = graph.nodes.filter(n => n.type === "sink");

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
  const edges: Edge[] = graph.edges.map((e, i) => ({
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
 * Convert React Flow state back to backend GraphConfig JSON.
 */
export function flowToGraphConfig(
  nodes: Node<FlowNodeData>[],
  edges: Edge[]
): GraphConfig {
  const graphNodes: GraphNode[] = nodes.map(n => ({
    id: n.id,
    type: n.data.nodeType,
    pipeline_id:
      n.data.nodeType === "pipeline" ? (n.data.pipelineId ?? null) : undefined,
  }));

  const graphEdges: GraphEdge[] = edges.map(e => ({
    from: e.source,
    from_port: e.sourceHandle || "video",
    to_node: e.target,
    to_port: e.targetHandle || "video",
    kind: "stream" as const,
  }));

  return { nodes: graphNodes, edges: graphEdges };
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

/**
 * Build a linear graph from settings panel config (frontend-only).
 * Produces: source -> preprocessor0 -> ... -> pipeline -> postprocessor0 -> ... -> sink.
 */
export function linearGraphFromSettings(
  pipelineId: string,
  preprocessorIds: string[],
  postprocessorIds: string[]
): GraphConfig {
  const allIds = [...preprocessorIds, pipelineId, ...postprocessorIds];
  const nodes: GraphNode[] = [
    { id: "input", type: "source" },
    ...allIds.map(pid => ({
      id: pid,
      type: "pipeline" as const,
      pipeline_id: pid,
    })),
    { id: "output", type: "sink" },
  ];

  const edges: GraphEdge[] = [];
  let prev = "input";
  for (const pid of allIds) {
    edges.push({
      from: prev,
      from_port: "video",
      to_node: pid,
      to_port: "video",
      kind: "stream",
    });
    prev = pid;
  }
  edges.push({
    from: prev,
    from_port: "video",
    to_node: "output",
    to_port: "video",
    kind: "stream",
  });

  return { nodes, edges };
}

/**
 * Try to extract linear pipeline settings from a graph config.
 *
 * Returns null if the graph is non-linear (branching, fan-out, etc.).
 * Returns the pipeline settings if the graph is a simple linear chain:
 * source → preprocessors → pipeline → postprocessors → sink.
 */
export function tryExtractLinearSettings(
  graph: GraphConfig,
  pipelines: Record<string, PipelineSchemaInfo>
): {
  pipelineId: string;
  preprocessorIds: string[];
  postprocessorIds: string[];
} | null {
  // 1. Must have exactly 1 source and 1 sink
  const sources = graph.nodes.filter(n => n.type === "source");
  const sinks = graph.nodes.filter(n => n.type === "sink");
  if (sources.length !== 1 || sinks.length !== 1) return null;

  const sourceId = sources[0].id;
  const sinkId = sinks[0].id;

  // 2. All edges must use "video" ports and be "stream" kind
  for (const edge of graph.edges) {
    if (edge.from_port !== "video" || edge.to_port !== "video") return null;
    if (edge.kind && edge.kind !== "stream") return null;
  }

  // 3. Walk the chain: source → ... → sink
  // Build adjacency: from_node → list of to_nodes
  const outgoing = new Map<string, string[]>();
  for (const edge of graph.edges) {
    const list = outgoing.get(edge.from) ?? [];
    list.push(edge.to_node);
    outgoing.set(edge.from, list);
  }

  // Walk from source to sink
  const chain: string[] = [];
  let current = sourceId;
  const visited = new Set<string>();

  while (current !== sinkId) {
    if (visited.has(current)) return null; // cycle
    visited.add(current);

    const next = outgoing.get(current);
    if (!next || next.length !== 1) return null; // 0 or 2+ outgoing = non-linear
    current = next[0];

    // Collect pipeline nodes (skip source/sink)
    if (current !== sinkId) {
      chain.push(current);
    }
  }

  // 4. Classify pipeline nodes
  const preprocessorIds: string[] = [];
  const postprocessorIds: string[] = [];
  let mainPipelineId: string | null = null;

  for (const nodeId of chain) {
    const node = graph.nodes.find(n => n.id === nodeId);
    if (!node || node.type !== "pipeline" || !node.pipeline_id) return null;

    const pipelineInfo = pipelines[node.pipeline_id];
    const usage = pipelineInfo?.usage ?? [];

    if (usage.includes("preprocessor")) {
      if (mainPipelineId !== null) return null; // preprocessor after main pipeline
      preprocessorIds.push(node.pipeline_id);
    } else if (usage.includes("postprocessor")) {
      if (mainPipelineId === null) return null; // postprocessor before main pipeline
      postprocessorIds.push(node.pipeline_id);
    } else {
      if (mainPipelineId !== null) return null; // 2+ main pipelines
      mainPipelineId = node.pipeline_id;
    }
  }

  // 5. Must have exactly 1 main pipeline
  if (mainPipelineId === null) return null;

  return { pipelineId: mainPipelineId, preprocessorIds, postprocessorIds };
}

// Default node dimensions for reference
export { NODE_WIDTH, NODE_HEIGHT };
