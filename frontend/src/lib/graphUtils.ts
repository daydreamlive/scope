import type { Node, Edge } from "@xyflow/react";
import type {
  GraphConfig,
  GraphNode,
  GraphEdge,
  PipelineSchemaInfo,
} from "./api";
import { inferPrimitiveFieldType } from "./schemaSettings";
import type { SchemaProperty } from "./schemaSettings";

// Layout constants
const NODE_WIDTH = 200;
const NODE_HEIGHT = 60;
const COLUMN_GAP = 300;
const ROW_GAP = 100;
const START_X = 50;
const START_Y = 50;

export type PortType = "stream" | "string" | "number" | "boolean";

export interface ParameterPortDef {
  name: string;
  type: "string" | "number" | "boolean" | "list_number";
  defaultValue?: unknown;
  label?: string;
  min?: number;
  max?: number;
  enum?: unknown[];
}

export interface PortInfo {
  name: string;
}

export interface FlowNodeData {
  label: string;
  pipelineId?: string | null;
  nodeType: "source" | "pipeline" | "sink" | "value" | "control" | "math" | "note";
  availablePipelineIds?: string[];
  /** Declared input ports for the selected pipeline */
  streamInputs?: string[];
  /** Declared output ports for the selected pipeline */
  streamOutputs?: string[];
  /** Parameter input ports (for pipeline nodes) */
  parameterInputs?: ParameterPortDef[];
  /** Parameter output ports (for value nodes) */
  parameterOutputs?: ParameterPortDef[];
  /** Pipeline schemas keyed by pipeline_id, for looking up ports on selection change */
  pipelinePortsMap?: Record<string, { inputs: string[]; outputs: string[] }>;
  /** For value nodes: the type of value (string, number, boolean) */
  valueType?: "string" | "number" | "boolean";
  /** For value nodes: the current value */
  value?: unknown;
  /** For control nodes: the type of control (float, int, string) */
  controlType?: "float" | "int" | "string";
  /** For control nodes: the animation pattern */
  controlPattern?: "sine" | "bounce" | "random_walk" | "linear" | "step";
  /** For control nodes: cycles per second */
  controlSpeed?: number;
  /** For control nodes: minimum value (for float/int) */
  controlMin?: number;
  /** For control nodes: maximum value (for float/int) */
  controlMax?: number;
  /** For control nodes: list of strings to cycle through (for string variant) */
  controlItems?: string[];
  /** For control nodes: whether animation is playing */
  isPlaying?: boolean;
  /** For control nodes: current animated value (updated by animation loop) */
  currentValue?: number | string;
  /** For math nodes: the operation to perform */
  mathOp?: "add" | "subtract" | "multiply" | "divide" | "mod" | "min" | "max" | "power" | "abs" | "negate" | "sqrt" | "floor" | "ceil" | "round";
  /** For note nodes: the note text content */
  noteText?: string;
  /** For source nodes: video source mode (video, camera, spout, ndi) */
  sourceMode?: "video" | "camera" | "spout" | "ndi";
  /** For source nodes: source name/identifier for Spout/NDI (sender name for Spout, identifier for NDI) */
  sourceName?: string;
  /** For source nodes: local video preview stream (camera or file) */
  localStream?: MediaStream | null;
  /** For source nodes: callback to upload a video file */
  onVideoFileUpload?: (file: File) => Promise<boolean>;
  /** For source nodes: callback when source mode changes */
  onSourceModeChange?: (mode: string) => void;
  /** For source nodes: whether Spout is available */
  spoutAvailable?: boolean;
  /** For source nodes: whether NDI is available */
  ndiAvailable?: boolean;
  /** For source nodes: callback when Spout receiver name changes */
  onSpoutSourceChange?: (name: string) => void;
  /** For source nodes: callback when NDI source changes */
  onNdiSourceChange?: (identifier: string) => void;
  /** For sink nodes: remote output stream */
  remoteStream?: MediaStream | null;
  /** For pipeline nodes: whether the selected pipeline supports prompts */
  supportsPrompts?: boolean;
  /** For pipeline nodes: current prompt text */
  promptText?: string;
  /** For pipeline nodes: callback when prompt text changes */
  onPromptChange?: (nodeId: string, text: string) => void;
  [key: string]: unknown;
}

/**
 * Parse a handle ID to extract its kind and name.
 * Handles both prefixed (stream:video, param:noise_scale) and legacy (video) formats.
 */
export function parseHandleId(handleId: string | null | undefined): {
  kind: "stream" | "param";
  name: string;
} | null {
  if (!handleId) return null;
  if (handleId.startsWith("stream:")) {
    return { kind: "stream", name: handleId.slice(7) };
  }
  if (handleId.startsWith("param:")) {
    return { kind: "param", name: handleId.slice(6) };
  }
  // Legacy format: assume stream for backward compatibility
  return { kind: "stream", name: handleId };
}

/**
 * Build a handle ID from kind and name.
 */
export function buildHandleId(
  kind: "stream" | "param",
  name: string
): string {
  return `${kind}:${name}`;
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
 * Extract parameter ports from a pipeline schema's config_schema.
 * Returns primitive types (string, number, boolean) and list types (list_number) that can be connected.
 */
export function extractParameterPorts(
  schema: PipelineSchemaInfo | null
): ParameterPortDef[] {
  if (!schema?.config_schema?.properties) return [];

  const params: ParameterPortDef[] = [];
  const properties = schema.config_schema.properties;

  for (const [key, prop] of Object.entries(properties)) {
    const schemaProp = prop as SchemaProperty;
    // Only include fields that have ui metadata (json_schema_extra), matching sidebar behavior
    if (!schemaProp.ui) continue;

    // Check for array types with integer/number items (e.g. denoising_steps: list[int])
    // Handles both direct { type: "array", items: ... } and anyOf: [{ type: "array" }, { type: "null" }]
    const isArrayOfNumbers = (obj: Record<string, unknown>): boolean => {
      if (obj.type === "array" && obj.items) {
        const items = obj.items as { type?: string };
        return items.type === "integer" || items.type === "number";
      }
      return false;
    };

    if (isArrayOfNumbers(schemaProp as unknown as Record<string, unknown>)) {
      const ui = schemaProp.ui;
      const label = ui?.label || key;
      params.push({ name: key, type: "list_number", defaultValue: schemaProp.default, label });
      continue;
    }

    // Check anyOf for array types (e.g. list[int] | None)
    const anyOf = (schemaProp as Record<string, unknown>).anyOf as Record<string, unknown>[] | undefined;
    if (anyOf?.length) {
      const arrayVariant = anyOf.find(v => isArrayOfNumbers(v));
      if (arrayVariant) {
        const ui = schemaProp.ui;
        const label = ui?.label || key;
        params.push({ name: key, type: "list_number", defaultValue: schemaProp.default, label });
        continue;
      }
    }

    const fieldType = inferPrimitiveFieldType(schemaProp);
    if (!fieldType) continue;

    // Map PrimitiveFieldType to PortType
    let paramType: "string" | "number" | "boolean" | null = null;
    if (fieldType === "text" || fieldType === "enum") {
      paramType = "string";
    } else if (fieldType === "number" || fieldType === "slider") {
      paramType = "number";
    } else if (fieldType === "toggle") {
      paramType = "boolean";
    }

    if (!paramType) continue;

    const ui = schemaProp.ui;
    const label = ui?.label || key;

    params.push({
      name: key,
      type: paramType,
      defaultValue: schemaProp.default,
      label,
      min: typeof schemaProp.minimum === "number" ? schemaProp.minimum : undefined,
      max: typeof schemaProp.maximum === "number" ? schemaProp.maximum : undefined,
      enum: Array.isArray(schemaProp.enum) ? schemaProp.enum : undefined,
    });
  }

  return params;
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

  // Layout sources (column 0) - use saved position if available, otherwise auto-layout
  sources.forEach((n, i) => {
    const savedX = n.x ?? undefined;
    const savedY = n.y ?? undefined;
    const w = n.w ?? 240;
    const h = n.h ?? 200;
    nodes.push({
      id: n.id,
      type: "source",
      position: {
        x: savedX !== undefined ? savedX : START_X,
        y: savedY !== undefined ? savedY : START_Y + i * (NODE_HEIGHT + ROW_GAP),
      },
      width: w,
      height: h,
      style: { width: w, height: h },
      data: {
        label: n.id,
        nodeType: "source",
        sourceMode: n.source_mode as "video" | "camera" | "spout" | "ndi" | undefined,
        sourceName: n.source_name ?? undefined,
      },
    });
  });

  // Layout pipelines (column 1) - use saved position if available, otherwise auto-layout
  pipelines.forEach((n, i) => {
    const ports = n.pipeline_id && portsMap ? portsMap[n.pipeline_id] : null;
    const savedX = n.x ?? undefined;
    const savedY = n.y ?? undefined;
    const sizeProps = n.w != null || n.h != null
      ? { width: n.w ?? undefined, height: n.h ?? undefined, style: { width: n.w ?? undefined, height: n.h ?? undefined } }
      : {};
    nodes.push({
      id: n.id,
      type: "pipeline",
      position: {
        x: savedX !== undefined ? savedX : START_X + COLUMN_GAP,
        y: savedY !== undefined ? savedY : START_Y + i * (NODE_HEIGHT + ROW_GAP),
      },
      ...sizeProps,
      data: {
        label: n.pipeline_id || n.id,
        pipelineId: n.pipeline_id,
        nodeType: "pipeline",
        streamInputs: ports?.inputs ?? ["video"],
        streamOutputs: ports?.outputs ?? ["video"],
      },
    });
  });

  // Layout sinks (column 2) - use saved position if available, otherwise auto-layout
  sinks.forEach((n, i) => {
    const savedX = n.x ?? undefined;
    const savedY = n.y ?? undefined;
    const w = n.w ?? 240;
    const h = n.h ?? 200;
    nodes.push({
      id: n.id,
      type: "sink",
      position: {
        x: savedX !== undefined ? savedX : START_X + COLUMN_GAP * 2,
        y: savedY !== undefined ? savedY : START_Y + i * (NODE_HEIGHT + ROW_GAP),
      },
      width: w,
      height: h,
      style: { width: w, height: h },
      data: { label: n.id, nodeType: "sink" },
    });
  });

  // Convert edges - add stream: prefix to handle IDs
  const edges: Edge[] = graph.edges.map((e, i) => {
    const sourceHandle = e.kind === "parameter" ? buildHandleId("param", e.from_port) : buildHandleId("stream", e.from_port);
    const targetHandle = e.kind === "parameter" ? buildHandleId("param", e.to_port) : buildHandleId("stream", e.to_port);
    return {
      id: `e-${i}-${e.from}-${e.to_node}`,
      source: e.from,
      sourceHandle,
      target: e.to_node,
      targetHandle,
      label: e.from_port !== "video" ? e.from_port : undefined,
      animated: false,
    };
  });

  return { nodes, edges };
}

/**
 * Convert React Flow state back to backend GraphConfig JSON.
 */
export function flowToGraphConfig(
  nodes: Node<FlowNodeData>[],
  edges: Edge[]
): GraphConfig {
  const graphNodes: GraphNode[] = nodes
    .filter(n => n.data.nodeType !== "value" && n.data.nodeType !== "control" && n.data.nodeType !== "math" && n.data.nodeType !== "note") // Filter out frontend-only nodes
    .map(n => {
      // Read dimensions: node.width/height (set by NodeResizer) > measured > style
      const w = n.width ?? n.measured?.width ?? (typeof n.style?.width === "number" ? n.style.width : undefined);
      const h = n.height ?? n.measured?.height ?? (typeof n.style?.height === "number" ? n.style.height : undefined);
      return {
        id: n.id,
        type: n.data.nodeType === "source" ? "source" : n.data.nodeType === "sink" ? "sink" : "pipeline",
        pipeline_id:
          n.data.nodeType === "pipeline" ? (n.data.pipelineId ?? null) : undefined,
        x: n.position.x,
        y: n.position.y,
        w: w && !Number.isNaN(w) ? w : undefined,
        h: h && !Number.isNaN(h) ? h : undefined,
        source_mode: n.data.nodeType === "source" ? (n.data.sourceMode ?? null) : undefined,
        source_name: n.data.nodeType === "source" ? (n.data.sourceName ?? null) : undefined,
      };
    });

  // Filter edges to only include those where both source and target exist in graphNodes
  const graphNodeIds = new Set(graphNodes.map(n => n.id));
  const graphEdges: GraphEdge[] = edges
    .filter(e => graphNodeIds.has(e.source) && graphNodeIds.has(e.target))
    .map(e => {
      const sourceParsed = parseHandleId(e.sourceHandle);
      const targetParsed = parseHandleId(e.targetHandle);
      const kind = sourceParsed?.kind === "param" && targetParsed?.kind === "param" ? "parameter" : "stream";
      return {
        from: e.source,
        from_port: sourceParsed?.name || "video",
        to_node: e.target,
        to_port: targetParsed?.name || "video",
        kind: kind as "stream" | "parameter",
      };
    });

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
