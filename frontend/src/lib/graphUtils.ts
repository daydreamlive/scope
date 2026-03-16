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

/* ── Subgraph types ── */

/** A port exposed on the boundary of a subgraph node. */
export interface SubgraphPort {
  /** Handle name on the subgraph node (e.g. "video_in", "noise_scale") */
  name: string;
  /** Whether this is a stream (video) or parameter connection */
  portType: "stream" | "param";
  /** For param ports: the data type */
  paramType?: "string" | "number" | "boolean" | "list_number";
  /** Which inner node this port maps to */
  innerNodeId: string;
  /** Which handle on that inner node */
  innerHandleId: string;
}

/** Serialized node stored inside a subgraph (same shape as UIStateNode). */
export interface SerializedSubgraphNode {
  id: string;
  type: string;
  position: { x: number; y: number };
  width?: number;
  height?: number;
  data: Record<string, unknown>;
}

/** Serialized edge stored inside a subgraph. */
export interface SerializedSubgraphEdge {
  id: string;
  source: string;
  sourceHandle?: string | null;
  target: string;
  targetHandle?: string | null;
}

export interface FlowNodeData {
  label: string;
  /** User-editable display name; when set, overrides the default header title. */
  customTitle?: string;
  pipelineId?: string | null;
  nodeType:
    | "source"
    | "pipeline"
    | "sink"
    | "primitive"
    | "control"
    | "math"
    | "note"
    | "output"
    | "slider"
    | "knobs"
    | "xypad"
    | "tuple"
    | "reroute"
    | "image"
    | "vace"
    | "midi"
    | "bool"
    | "subgraph"
    | "subgraph_input"
    | "subgraph_output"
    | "timeline"
    | "trigger_action"
    | "curve";
  availablePipelineIds?: string[];
  /** Declared input ports for the selected pipeline */
  streamInputs?: string[];
  /** Declared output ports for the selected pipeline */
  streamOutputs?: string[];
  /** Parameter input ports (for pipeline nodes) */
  parameterInputs?: ParameterPortDef[];
  /** Parameter output ports (for primitive / value-producing nodes) */
  parameterOutputs?: ParameterPortDef[];
  /** Pipeline schemas keyed by pipeline_id, for looking up ports on selection change */
  pipelinePortsMap?: Record<string, { inputs: string[]; outputs: string[] }>;
  /** For primitive nodes: the type of value (string, number, boolean) */
  valueType?: "string" | "number" | "boolean";
  /** For primitive / slider nodes: the current value */
  value?: unknown;
  /** For control nodes: the type of control (float, int, string) */
  controlType?: "float" | "int" | "string";
  /** For string control nodes: animated (pattern cycling) or switch (input-selected) */
  controlMode?: "animated" | "switch";
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
  mathOp?:
    | "add"
    | "subtract"
    | "multiply"
    | "divide"
    | "mod"
    | "min"
    | "max"
    | "power"
    | "abs"
    | "negate"
    | "sqrt"
    | "floor"
    | "ceil"
    | "round"
    | "toInt"
    | "toFloat";
  /** For math nodes: output type conversion (null = auto, "int" = truncate, "float" = ensure float) */
  mathOutputType?: "int" | "float" | null;
  /** For note nodes: the note text content */
  noteText?: string;
  /** For output nodes: sink type (spout, ndi, syphon) */
  outputSinkType?: string;
  /** For output nodes: whether the output is enabled */
  outputSinkEnabled?: boolean;
  /** For output nodes: the sender name */
  outputSinkName?: string;
  /** For source nodes: video source mode (video, camera, spout, ndi, syphon) */
  sourceMode?: "video" | "camera" | "spout" | "ndi" | "syphon";
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
  /** For source nodes: whether Syphon is available (macOS) */
  syphonAvailable?: boolean;
  /** For source nodes: callback when Spout receiver name changes */
  onSpoutSourceChange?: (name: string) => void;
  /** For source nodes: callback when NDI source changes */
  onNdiSourceChange?: (identifier: string) => void;
  /** For source nodes: callback when Syphon source changes */
  onSyphonSourceChange?: (identifier: string) => void;
  /** For sink nodes: remote output stream */
  remoteStream?: MediaStream | null;
  /** For pipeline nodes: whether the selected pipeline supports prompts */
  supportsPrompts?: boolean;
  /** For pipeline nodes: whether the selected pipeline supports cache management (shows Reset Cache button) */
  supportsCacheManagement?: boolean;
  /** For pipeline nodes: current prompt text */
  promptText?: string;
  /** For pipeline nodes: callback when prompt text changes */
  onPromptChange?: (nodeId: string, text: string) => void;
  /** For pipeline nodes: whether the selected pipeline is installed locally */
  pipelineAvailable?: boolean;

  /* ── Slider node fields ── */
  /** For slider nodes: minimum value */
  sliderMin?: number;
  /** For slider nodes: maximum value */
  sliderMax?: number;
  /** For slider nodes: step size */
  sliderStep?: number;

  /* ── Knobs node fields ── */
  /** For knobs nodes: array of knob definitions */
  knobs?: Array<{ label: string; min: number; max: number; value: number }>;

  /* ── XY Pad node fields ── */
  padMinX?: number;
  padMaxX?: number;
  padMinY?: number;
  padMaxY?: number;
  padX?: number;
  padY?: number;

  /* ── Tuple node fields ── */
  /** For tuple nodes: the array of values */
  tupleValues?: number[];
  /** For tuple nodes: minimum for each value */
  tupleMin?: number;
  /** For tuple nodes: maximum for each value */
  tupleMax?: number;
  /** For tuple nodes: step for each value */
  tupleStep?: number;
  /** For tuple nodes: whether to enforce ordering */
  tupleEnforceOrder?: boolean;
  /** For tuple nodes: ordering direction */
  tupleOrderDirection?: "asc" | "desc";

  /* ── Image / Media node fields ── */
  /** For image/media nodes: the selected asset path (image or video) */
  imagePath?: string;
  /** For image/media nodes: detected media type based on file extension */
  mediaType?: "image" | "video";

  /* ── MIDI node fields ── */
  /** For MIDI nodes: array of channel definitions */
  midiChannels?: Array<{
    label: string;
    type: "cc" | "note";
    channel: number;
    cc: number;
    value: number;
  }>;
  /** For MIDI nodes: selected MIDI device ID */
  midiDeviceId?: string;

  /* ── Bool node fields ── */
  /** For bool nodes: conversion mode */
  boolMode?: "gate" | "toggle";
  /** For bool nodes: threshold value (input > threshold → true) */
  boolThreshold?: number;

  /* ── VACE node fields ── */
  /** For VACE nodes: context scale (0.0-2.0) */
  vaceContextScale?: number;
  /** For VACE nodes: reference image path (set via Image node connection) */
  vaceRefImage?: string;
  /** For VACE nodes: first frame image path (set via Image node connection) */
  vaceFirstFrame?: string;
  /** For VACE nodes: last frame image path (set via Image node connection) */
  vaceLastFrame?: string;
  /** For VACE nodes: video file path (set via Media node connection) */
  vaceVideo?: string;

  /* ── Pipeline VACE support ── */
  /** For pipeline nodes: whether the selected pipeline supports VACE */
  supportsVace?: boolean;

  /* ── Subgraph node fields ── */
  /** For subgraph nodes: serialized inner nodes */
  subgraphNodes?: SerializedSubgraphNode[];
  /** For subgraph nodes: serialized inner edges */
  subgraphEdges?: SerializedSubgraphEdge[];
  /** For subgraph nodes: exposed input ports */
  subgraphInputs?: SubgraphPort[];
  /** For subgraph nodes: exposed output ports */
  subgraphOutputs?: SubgraphPort[];
  /** Callback to enter / navigate into a subgraph */
  onEnterSubgraph?: (nodeId: string) => void;
  /** Callback to rename a port on a boundary node (inside a subgraph) */
  onPortRename?: (oldName: string, newName: string, portType: string) => void;
  /** Live port values for boundary / subgraph nodes (transient, not serialized). */
  portValues?: Record<string, unknown>;

  /* ── Timeline node fields ── */
  /** For timeline nodes: total duration in seconds */
  timelineDuration?: number;
  /** For timeline nodes: user-placed trigger markers */
  timelineTriggers?: Array<{ id: string; time: number; label?: string }>;
  /** For timeline nodes: current playhead position (transient, not serialized) */
  timelineCurrentTime?: number;
  /** For timeline nodes: whether to loop at the end */
  timelineLoop?: boolean;
  /** For timeline nodes: current trigger output values (trigger_id → 0 or 1) */
  triggerValues?: Record<string, number>;
  /** For timeline nodes: wall-clock timestamp (Date.now()) when play started (transient) */
  _timelineWallStart?: number;
  /** For timeline nodes: playhead offset when play started (transient) */
  _timelineWallOffset?: number;

  /* ── TriggerAction node fields ── */
  /** For trigger_action nodes: the action type */
  triggerActionType?:
    | "set_string"
    | "set_number"
    | "set_bool"
    | "animate_number"
    | "toggle_bool"
    | "cycle_strings";
  /** For trigger_action nodes: static value for set_* actions */
  triggerSetValue?: unknown;
  /** For trigger_action nodes: animate_number start value */
  triggerAnimateFrom?: number;
  /** For trigger_action nodes: animate_number end value */
  triggerAnimateTo?: number;
  /** For trigger_action nodes: animate_number duration in seconds */
  triggerAnimateDuration?: number;
  /** For trigger_action nodes: easing curve for animate_number */
  triggerAnimateCurve?: "linear" | "ease_in" | "ease_out" | "ease_in_out";
  /** For trigger_action nodes: current toggle state */
  triggerToggleState?: boolean;
  /** For trigger_action nodes: strings list for cycle_strings */
  triggerCycleItems?: string[];
  /** For trigger_action nodes: current index in cycle */
  triggerCycleIndex?: number;

  /* ── Curve node fields ── */
  /** For curve nodes: breakpoints defining the shape, normalized 0-1 on both axes, sorted by x */
  curvePoints?: Array<{ x: number; y: number }>;
  /** For curve nodes: output range minimum (default 0) */
  curveMin?: number;
  /** For curve nodes: output range maximum (default 1) */
  curveMax?: number;

  /* ── Node lock / pin / collapse ── */
  /** When true, parameter inputs on this node are disabled (read-only). */
  locked?: boolean;
  /** When true, the node cannot be dragged on the canvas. */
  pinned?: boolean;
  /** When true, the node is visually collapsed to a compact pill. */
  collapsed?: boolean;

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
export function buildHandleId(kind: "stream" | "param", name: string): string {
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

    // Skip complex component fields that get special handling in the node
    // (e.g. manage_cache has component "cache" and is replaced by a Reset Cache button,
    //  vace_context_scale has component "vace" and is handled by the VACE node)
    if (
      schemaProp.ui.component === "cache" ||
      schemaProp.ui.component === "vace"
    )
      continue;

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
      params.push({
        name: key,
        type: "list_number",
        defaultValue: schemaProp.default,
        label,
      });
      continue;
    }

    // Check anyOf for array types (e.g. list[int] | None)
    const anyOf = (schemaProp as Record<string, unknown>).anyOf as
      | Record<string, unknown>[]
      | undefined;
    if (anyOf?.length) {
      const arrayVariant = anyOf.find(v => isArrayOfNumbers(v));
      if (arrayVariant) {
        const ui = schemaProp.ui;
        const label = ui?.label || key;
        params.push({
          name: key,
          type: "list_number",
          defaultValue: schemaProp.default,
          label,
        });
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
      min:
        typeof schemaProp.minimum === "number" ? schemaProp.minimum : undefined,
      max:
        typeof schemaProp.maximum === "number" ? schemaProp.maximum : undefined,
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
  // Collect subgraph node IDs from ui_state so we can skip their flattened
  // inner nodes (which were expanded into the backend nodes/edges on export).
  const subgraphNodeIds = new Set<string>();
  if (graph.ui_state) {
    const uiNodes = (graph.ui_state.nodes ?? []) as UIStateNode[];
    for (const un of uiNodes) {
      if (un.type === "subgraph") {
        subgraphNodeIds.add(un.id);
      }
    }
  }

  /** True if `nodeId` is a flattened inner node owned by a known subgraph. */
  const isSubgraphInnerNode = (nodeId: string): boolean => {
    for (const sgId of subgraphNodeIds) {
      if (nodeId.startsWith(sgId + ":")) return true;
    }
    return false;
  };

  const sources = graph.nodes.filter(n => n.type === "source" && !isSubgraphInnerNode(n.id));
  const pipelines = graph.nodes.filter(n => n.type === "pipeline" && !isSubgraphInnerNode(n.id));
  const sinks = graph.nodes.filter(n => n.type === "sink" && !isSubgraphInnerNode(n.id));

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
        y:
          savedY !== undefined ? savedY : START_Y + i * (NODE_HEIGHT + ROW_GAP),
      },
      width: w,
      height: h,
      style: { width: w, height: h },
      data: {
        label: n.id,
        nodeType: "source",
        sourceMode:
          (n.source_mode as
            | "video"
            | "camera"
            | "spout"
            | "ndi"
            | "syphon"
            | undefined) ?? "video",
        sourceName: n.source_name ?? undefined,
      },
    });
  });

  // Layout pipelines (column 1) - use saved position if available, otherwise auto-layout
  pipelines.forEach((n, i) => {
    const ports = n.pipeline_id && portsMap ? portsMap[n.pipeline_id] : null;
    const savedX = n.x ?? undefined;
    const savedY = n.y ?? undefined;
    const sizeProps =
      n.w != null || n.h != null
        ? {
            width: n.w ?? undefined,
            height: n.h ?? undefined,
            style: { width: n.w ?? undefined, height: n.h ?? undefined },
          }
        : {};
    nodes.push({
      id: n.id,
      type: "pipeline",
      position: {
        x: savedX !== undefined ? savedX : START_X + COLUMN_GAP,
        y:
          savedY !== undefined ? savedY : START_Y + i * (NODE_HEIGHT + ROW_GAP),
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
        y:
          savedY !== undefined ? savedY : START_Y + i * (NODE_HEIGHT + ROW_GAP),
      },
      width: w,
      height: h,
      style: { width: w, height: h },
      data: { label: n.id, nodeType: "sink" },
    });
  });

  // Convert edges - add stream: prefix to handle IDs
  // Skip edges that reference flattened inner subgraph nodes
  const edges: Edge[] = graph.edges
    .filter(e => !isSubgraphInnerNode(e.from) && !isSubgraphInnerNode(e.to_node))
    .map((e, i) => {
    const sourceHandle =
      e.kind === "parameter"
        ? buildHandleId("param", e.from_port)
        : buildHandleId("stream", e.from_port);
    const targetHandle =
      e.kind === "parameter"
        ? buildHandleId("param", e.to_port)
        : buildHandleId("stream", e.to_port);
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

  // Restore frontend-only nodes and edges from ui_state
  if (graph.ui_state) {
    const uiNodes = (graph.ui_state.nodes ?? []) as UIStateNode[];
    const uiEdges = (graph.ui_state.edges ?? []) as UIStateEdge[];

    for (const un of uiNodes) {
      // Migrate old "value" nodes to "primitive"
      const nodeType = un.type === "value" ? "primitive" : un.type;
      const nodeData = un.data as FlowNodeData;
      if (un.type === "value") {
        nodeData.nodeType = "primitive";
      }
      const sizeProps =
        un.width != null || un.height != null
          ? {
              width: un.width ?? undefined,
              height: un.height ?? undefined,
              style: {
                width: un.width ?? undefined,
                height: un.height ?? undefined,
              },
            }
          : {};
      nodes.push({
        id: un.id,
        type: nodeType,
        position: { x: un.position.x, y: un.position.y },
        ...sizeProps,
        data: nodeData,
      });
    }

    for (const ue of uiEdges) {
      edges.push({
        id: ue.id,
        source: ue.source,
        sourceHandle: ue.sourceHandle ?? undefined,
        target: ue.target,
        targetHandle: ue.targetHandle ?? undefined,
      });
    }

    // Restore lock/pin/collapsed flags for all nodes (including backend nodes)
    const savedFlags = graph.ui_state.node_flags as
      | Record<
          string,
          { locked?: boolean; pinned?: boolean; collapsed?: boolean }
        >
      | undefined;
    if (savedFlags) {
      for (const node of nodes) {
        const flags = savedFlags[node.id];
        if (flags) {
          if (flags.locked) node.data.locked = true;
          if (flags.pinned) {
            node.data.pinned = true;
            node.draggable = false;
          }
          if (flags.collapsed) node.data.collapsed = true;
        }
      }
    }
  }

  return { nodes, edges };
}

/** Node types that are frontend-only and not sent to the backend graph. */
const FRONTEND_ONLY_TYPES = new Set<FlowNodeData["nodeType"]>([
  "primitive",
  "control",
  "math",
  "note",
  "output",
  "slider",
  "knobs",
  "xypad",
  "tuple",
  "reroute",
  "image",
  "vace",
  "midi",
  "bool",
  "subgraph",
  "subgraph_input",
  "subgraph_output",
  "timeline",
  "trigger_action",
  "curve",
]);

/** Fields in FlowNodeData that are non-serializable (functions, streams, etc.) */
const NON_SERIALIZABLE_KEYS = new Set<string>([
  "localStream",
  "remoteStream",
  "onVideoFileUpload",
  "onSourceModeChange",
  "onSpoutSourceChange",
  "onNdiSourceChange",
  "onSyphonSourceChange",
  "onPromptChange",
  "pipelinePortsMap",
  "_savedWidth",
  "_savedHeight",
  "onEnterSubgraph",
  "onPortRename",
  "portValues",
  "timelineCurrentTime",
  "triggerValues",
  "_timelineWallStart",
  "_timelineWallOffset",
]);

/**
 * Pick only serializable data fields from FlowNodeData.
 */
function serializableData(data: FlowNodeData): Record<string, unknown> {
  const result: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(data)) {
    if (NON_SERIALIZABLE_KEYS.has(key)) continue;
    if (typeof value === "function") continue;
    result[key] = value;
  }
  return result;
}

/** Shape of a serialized UI node stored in ui_state. */
interface UIStateNode {
  id: string;
  type: string;
  position: { x: number; y: number };
  width?: number;
  height?: number;
  data: Record<string, unknown>;
}

/** Shape of a serialized UI edge stored in ui_state. */
interface UIStateEdge {
  id: string;
  source: string;
  sourceHandle?: string | null;
  target: string;
  targetHandle?: string | null;
}

/**
 * Recursively flatten subgraph nodes so the backend sees only
 * source / pipeline / sink nodes.  Inner backend nodes are hoisted
 * to the top-level with a prefixed ID (`subgraphId:innerNodeId`).
 * Edges that cross the subgraph boundary are remapped through the
 * SubgraphPort mappings.
 *
 * Returns new top-level arrays (nodes & edges) with all subgraphs dissolved.
 */
export function flattenSubgraphs(
  nodes: Node<FlowNodeData>[],
  edges: Edge[],
  prefix = ""
): { nodes: Node<FlowNodeData>[]; edges: Edge[] } {
  const flatNodes: Node<FlowNodeData>[] = [];
  const flatEdges: Edge[] = [...edges];

  // Maps from "subgraphNodeId + portName" → actual inner nodeId + handleId
  // Used to remap edges that connect to a subgraph's external ports.
  const inputPortMap = new Map<string, { nodeId: string; handleId: string }>();
  const outputPortMap = new Map<string, { nodeId: string; handleId: string }>();

  for (const node of nodes) {
    if (node.data.nodeType !== "subgraph") {
      flatNodes.push(node);
      continue;
    }

    // This is a subgraph node – extract and flatten its contents
    const sg = node;
    const innerNodesRaw = sg.data.subgraphNodes ?? [];
    const innerEdgesRaw = sg.data.subgraphEdges ?? [];
    const sgInputs = sg.data.subgraphInputs ?? [];
    const sgOutputs = sg.data.subgraphOutputs ?? [];
    const sgPrefix = prefix ? `${prefix}${sg.id}:` : `${sg.id}:`;

    // Build port maps for this subgraph
    for (const port of sgInputs) {
      inputPortMap.set(`${sg.id}::${port.name}`, {
        nodeId: sgPrefix + port.innerNodeId,
        handleId: port.innerHandleId,
      });
    }
    for (const port of sgOutputs) {
      outputPortMap.set(`${sg.id}::${port.name}`, {
        nodeId: sgPrefix + port.innerNodeId,
        handleId: port.innerHandleId,
      });
    }

    // Reconstruct inner nodes as React Flow nodes for recursive flattening
    const innerFlowNodes: Node<FlowNodeData>[] = innerNodesRaw.map(n => ({
      id: sgPrefix + n.id,
      type: n.type,
      position: n.position,
      width: n.width,
      height: n.height,
      data: { ...n.data, label: n.data.label ?? n.id } as FlowNodeData,
    }));

    // Remap inner edge node references
    const innerFlowEdges: Edge[] = innerEdgesRaw.map(e => ({
      ...e,
      id: `${sgPrefix}${e.id}`,
      source: sgPrefix + e.source,
      target: sgPrefix + e.target,
    }));

    // Recursively flatten (handles nested subgraphs)
    const { nodes: hoisted, edges: hoistedEdges } = flattenSubgraphs(
      innerFlowNodes,
      innerFlowEdges,
      "" // prefix already applied
    );

    flatNodes.push(...hoisted);
    flatEdges.push(...hoistedEdges);
  }

  // Now remap edges that reference subgraph ports
  const remappedEdges = flatEdges.map(e => {
    let { source, sourceHandle, target, targetHandle } = e;

    // Check if the source is a subgraph output port
    const sourceParsed = parseHandleId(sourceHandle);
    if (sourceParsed) {
      const key = `${source}::${sourceParsed.name}`;
      const mapping = outputPortMap.get(key);
      if (mapping) {
        source = mapping.nodeId;
        sourceHandle = mapping.handleId;
      }
    }

    // Check if the target is a subgraph input port
    const targetParsed = parseHandleId(targetHandle);
    if (targetParsed) {
      const key = `${target}::${targetParsed.name}`;
      const mapping = inputPortMap.get(key);
      if (mapping) {
        target = mapping.nodeId;
        targetHandle = mapping.handleId;
      }
    }

    if (
      source !== e.source ||
      sourceHandle !== e.sourceHandle ||
      target !== e.target ||
      targetHandle !== e.targetHandle
    ) {
      return { ...e, source, sourceHandle, target, targetHandle };
    }
    return e;
  });

  // Filter out edges that still reference subgraph node IDs (shouldn't happen, but safety)
  const subgraphIds = new Set(
    nodes.filter(n => n.data.nodeType === "subgraph").map(n => n.id)
  );
  const cleanEdges = remappedEdges.filter(
    e => !subgraphIds.has(e.source) && !subgraphIds.has(e.target)
  );

  return { nodes: flatNodes, edges: cleanEdges };
}

/**
 * Convert React Flow state back to backend GraphConfig JSON.
 */
export function flowToGraphConfig(
  nodes: Node<FlowNodeData>[],
  edges: Edge[]
): GraphConfig {
  // Flatten any subgraph nodes so the backend only sees source/pipeline/sink
  const hasSubgraphs = nodes.some(n => n.data.nodeType === "subgraph");
  const { nodes: flatNodes, edges: flatEdges } = hasSubgraphs
    ? flattenSubgraphs(nodes, edges)
    : { nodes, edges };

  // Separate backend nodes from frontend-only nodes
  const frontendNodeIds = new Set<string>();

  // Collect frontend-only IDs from the flattened set
  const backendFlatNodes = flatNodes.filter(n => {
    if (FRONTEND_ONLY_TYPES.has(n.data.nodeType)) {
      frontendNodeIds.add(n.id);
      return false;
    }
    return true;
  });

  // Also include frontend-only nodes from the original (un-flattened) set.
  // Subgraph nodes are removed by flattenSubgraphs (replaced by their inner
  // nodes) but still need to be serialized into ui_state so they persist.
  for (const n of nodes) {
    if (FRONTEND_ONLY_TYPES.has(n.data.nodeType)) {
      frontendNodeIds.add(n.id);
    }
  }

  const graphNodes: GraphNode[] = backendFlatNodes
    .map(n => {
      // Read dimensions: node.width/height (set by NodeResizer) > measured > style
      const w =
        n.width ??
        n.measured?.width ??
        (typeof n.style?.width === "number" ? n.style.width : undefined);
      const h =
        n.height ??
        n.measured?.height ??
        (typeof n.style?.height === "number" ? n.style.height : undefined);
      return {
        id: n.id,
        type:
          n.data.nodeType === "source"
            ? "source"
            : n.data.nodeType === "sink"
              ? "sink"
              : "pipeline",
        pipeline_id:
          n.data.nodeType === "pipeline"
            ? (n.data.pipelineId ?? null)
            : undefined,
        x: n.position.x,
        y: n.position.y,
        w: w && !Number.isNaN(w) ? w : undefined,
        h: h && !Number.isNaN(h) ? h : undefined,
        source_mode:
          n.data.nodeType === "source"
            ? (n.data.sourceMode ?? null)
            : undefined,
        source_name:
          n.data.nodeType === "source"
            ? (n.data.sourceName ?? null)
            : undefined,
      };
    });

  // Filter edges to only include those where both source and target exist in graphNodes
  const graphNodeIds = new Set(graphNodes.map(n => n.id));
  const graphEdges: GraphEdge[] = flatEdges
    .filter(e => graphNodeIds.has(e.source) && graphNodeIds.has(e.target))
    .map(e => {
      const sourceParsed = parseHandleId(e.sourceHandle);
      const targetParsed = parseHandleId(e.targetHandle);
      const kind =
        sourceParsed?.kind === "param" && targetParsed?.kind === "param"
          ? "parameter"
          : "stream";
      return {
        from: e.source,
        from_port: sourceParsed?.name || "video",
        to_node: e.target,
        to_port: targetParsed?.name || "video",
        kind: kind as "stream" | "parameter",
      };
    });

  // Serialize frontend-only nodes and their edges into ui_state
  let ui_state: Record<string, unknown> | undefined;

  // Collect lock/pin/collapsed flags for ALL nodes (including backend nodes like source/pipeline/sink)
  const nodeFlags: Record<
    string,
    { locked?: boolean; pinned?: boolean; collapsed?: boolean }
  > = {};
  for (const n of nodes) {
    if (n.data.locked || n.data.pinned || n.data.collapsed) {
      nodeFlags[n.id] = {
        ...(n.data.locked ? { locked: true } : {}),
        ...(n.data.pinned ? { pinned: true } : {}),
        ...(n.data.collapsed ? { collapsed: true } : {}),
      };
    }
  }

  if (frontendNodeIds.size > 0 || Object.keys(nodeFlags).length > 0) {
    const uiNodes: UIStateNode[] = nodes
      .filter(n => frontendNodeIds.has(n.id))
      .map(n => {
        const w =
          n.width ??
          n.measured?.width ??
          (typeof n.style?.width === "number" ? n.style.width : undefined);
        const h =
          n.height ??
          n.measured?.height ??
          (typeof n.style?.height === "number" ? n.style.height : undefined);
        return {
          id: n.id,
          type: n.data.nodeType,
          position: { x: n.position.x, y: n.position.y },
          ...(w && !Number.isNaN(w) ? { width: w } : {}),
          ...(h && !Number.isNaN(h) ? { height: h } : {}),
          data: serializableData(n.data),
        };
      });

    // Edges that touch at least one frontend-only node (and aren't already in graphEdges)
    const uiEdges: UIStateEdge[] = edges
      .filter(
        e => frontendNodeIds.has(e.source) || frontendNodeIds.has(e.target)
      )
      .map(e => ({
        id: e.id,
        source: e.source,
        sourceHandle: e.sourceHandle ?? null,
        target: e.target,
        targetHandle: e.targetHandle ?? null,
      }));

    ui_state = {
      nodes: uiNodes,
      edges: uiEdges,
      ...(Object.keys(nodeFlags).length > 0 ? { node_flags: nodeFlags } : {}),
    };
  }

  return {
    nodes: graphNodes,
    edges: graphEdges,
    ...(ui_state ? { ui_state } : {}),
  };
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
  const allPipelineIds = [...preprocessorIds, pipelineId, ...postprocessorIds];
  // Generate unique node IDs so duplicate pipeline_ids get distinct nodes
  const usedIds = new Set<string>(["input", "output"]);
  const nodeEntries = allPipelineIds.map(pid => {
    const nodeId = generateNodeId(pid, usedIds);
    usedIds.add(nodeId);
    return { nodeId, pid };
  });

  const nodes: GraphNode[] = [
    { id: "input", type: "source", source_mode: "video" },
    ...nodeEntries.map(({ nodeId, pid }) => ({
      id: nodeId,
      type: "pipeline" as const,
      pipeline_id: pid,
    })),
    { id: "output", type: "sink" },
  ];

  const edges: GraphEdge[] = [];
  let prev = "input";
  for (const { nodeId } of nodeEntries) {
    edges.push({
      from: prev,
      from_port: "video",
      to_node: nodeId,
      to_port: "video",
      kind: "stream",
    });
    prev = nodeId;
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

/**
 * Strip frontend-only fields (position, size, ui_state) from a GraphConfig
 * before sending to the backend. The backend only needs node identity/type,
 * edges, and source config — not layout or UI state.
 */
export function stripUIFields(graph: GraphConfig): GraphConfig {
  return {
    nodes: graph.nodes.map(({ x, y, w, h, ...rest }) => rest),
    edges: graph.edges,
  };
}

// Default node dimensions for reference
export { NODE_WIDTH, NODE_HEIGHT };
