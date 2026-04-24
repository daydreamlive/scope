import type { Node, Edge } from "@xyflow/react";
import type {
  GraphConfig,
  GraphNode,
  GraphEdge,
  PipelineSchemaInfo,
  LoRAFileInfo,
  NodePortDef,
  NodeParamDef,
} from "./api";
import { inferPrimitiveFieldType } from "./schemaSettings";
import { resolveLoRAPath } from "./workflowSettings";
import type { SchemaProperty } from "./schemaSettings";

// Layout constants
const NODE_WIDTH = 200;
const NODE_HEIGHT = 60;
const COLUMN_GAP = 300;
const ROW_GAP = 100;
const START_X = 50;
const START_Y = 50;

export type PortType = "stream" | "string" | "number" | "boolean";

export interface BrowserSourceMode {
  id: string;
  label: string;
  description: string;
  keywords: string[];
}

/** Source modes provided by the browser (WebRTC). Everything else is a
 *  server-side :class:`InputSource` (built-in spout/ndi/syphon/video_file
 *  or any plugin-registered id). The label/description/keywords are used by
 *  the canvas context menu and Add Node modal. */
export const BROWSER_SOURCE_MODES: readonly BrowserSourceMode[] = [
  {
    id: "video",
    label: "File",
    description: "Upload or cycle sample video files",
    keywords: ["file", "video", "upload"],
  },
  {
    id: "camera",
    label: "Camera",
    description: "Use the browser's webcam",
    keywords: ["webcam", "camera", "capture"],
  },
];

const BROWSER_SOURCE_MODE_SET: ReadonlySet<string> = new Set(
  BROWSER_SOURCE_MODES.map(m => m.id)
);

/** True when *mode* is browser-driven (file upload or webcam) and gets a
 *  WebRTC track; false for any server-side input source. */
export function isBrowserSourceMode(mode: string | null | undefined): boolean {
  return mode != null && BROWSER_SOURCE_MODE_SET.has(mode);
}

export interface ParameterPortDef {
  name: string;
  type: "string" | "number" | "boolean" | "list_number" | "trigger";
  defaultValue?: unknown;
  label?: string;
  min?: number;
  max?: number;
  step?: number;
  enum?: unknown[];
  isLoadParam?: boolean;
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
  paramType?: "string" | "number" | "boolean" | "list_number" | "trigger";
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

/**
 * Per-parameter OSC opt-in / address override / default override.
 *
 * Stored on `node.data.oscConfig` keyed by the node-data field name
 * (e.g. "value" for a slider, "sourceMode" for a source). Computed by
 * `useOscInventory` into the global OSC inventory POSTed to the
 * backend. Persists with the rest of the graph.
 */
export interface OscParamConfig {
  exposed: boolean;
  /** Full OSC address (e.g. "/scope/tempo/value"); auto-derived when omitted. */
  address?: string;
  /** Advisory default published in the OSC docs; not auto-applied to the param. */
  default?: unknown;
}

export interface FlowNodeData {
  label: string;
  /** User-editable display name; when set, overrides the default header title. */
  customTitle?: string;
  /** Per-param OSC opt-in/override map. Absent ⇒ no params exposed for this node. */
  oscConfig?: Record<string, OscParamConfig>;
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
    | "lora"
    | "midi"
    | "bool"
    | "trigger"
    | "subgraph"
    | "subgraph_input"
    | "subgraph_output"
    | "record"
    | "tempo"
    | "prompt_list"
    | "prompt_blend"
    | "scheduler"
    | "audio"
    | "custom_node";
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
  valueType?: "string" | "number" | "boolean" | "trigger";
  /** For primitive / slider nodes: the current value */
  value?: unknown;
  /** For primitive nodes: whether value changes are sent immediately (default: true) */
  primitiveAutoSend?: boolean;
  /** For primitive nodes: the last explicitly committed/sent value (runtime-only, used when autoSend is off) */
  committedValue?: unknown;
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
  /** For math nodes: manual default for input A when not connected */
  mathDefaultA?: number;
  /** For math nodes: manual default for input B when not connected */
  mathDefaultB?: number;
  /** For note nodes: the note text content */
  noteText?: string;
  /** For output nodes: sink type (spout, ndi, syphon) */
  outputSinkType?: string;
  /** For output nodes: whether the output is enabled */
  outputSinkEnabled?: boolean;
  /** For output nodes: the sender name */
  outputSinkName?: string;
  /** For source nodes: video source mode. "video" and "camera" are browser-
   *  provided (WebRTC); any other id is a server-side :class:`InputSource`
   *  (built-in spout/ndi/syphon/video_file or any plugin-registered source). */
  sourceMode?: string;
  /** For source nodes: source name/identifier for Spout/NDI (sender name for Spout, identifier for NDI) */
  sourceName?: string;
  /** For source nodes: whether incoming Syphon frames should be flipped vertically */
  sourceFlipVertical?: boolean;
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
  /** For source nodes: full input-source catalog from the backend
   *  (including plugin-registered ones). Used to build the Source
   *  dropdown dynamically. */
  availableInputSources?: import("./api").InputSourceType[];
  /** For source/output nodes: install hint shown when Spout is unavailable */
  spoutReason?: string | null;
  /** For source/output nodes: install hint shown when NDI is unavailable */
  ndiReason?: string | null;
  /** For source/output nodes: install hint shown when Syphon is unavailable */
  syphonReason?: string | null;
  /** For source nodes: callback when Spout receiver name changes */
  onSpoutSourceChange?: (name: string) => void;
  /** For source nodes: callback when NDI source changes */
  onNdiSourceChange?: (identifier: string) => void;
  /** For source nodes: callback when Syphon source changes */
  onSyphonSourceChange?: (identifier: string) => void;
  /** For source nodes: callback to cycle through sample videos */
  onCycleSampleVideo?: () => void;
  /** For source nodes: callback to initialize the first sample video (test.mp4) */
  onInitSampleVideo?: () => void;
  /** For sink nodes: remote output stream */
  remoteStream?: MediaStream | null;
  /** For sink nodes: per-sink WebRTC stats (FPS, bitrate) */
  sinkStats?: { fps: number; bitrate: number };
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
  /** Whether the stream is currently active (used to disable load params) */
  isStreaming?: boolean;

  /* ── Record node fields ── */
  /** For record nodes: callback to start recording (node_id is bound by enrichment) */
  onStartRecording?: () => void;
  /** For record nodes: callback to stop recording (node_id is bound by enrichment) */
  onStopRecording?: () => void;
  /** For record nodes: incoming trigger value from connected nodes */
  triggerValue?: boolean;

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

  /* ── Audio node fields ── */
  /** For audio nodes: the selected audio asset path */
  audioPath?: string;

  /* ── Image / Media node fields ── */
  /** For image/media nodes: the selected asset path (image or video) */
  imagePath?: string;
  /** For image/media nodes: detected media type based on file extension */
  mediaType?: "image" | "video";
  /** For video media nodes: playback loop mode */
  videoLoopMode?: "none" | "loop" | "ping-pong";

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
  /** For bool nodes: armed state for boolean trigger sources */
  boolTriggerArmed?: boolean;
  /** For bool nodes: per-edge fire count tracking for counter trigger sources */
  _boolTriggerCounters?: Record<string, number>;
  /** For bool nodes: timestamp of last gate fire (for auto-reset) */
  _boolGateTimer?: number;

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

  /* ── LoRA node fields ── */
  /** For lora nodes: list of configured LoRA adapters */
  loras?: Array<{ path: string; scale: number; mergeMode?: string }>;
  /** For lora nodes: global merge strategy */
  loraMergeMode?: string;

  /* ── Pipeline LoRA support ── */
  /** For pipeline nodes: whether the selected pipeline supports LoRA */
  supportsLoRA?: boolean;

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

  /* ── Tempo node fields ── */
  tempoBpm?: number | null;
  tempoBeatPhase?: number;
  tempoBeatCount?: number;
  tempoBarPosition?: number;
  tempoIsPlaying?: boolean;
  tempoEnabled?: boolean;
  tempoSourceType?: string | null;
  tempoNumPeers?: number | null;
  tempoBeatsPerBar?: number;
  tempoLoading?: boolean;
  tempoError?: string | null;
  tempoSources?: unknown;
  onEnableTempo?: (req: import("./api").TempoEnableRequest) => void;
  onDisableTempo?: () => void;
  onSetTempo?: (bpm: number) => void;
  onRefreshTempoSources?: () => void;
  tempoQuantizeMode?: string;
  tempoLookaheadMs?: number;
  tempoBeatResetRate?: string;

  /* ── Prompt list node fields ── */
  promptListItems?: string[];
  promptListActiveIndex?: number;
  promptListActiveText?: string;
  promptListCycleValue?: number;

  /* ── Prompt blend node fields ── */
  promptBlendItems?: Array<{ text: string; weight: number }>;
  promptBlendMethod?: "linear" | "slerp";

  /* ── Scheduler node fields ── */
  schedulerTriggers?: Array<{ time: number; port_name: string }>;
  schedulerDuration?: number;
  schedulerLoop?: boolean;
  schedulerElapsed?: number;
  schedulerIsPlaying?: boolean;
  schedulerFireCounts?: Record<string, number>;
  schedulerTickCount?: number;
  _schedulerStartCount?: number;
  _schedulerStartArmed?: boolean;
  _schedulerResetCount?: number;
  _schedulerResetArmed?: boolean;

  /* ── Tempo beat count offset ── */
  tempoBeatCountOffset?: number;

  /* ── Custom node fields ── */
  /** For custom_node: the node_type_id from the backend registry */
  customNodeTypeId?: string;
  /** For custom_node: display name from node definition */
  customNodeDisplayName?: string;
  /** For custom_node: category from node definition */
  customNodeCategory?: string;
  /** For custom_node: input port definitions */
  customNodeInputs?: NodePortDef[];
  /** For custom_node: output port definitions */
  customNodeOutputs?: NodePortDef[];
  /** For custom_node: current parameter values (user-editable) */
  customNodeParams?: Record<string, unknown>;
  /** For custom_node: parameter definitions from API (widget metadata) */
  customNodeParamDefs?: NodeParamDef[];
  /** For custom_node: push a param change through to the backend so the
   *  running node picks up the new value without a session restart. */
  onCustomNodeParamChange?: (key: string, value: unknown) => void;

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
 * CustomNode handles are namespaced with a direction prefix so input and
 * output ports that share the same name (e.g. "video" passthrough) don't
 * collide on handle id. Edges store only the bare port name on the wire
 * format, so consumers strip the prefix before looking up the port and
 * re-add it when rebuilding handles from saved graphs.
 */
const CUSTOM_NODE_IN_PREFIX = "in:";
const CUSTOM_NODE_OUT_PREFIX = "out:";

export function customNodeInputHandleId(portName: string): string {
  return buildHandleId("stream", `${CUSTOM_NODE_IN_PREFIX}${portName}`);
}

export function customNodeOutputHandleId(portName: string): string {
  return buildHandleId("stream", `${CUSTOM_NODE_OUT_PREFIX}${portName}`);
}

/** Strip the ``in:``/``out:`` prefix that CustomNode adds to handle names. */
export function stripCustomNodeDirection(name: string): string {
  if (name.startsWith(CUSTOM_NODE_IN_PREFIX)) {
    return name.slice(CUSTOM_NODE_IN_PREFIX.length);
  }
  if (name.startsWith(CUSTOM_NODE_OUT_PREFIX)) {
    return name.slice(CUSTOM_NODE_OUT_PREFIX.length);
  }
  return name;
}

/**
 * Prompt port alias: the UI renders the "Enter prompt…" textarea under the
 * widget handle ``param:__prompt`` (a composite pseudo-param that also
 * stores raw widget state), while the backend pipeline exposes the
 * matching port as ``prompts`` so the kwarg name matches at runtime.
 * These two constants and the helpers below are the single spot where the
 * rename is known — callers that import/export graphs translate through
 * ``aliasWirePortForPipelineTarget`` / ``aliasWidgetPortForWire``.
 */
export const PROMPT_WIRE_PORT = "prompts";
export const PROMPT_WIDGET_PORT = "__prompt";

/** On import: rewrite the on-wire ``prompts`` target port to the widget name. */
export function aliasWirePortForPipelineTarget(
  toNode: string,
  toPort: string,
  pipelineTargetIds: ReadonlySet<string>
): string {
  if (pipelineTargetIds.has(toNode) && toPort === PROMPT_WIRE_PORT) {
    return PROMPT_WIDGET_PORT;
  }
  return toPort;
}

/** On export: rewrite the widget ``__prompt`` target name back to the wire port. */
export function aliasWidgetPortForWire(
  targetNodeId: string,
  toPort: string,
  kind: "stream" | "parameter",
  pipelineTargetIds: ReadonlySet<string>
): string {
  if (
    kind === "stream" &&
    toPort === PROMPT_WIDGET_PORT &&
    pipelineTargetIds.has(targetNodeId)
  ) {
    return PROMPT_WIRE_PORT;
  }
  return toPort;
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
    //  vace_context_scale has component "vace" and is handled by the VACE node,
    //  lora_merge_strategy has component "lora" and is handled by the LoRA node)
    if (
      schemaProp.ui.component === "cache" ||
      schemaProp.ui.component === "vace" ||
      schemaProp.ui.component === "lora"
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
        isLoadParam: ui?.is_load_param,
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
          isLoadParam: ui?.is_load_param,
        });
        continue;
      }
    }

    // Resolve $ref-based enums from $defs (e.g. Python Enum classes via Pydantic).
    // Also handles anyOf: [{ $ref: "..." }, { type: "null" }] (nullable enum).
    const resolveEnumFromRef = (): unknown[] | undefined => {
      const defs = schema.config_schema?.$defs;
      if (!defs) return undefined;

      const resolveRef = (ref: string): unknown[] | undefined => {
        const refName = ref.split("/").pop();
        if (!refName) return undefined;
        const def = defs[refName];
        return def?.enum && Array.isArray(def.enum) ? def.enum : undefined;
      };

      if (schemaProp.$ref) return resolveRef(schemaProp.$ref);

      if (anyOf?.length) {
        for (const variant of anyOf) {
          const ref = (variant as Record<string, unknown>).$ref as
            | string
            | undefined;
          if (ref) return resolveRef(ref);
        }
      }
      return undefined;
    };

    const fieldType = inferPrimitiveFieldType(schemaProp);
    const refEnumValues = resolveEnumFromRef();
    // If inferPrimitiveFieldType missed a $ref inside anyOf, treat it as enum
    const effectiveFieldType = !fieldType && refEnumValues ? "enum" : fieldType;
    if (!effectiveFieldType) continue;

    let paramType: "string" | "number" | "boolean" | null = null;
    if (effectiveFieldType === "text" || effectiveFieldType === "enum") {
      paramType = "string";
    } else if (
      effectiveFieldType === "number" ||
      effectiveFieldType === "slider"
    ) {
      paramType = "number";
    } else if (effectiveFieldType === "toggle") {
      paramType = "boolean";
    }

    if (!paramType) continue;

    // Detect integer type from schema (direct type or non-null anyOf variant)
    const isInteger =
      schemaProp.type === "integer" ||
      (anyOf?.some(
        v =>
          (v as Record<string, unknown>).type === "integer" &&
          (v as Record<string, unknown>).type !== "null"
      ) ??
        false);

    const ui = schemaProp.ui;
    const label = ui?.label || key;
    const baseEnumValues = Array.isArray(schemaProp.enum)
      ? schemaProp.enum
      : refEnumValues;
    // For nullable enums (anyOf with { type: "null" }), prepend null so the UI
    // can render a "None" option.
    const isNullable = anyOf?.some(
      v => (v as Record<string, unknown>).type === "null"
    );
    const enumValues =
      baseEnumValues && isNullable ? [null, ...baseEnumValues] : baseEnumValues;

    // For nullable numbers, pull min/max from the non-null anyOf variant
    let minimum = schemaProp.minimum;
    let maximum = schemaProp.maximum;
    if (minimum === undefined || maximum === undefined) {
      if (anyOf?.length) {
        const numVariant = anyOf.find(
          v =>
            ((v as Record<string, unknown>).type === "integer" ||
              (v as Record<string, unknown>).type === "number") &&
            (v as Record<string, unknown>).type !== "null"
        ) as Record<string, unknown> | undefined;
        if (numVariant) {
          if (minimum === undefined && typeof numVariant.minimum === "number")
            minimum = numVariant.minimum;
          if (maximum === undefined && typeof numVariant.maximum === "number")
            maximum = numVariant.maximum;
        }
      }
    }

    params.push({
      name: key,
      type: paramType,
      defaultValue: schemaProp.default,
      label,
      min: typeof minimum === "number" ? minimum : undefined,
      max: typeof maximum === "number" ? maximum : undefined,
      step: paramType === "number" && isInteger ? 1 : undefined,
      enum: enumValues,
      isLoadParam: ui?.is_load_param,
    });
  }

  if (schema?.supports_cache_management) {
    params.push({
      name: "reset_cache",
      type: "boolean",
      defaultValue: false,
      label: "Reset Cache",
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

  const sources = graph.nodes.filter(
    n => n.type === "source" && !isSubgraphInnerNode(n.id)
  );
  const pipelines = graph.nodes.filter(
    n => n.type === "pipeline" && !isSubgraphInnerNode(n.id)
  );
  // Separate regular sinks from output-sink nodes (sink_mode set).
  // Output-sink nodes are restored from ui_state as OutputNodes.
  const outputSinkNodes = graph.nodes.filter(
    n => n.type === "sink" && !isSubgraphInnerNode(n.id) && n.sink_mode
  );
  const sinks = graph.nodes.filter(
    n => n.type === "sink" && !isSubgraphInnerNode(n.id) && !n.sink_mode
  );

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
        sourceMode: n.source_mode ?? "video",
        sourceName: n.source_name ?? undefined,
        sourceFlipVertical: n.source_flip_vertical ?? false,
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

  const records = graph.nodes.filter(
    n => n.type === "record" && !isSubgraphInnerNode(n.id)
  );
  records.forEach((n, i) => {
    const savedX = n.x ?? undefined;
    const savedY = n.y ?? undefined;
    const w = n.w ?? 180;
    const h = n.h ?? 95;
    nodes.push({
      id: n.id,
      type: "record",
      position: {
        x: savedX !== undefined ? savedX : START_X + COLUMN_GAP * 3,
        y:
          savedY !== undefined ? savedY : START_Y + i * (NODE_HEIGHT + ROW_GAP),
      },
      width: w,
      height: h,
      style: { width: w, height: h },
      data: { label: n.id, nodeType: "record" },
    });
  });

  // Backend custom nodes (type="node"). Port metadata is hydrated later
  // from GET /api/v1/nodes/definitions in useGraphPersistence.
  const customNodes = graph.nodes.filter(
    n => n.type === "node" && !isSubgraphInnerNode(n.id)
  );
  customNodes.forEach((n, i) => {
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
    const position = {
      x: savedX !== undefined ? savedX : START_X + COLUMN_GAP * 1.5,
      y: savedY !== undefined ? savedY : START_Y + i * (NODE_HEIGHT + ROW_GAP),
    };
    // Scheduler has its own React renderer — rehydrate into its native
    // nodeType so the bespoke widget (trigger list, transport controls)
    // comes back after a round-trip, instead of the generic custom_node UI.
    if (n.node_type_id === "scheduler") {
      const params = (n.params ?? {}) as Record<string, unknown>;
      nodes.push({
        id: n.id,
        type: "scheduler",
        position,
        ...sizeProps,
        data: {
          label: "Scheduler",
          nodeType: "scheduler",
          schedulerTriggers:
            (params.triggers as Array<{ time: number; port_name: string }>) ??
            [],
          schedulerLoop: (params.loop as boolean) ?? false,
          schedulerDuration: (params.duration as number) ?? 30,
          schedulerElapsed: 0,
          schedulerIsPlaying: false,
          schedulerFireCounts: {},
          schedulerTickCount: 0,
        },
      });
      return;
    }
    nodes.push({
      id: n.id,
      type: "custom_node",
      position,
      ...sizeProps,
      data: {
        label: n.node_type_id || n.id,
        nodeType: "custom_node",
        customNodeTypeId: n.node_type_id ?? undefined,
        customNodeParams: (n.params as Record<string, unknown>) ?? undefined,
      },
    });
  });

  // Convert edges - add stream: prefix to handle IDs
  // Skip edges that reference flattened inner subgraph nodes
  // Custom nodes namespace their handles with in:/out: so the same port
  // name on both sides doesn't collide; restore that prefix here when
  // rebuilding handles for edges touching a custom_node.
  const customNodeIds = new Set(
    customNodes
      .map(n => n.id)
      .concat(
        // Also include custom_node entries that may have been pushed via
        // other code paths (defensive, cheap).
        nodes.filter(n => n.type === "custom_node").map(n => n.id)
      )
  );
  const pipelineNodeIds = new Set(pipelines.map(n => n.id));
  const edges: Edge[] = graph.edges
    .filter(
      e => !isSubgraphInnerNode(e.from) && !isSubgraphInnerNode(e.to_node)
    )
    .map((e, i) => {
      const toPort = aliasWirePortForPipelineTarget(
        e.to_node,
        e.to_port,
        pipelineNodeIds
      );
      const isPromptPortTarget = toPort !== e.to_port;
      const sourceHandle =
        e.kind === "parameter"
          ? buildHandleId("param", e.from_port)
          : customNodeIds.has(e.from)
            ? customNodeOutputHandleId(e.from_port)
            : buildHandleId("stream", e.from_port);
      const targetHandle = isPromptPortTarget
        ? buildHandleId("param", toPort)
        : e.kind === "parameter"
          ? buildHandleId("param", e.to_port)
          : customNodeIds.has(e.to_node)
            ? customNodeInputHandleId(e.to_port)
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
      if (nodeType === "record") {
        continue;
      }
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

  // Fallback: create OutputNodes for sink_mode nodes not restored from ui_state
  const restoredIds = new Set(nodes.map(n => n.id));
  for (const n of outputSinkNodes) {
    if (restoredIds.has(n.id)) continue;
    const savedX = n.x ?? undefined;
    const savedY = n.y ?? undefined;
    nodes.push({
      id: n.id,
      type: "output",
      position: {
        x: savedX !== undefined ? savedX : START_X + COLUMN_GAP * 2 + 300,
        y: savedY !== undefined ? savedY : START_Y,
      },
      data: {
        label: n.sink_name || "Output",
        nodeType: "output",
        outputSinkType: n.sink_mode ?? "spout",
        outputSinkEnabled: true,
        outputSinkName: n.sink_name ?? "",
      },
    });
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
  "audio",
  "vace",
  "lora",
  "midi",
  "bool",
  "trigger",
  "subgraph",
  "subgraph_input",
  "subgraph_output",
  "tempo",
  "prompt_list",
  "prompt_blend",
]);

/** Fields in FlowNodeData that are non-serializable (functions, streams, etc.) */
const NON_SERIALIZABLE_KEYS = new Set<string>([
  "localStream",
  "remoteStream",
  "sinkStats",
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
  "onStartRecording",
  "onStopRecording",
  "triggerValue",
  "onEnableTempo",
  "onDisableTempo",
  "onSetTempo",
  "onRefreshTempoSources",
  "tempoSources",
  "committedValue",
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

    // Build port maps using actual flattened node ids (required for nested subgraphs:
    // port.innerNodeId may refer to a node deep inside, so resolve to hoisted id)
    const resolveFlattenedId = (innerNodeId: string): string => {
      const direct = sgPrefix + innerNodeId;
      const exact = hoisted.find(n => n.id === direct);
      if (exact) return exact.id;
      const suffix = ":" + innerNodeId;
      const nested = hoisted.find(n => n.id.endsWith(suffix));
      return nested ? nested.id : direct;
    };
    for (const port of sgInputs) {
      inputPortMap.set(`${sg.id}::${port.name}`, {
        nodeId: resolveFlattenedId(port.innerNodeId),
        handleId: port.innerHandleId,
      });
    }
    for (const port of sgOutputs) {
      outputPortMap.set(`${sg.id}::${port.name}`, {
        nodeId: resolveFlattenedId(port.innerNodeId),
        handleId: port.innerHandleId,
      });
    }
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

  // Determine which pipeline nodes are connected to a tempo node (directly
  // or transitively through reroute nodes).  Only those pipelines will get
  // backend-side beat injection (tempo_sync flag on GraphNode).
  const tempoConnectedPipelineIds = new Set<string>();
  const tempoNodeIds = flatNodes
    .filter(n => n.data.nodeType === "tempo")
    .map(n => n.id);
  if (tempoNodeIds.length > 0) {
    const flatNodeMap = new Map(flatNodes.map(n => [n.id, n]));
    const edgesBySource = new Map<string, Edge[]>();
    for (const e of flatEdges) {
      const list = edgesBySource.get(e.source) ?? [];
      list.push(e);
      edgesBySource.set(e.source, list);
    }
    const traceToBackendNodes = (nodeId: string, visited: Set<string>) => {
      if (visited.has(nodeId)) return;
      visited.add(nodeId);
      for (const edge of edgesBySource.get(nodeId) ?? []) {
        const target = flatNodeMap.get(edge.target);
        if (!target) continue;
        if (target.data.nodeType === "pipeline") {
          tempoConnectedPipelineIds.add(target.id);
        } else if (target.data.nodeType === "reroute") {
          traceToBackendNodes(target.id, visited);
        }
      }
    };
    for (const tid of tempoNodeIds) {
      traceToBackendNodes(tid, new Set());
    }
  }

  const graphNodes: GraphNode[] = backendFlatNodes.map(n => {
    // Read dimensions: node.width/height (set by NodeResizer) > measured > style
    const w =
      n.width ??
      n.measured?.width ??
      (typeof n.style?.width === "number" ? n.style.width : undefined);
    const h =
      n.height ??
      n.measured?.height ??
      (typeof n.style?.height === "number" ? n.style.height : undefined);
    const isBackendNode =
      n.data.nodeType === "custom_node" || n.data.nodeType === "scheduler";
    const schedulerParams =
      n.data.nodeType === "scheduler"
        ? {
            triggers: n.data.schedulerTriggers ?? [],
            loop: n.data.schedulerLoop ?? false,
            duration: n.data.schedulerDuration ?? 30,
          }
        : undefined;
    return {
      id: n.id,
      type:
        n.data.nodeType === "source"
          ? "source"
          : n.data.nodeType === "sink"
            ? "sink"
            : n.data.nodeType === "record"
              ? "record"
              : isBackendNode
                ? "node"
                : "pipeline",
      pipeline_id:
        n.data.nodeType === "pipeline"
          ? (n.data.pipelineId ?? null)
          : undefined,
      node_type_id:
        n.data.nodeType === "custom_node"
          ? (n.data.customNodeTypeId ?? null)
          : n.data.nodeType === "scheduler"
            ? "scheduler"
            : undefined,
      params:
        n.data.nodeType === "custom_node" && n.data.customNodeParams
          ? n.data.customNodeParams
          : schedulerParams,
      x: n.position.x,
      y: n.position.y,
      w: w && !Number.isNaN(w) ? w : undefined,
      h: h && !Number.isNaN(h) ? h : undefined,
      source_mode:
        n.data.nodeType === "source" ? (n.data.sourceMode ?? null) : undefined,
      source_name:
        n.data.nodeType === "source" ? (n.data.sourceName ?? null) : undefined,
      source_flip_vertical:
        n.data.nodeType === "source"
          ? Boolean(n.data.sourceFlipVertical)
          : undefined,
      tempo_sync: tempoConnectedPipelineIds.has(n.id) || undefined,
    };
  });

  // Convert enabled OutputNodes to backend sink nodes with sink_mode/sink_name.
  // This allows the backend's multi-sink system to create per-node Syphon/Spout/NDI
  // senders, each receiving frames from the specific pipeline they're connected to.
  for (const n of flatNodes) {
    if (n.data.nodeType !== "output") continue;
    const enabled = (n.data.outputSinkEnabled as boolean) ?? false;
    if (!enabled) continue;

    const sinkType = (n.data.outputSinkType as string) || "spout";
    const sinkName = (n.data.outputSinkName as string) || "";
    const ow =
      n.width ??
      n.measured?.width ??
      (typeof n.style?.width === "number" ? n.style.width : undefined);
    const oh =
      n.height ??
      n.measured?.height ??
      (typeof n.style?.height === "number" ? n.style.height : undefined);

    graphNodes.push({
      id: n.id,
      type: "sink",
      x: n.position.x,
      y: n.position.y,
      w: ow && !Number.isNaN(ow) ? ow : undefined,
      h: oh && !Number.isNaN(oh) ? oh : undefined,
      sink_mode: sinkType,
      sink_name: sinkName,
    });
  }

  // Filter edges to only include those where both source and target exist in graphNodes
  // Strip the in:/out: prefix added to custom_node handles before writing
  // the bare port name back to the wire format.
  const graphNodeIds = new Set(graphNodes.map(n => n.id));
  const pipelineTargetIds = new Set(
    graphNodes.filter(n => n.type === "pipeline").map(n => n.id)
  );
  const graphEdges: GraphEdge[] = flatEdges
    .filter(e => graphNodeIds.has(e.source) && graphNodeIds.has(e.target))
    .map(e => {
      const sourceParsed = parseHandleId(e.sourceHandle);
      const targetParsed = parseHandleId(e.targetHandle);
      const kind =
        sourceParsed?.kind === "param" && targetParsed?.kind === "param"
          ? "parameter"
          : "stream";
      const fromName = sourceParsed?.name
        ? stripCustomNodeDirection(sourceParsed.name)
        : "video";
      const toName = aliasWidgetPortForWire(
        e.target,
        targetParsed?.name
          ? stripCustomNodeDirection(targetParsed.name)
          : "video",
        kind as "stream" | "parameter",
        pipelineTargetIds
      );
      return {
        from: e.source,
        from_port: fromName,
        to_node: e.target,
        to_port: toName,
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

    // Edges that touch at least one frontend-only node but aren't already
    // in graphEdges (both endpoints in graphNodeIds means it's a backend edge)
    const uiEdges: UIStateEdge[] = edges
      .filter(
        e =>
          (frontendNodeIds.has(e.source) || frontendNodeIds.has(e.target)) &&
          !(graphNodeIds.has(e.source) && graphNodeIds.has(e.target))
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
  postprocessorIds: string[],
  vaceInputVideoIds?: Set<string>
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
  for (const { nodeId, pid } of nodeEntries) {
    const toPort = vaceInputVideoIds?.has(pid) ? "vace_input_frames" : "video";
    edges.push({
      from: prev,
      from_port: "video",
      to_node: nodeId,
      to_port: toPort,
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
 * Perform mode sends a linear graph whose default source is WebRTC (`video`).
 * When the user selects Spout, NDI, or Syphon, `input_source` is also set, but
 * the backend drops `input_source` whenever the session includes any graph
 * source node — hardware capture is wired only from per-node `source_mode` /
 * `source_name` (see FrameProcessor.start). Patch the linear graph's `input`
 * node so multi-source setup matches Workflow Builder.
 */
export function applyHardwareInputSourceToLinearGraph(
  graph: GraphConfig,
  inputSource?: {
    enabled: boolean;
    source_type: string;
    source_name: string;
    flip_vertical?: boolean;
  }
): GraphConfig {
  const t = inputSource?.source_type;
  if (
    !inputSource?.enabled ||
    (t !== "ndi" && t !== "spout" && t !== "syphon")
  ) {
    return graph;
  }
  return {
    ...graph,
    nodes: graph.nodes.map(n =>
      n.id === "input" && n.type === "source"
        ? {
            ...n,
            source_mode: t,
            source_name: inputSource.source_name ?? "",
            source_flip_vertical:
              t === "syphon" ? Boolean(inputSource.flip_vertical) : false,
          }
        : n
    ),
  };
}

/**
 * Drop layout fields (x, y, w, h) and omit `ui_state` from the payload to the
 * server. Execution topology — including `type: "record"` nodes and their
 * edges — stays in `nodes` / `edges`; record nodes are not frontend-only.
 */
export function stripUIFields(graph: GraphConfig): GraphConfig {
  return {
    nodes: graph.nodes.map(({ x: _x, y: _y, w: _w, h: _h, ...rest }) => rest),
    edges: graph.edges,
  };
}

// ---------------------------------------------------------------------------
// Import: ScopeWorkflow (without graph field) -> GraphConfig
// ---------------------------------------------------------------------------

const VACE_PORTS = ["vace_input_frames", "vace_input_masks"];

/**
 * Build a GraphConfig from a ScopeWorkflow that has no embedded graph.
 *
 * For simple chains this produces: source -> pipeline1 -> pipeline2 -> ... -> sink.
 * When a preprocessor declares VACE-specific output ports (e.g. yolo_mask with
 * vace_input_frames + vace_input_masks), the topology fans out so the source
 * feeds both the preprocessor and the main pipeline, and the preprocessor
 * connects to the main pipeline via its VACE ports.
 *
 * When `vace_enabled` is true on a pipeline, a graph `vace` node is added and
 * connected via `param:__vace`. An empty `image` (Media) node is placed to the
 * left and wired to `ref_image` with no path; workflow `vace_ref_images` stay
 * in pipeline `node_params` for runtime.
 */
export function workflowToGraphConfig(
  workflow: {
    pipelines: Array<{
      pipeline_id: string;
      params?: Record<string, unknown>;
      loras?: Array<{
        filename: string;
        weight: number;
        merge_mode?: string;
      }>;
      role?: string | null;
    }>;
    prompts?: Array<{ text: string; weight: number }> | null;
    timeline?: { entries: Array<{ prompts: Array<{ text: string }> }> } | null;
  },
  options?: {
    availableLoRAs?: LoRAFileInfo[];
    portsMap?: Record<string, { inputs: string[]; outputs: string[] }>;
  }
): {
  graphConfig: GraphConfig;
  nodeParams: Record<string, Record<string, unknown>>;
} {
  const nodes: GraphNode[] = [];
  const edges: GraphEdge[] = [];
  const nodeParams: Record<string, Record<string, unknown>> = {};
  const uiNodes: UIStateNode[] = [];
  const uiEdges: UIStateEdge[] = [];

  const Y = 200;
  let x = START_X;

  // Source node
  const sourceId = "input";
  nodes.push({ id: sourceId, type: "source", source_mode: "video", x, y: Y });
  x += COLUMN_GAP;

  const usedIds = new Set<string>([sourceId]);

  // Determine main pipeline before the loop so we can track its generated nodeId
  const mainPipeline =
    workflow.pipelines.find(p => p.role === "main") ?? workflow.pipelines[0];

  const mainIsVaceTarget =
    mainPipeline.params?.vace_enabled === true &&
    mainPipeline.params?.vace_use_input_video === true;

  // -- Pass 1: create all pipeline nodes and collect metadata ----------------
  interface PipelineNodeMeta {
    wp: (typeof workflow.pipelines)[number];
    nodeId: string;
    x: number;
    y: number;
    isMain: boolean;
    isVaceTarget: boolean;
    outputPorts: string[];
  }
  const pipelineMetas: PipelineNodeMeta[] = [];

  for (const wp of workflow.pipelines) {
    const nodeId = generateNodeId(wp.pipeline_id, usedIds);
    usedIds.add(nodeId);

    const isMain = wp === mainPipeline;
    const isVaceTarget =
      wp.params?.vace_enabled === true &&
      wp.params?.vace_use_input_video === true;
    const outputPorts = options?.portsMap?.[wp.pipeline_id]?.outputs ?? [
      "video",
    ];

    pipelineMetas.push({
      wp,
      nodeId,
      x,
      y: Y,
      isMain,
      isVaceTarget,
      outputPorts,
    });
    x += COLUMN_GAP;
  }

  // Identify preprocessors that should use VACE multi-port routing:
  // the preprocessor must declare VACE output ports AND the main pipeline
  // must be a VACE target.
  const vaceRoutedPreprocessors = new Set<string>();
  if (mainIsVaceTarget) {
    for (const meta of pipelineMetas) {
      if (
        meta.wp.role === "preprocessor" &&
        VACE_PORTS.some(p => meta.outputPorts.includes(p))
      ) {
        vaceRoutedPreprocessors.add(meta.nodeId);
      }
    }
  }

  const mainMeta = pipelineMetas.find(m => m.isMain);
  // When VACE multi-port preprocessors are present, offset them below the main
  // pipeline row so the fan-out layout is visually clear.
  if (vaceRoutedPreprocessors.size > 0) {
    for (const meta of pipelineMetas) {
      if (vaceRoutedPreprocessors.has(meta.nodeId)) {
        meta.y = Y + 300;
      }
    }
  }

  // -- Pass 1b: create GraphNode entries ------------------------------------
  for (const meta of pipelineMetas) {
    nodes.push({
      id: meta.nodeId,
      type: "pipeline",
      pipeline_id: meta.wp.pipeline_id,
      x: meta.x,
      y: meta.y,
    });

    if (meta.wp.params && Object.keys(meta.wp.params).length > 0) {
      nodeParams[meta.nodeId] = { ...meta.wp.params };
    }
  }

  // -- Pass 2: create edges with full topology context ----------------------
  let prevNodeId = sourceId;
  let loraIdx = 0;

  // Track whether we already added a source -> main edge (to avoid duplicates
  // when multiple preprocessors use VACE routing).
  let addedSourceToMain = false;

  for (const meta of pipelineMetas) {
    const { wp, nodeId } = meta;

    if (vaceRoutedPreprocessors.has(nodeId) && mainMeta) {
      // VACE multi-port preprocessor: fan-out topology
      // 1) source -> preprocessor (video -> video)
      edges.push({
        from: sourceId,
        from_port: "video",
        to_node: nodeId,
        to_port: "video",
        kind: "stream",
      });
      // 2) source -> main pipeline (video -> video) -- only once
      if (!addedSourceToMain) {
        edges.push({
          from: sourceId,
          from_port: "video",
          to_node: mainMeta.nodeId,
          to_port: "video",
          kind: "stream",
        });
        addedSourceToMain = true;
      }
      // 3) preprocessor -> main pipeline for each VACE output port
      for (const port of VACE_PORTS) {
        if (meta.outputPorts.includes(port)) {
          edges.push({
            from: nodeId,
            from_port: port,
            to_node: mainMeta.nodeId,
            to_port: port,
            kind: "stream",
          });
        }
      }
      // Don't update prevNodeId -- the main pipeline is fed by source + preprocessor,
      // not linearly from this preprocessor's video output.
    } else {
      // Standard linear edge from the previous node
      const toPort = meta.isVaceTarget ? "vace_input_frames" : "video";
      edges.push({
        from: prevNodeId,
        from_port: "video",
        to_node: nodeId,
        to_port: toPort,
        kind: "stream",
      });
      prevNodeId = nodeId;
    }

    // Create LoRA node for pipelines that have LoRAs configured
    if (wp.loras && wp.loras.length > 0) {
      const loraNodeId = `lora-${loraIdx++}`;
      const loraEntries = wp.loras.map(l => ({
        path: options?.availableLoRAs
          ? resolveLoRAPath(l.filename, options.availableLoRAs)
          : l.filename,
        scale: l.weight,
        mergeMode: l.merge_mode ?? "permanent_merge",
      }));
      const globalMergeMode = wp.loras[0].merge_mode ?? "permanent_merge";

      uiNodes.push({
        id: loraNodeId,
        type: "lora",
        position: { x: meta.x - 180, y: meta.y - 160 },
        data: {
          label: "LoRA",
          nodeType: "lora",
          loras: loraEntries,
          loraMergeMode: globalMergeMode,
        } as FlowNodeData,
      });

      const edgeId = `e-${loraNodeId}-${nodeId}`;
      uiEdges.push({
        id: edgeId,
        source: loraNodeId,
        sourceHandle: buildHandleId("param", "__loras"),
        target: nodeId,
        targetHandle: buildHandleId("param", "__loras"),
      });
    }

    // VACE compound node: matches graph-native workflows so the pipeline shows
    // "VACE: Connected" and getGraphVaceSettings() can supply ref/context.
    if (wp.params?.vace_enabled === true) {
      const refImageNodeId = generateNodeId("ref-image", usedIds);
      usedIds.add(refImageNodeId);

      const vaceNodeId = generateNodeId("vace", usedIds);
      usedIds.add(vaceNodeId);

      const vaceContextScale =
        typeof wp.params.vace_context_scale === "number"
          ? wp.params.vace_context_scale
          : 1.0;

      const vaceY =
        wp.loras && wp.loras.length > 0 ? meta.y - 360 : meta.y - 160;

      // Empty Media node (Ref Image slot); paths remain in pipeline node_params only.
      uiNodes.push({
        id: refImageNodeId,
        type: "image",
        position: { x: meta.x - 420, y: vaceY },
        width: 160,
        height: 140,
        data: {
          label: "Ref Image",
          customTitle: "Ref Image",
          nodeType: "image",
          imagePath: "",
          mediaType: "image",
          parameterOutputs: [
            { name: "value", type: "string", defaultValue: "" },
          ],
        } as FlowNodeData,
      });

      uiNodes.push({
        id: vaceNodeId,
        type: "vace",
        position: { x: meta.x - 180, y: vaceY },
        width: 240,
        height: 178,
        data: {
          label: "VACE",
          nodeType: "vace",
          vaceContextScale,
          vaceRefImage: "",
          vaceFirstFrame: "",
          vaceLastFrame: "",
          vaceVideo: "",
          parameterOutputs: [
            { name: "__vace", type: "string", defaultValue: "" },
          ],
        } as FlowNodeData,
      });

      uiEdges.push({
        id: `e-${refImageNodeId}-${vaceNodeId}-ref`,
        source: refImageNodeId,
        sourceHandle: buildHandleId("param", "value"),
        target: vaceNodeId,
        targetHandle: buildHandleId("param", "ref_image"),
      });

      uiEdges.push({
        id: `e-${vaceNodeId}-${nodeId}`,
        source: vaceNodeId,
        sourceHandle: buildHandleId("param", "__vace"),
        target: nodeId,
        targetHandle: buildHandleId("param", "__vace"),
      });
    }
  }

  // Sink node
  const sinkId = "output";
  nodes.push({ id: sinkId, type: "sink", x, y: Y });
  edges.push({
    from: prevNodeId,
    from_port: "video",
    to_node: sinkId,
    to_port: "video",
    kind: "stream",
  });

  // Assign prompt to the main pipeline node (using tracked nodeId, not pipeline_id)
  const mainPipelineNodeId = mainMeta?.nodeId;
  if (mainPipelineNodeId) {
    const promptText =
      workflow.timeline?.entries?.[0]?.prompts?.[0]?.text ??
      workflow.prompts?.[0]?.text;
    if (promptText) {
      const bag = nodeParams[mainPipelineNodeId] ?? {};
      bag.__prompt = promptText;
      nodeParams[mainPipelineNodeId] = bag;
    }
  }

  const uiState: Record<string, unknown> = {};
  if (Object.keys(nodeParams).length > 0) {
    uiState.node_params = nodeParams;
  }
  if (uiNodes.length > 0) {
    uiState.nodes = uiNodes;
  }
  if (uiEdges.length > 0) {
    uiState.edges = uiEdges;
  }

  return {
    graphConfig: {
      nodes,
      edges,
      ui_state: Object.keys(uiState).length > 0 ? uiState : null,
    },
    nodeParams,
  };
}

// Default node dimensions for reference
export { NODE_WIDTH, NODE_HEIGHT };
