/**
 * Per-node-type OSC parameter descriptors.
 *
 * These describe which fields on a non-pipeline node's `data` are eligible
 * to be exposed via OSC. Pipeline nodes use the live pipeline schema
 * instead — see `pickPipelineOscParams` below.
 *
 * Keep this list minimal and additive. We deliberately skip composite /
 * dynamic-shaped fields (knobs[], midiChannels[], tupleValues[]) for the
 * MVP — those need a richer addressing scheme (e.g. /scope/.../knob_0)
 * that can be added in a follow-up without a breaking API change.
 */
import type { FlowNodeData } from "./graphUtils";
import type { PipelineConfigSchema, PipelineSchemaProperty } from "./api";

export type OscParamType =
  | "float"
  | "integer"
  | "bool"
  | "string"
  | "integer_list";

export interface OscParamDescriptor {
  /** Field name on `node.data` we read/write. Also the default param name in the OSC address. */
  name: string;
  /** Human label for the Configure OSC modal. */
  label: string;
  type: OscParamType;
  /** Optional constraints surfaced in OSC docs + used for validation. */
  min?: number;
  max?: number;
  enum?: string[];
  /** Brief description shown in OSC docs. */
  description?: string;
}

const SOURCE_PARAMS: OscParamDescriptor[] = [
  {
    name: "sourceMode",
    label: "Source mode",
    type: "string",
    enum: ["video", "camera", "spout", "ndi", "syphon"],
    description: "Switch between file / camera / Spout / NDI / Syphon",
  },
  {
    name: "sourceFlipVertical",
    label: "Flip vertical",
    type: "bool",
    description: "Flip incoming frames vertically (Syphon-friendly)",
  },
];

const OUTPUT_PARAMS: OscParamDescriptor[] = [
  {
    name: "outputSinkEnabled",
    label: "Enabled",
    type: "bool",
    description: "Toggle the output sink on/off",
  },
  {
    name: "outputSinkType",
    label: "Sink type",
    type: "string",
    enum: ["spout", "ndi", "syphon"],
  },
];

const SLIDER_PARAMS: OscParamDescriptor[] = [
  {
    name: "value",
    label: "Value",
    type: "float",
    description: "Slider value (clamped to slider min/max)",
  },
];

const XYPAD_PARAMS: OscParamDescriptor[] = [
  { name: "padX", label: "X", type: "float" },
  { name: "padY", label: "Y", type: "float" },
];

const BOOL_PARAMS: OscParamDescriptor[] = [
  { name: "value", label: "Value", type: "bool" },
];

const TRIGGER_PARAMS: OscParamDescriptor[] = [
  {
    name: "value",
    label: "Trigger",
    type: "bool",
    description: "Send true to fire the trigger",
  },
];

const TEMPO_PARAMS: OscParamDescriptor[] = [
  {
    name: "tempoBpm",
    label: "BPM",
    type: "float",
    min: 20,
    max: 999,
    description: "Override tempo BPM (when tempo source = manual)",
  },
  { name: "tempoEnabled", label: "Tempo enabled", type: "bool" },
];

const PRIMITIVE_PARAMS: OscParamDescriptor[] = [
  {
    name: "value",
    label: "Value",
    // Type depends on `valueType` — caller can branch when emitting the
    // inventory entry. We default to string here as the most permissive.
    type: "string",
  },
];

const NOTE_PARAMS: OscParamDescriptor[] = [
  { name: "noteText", label: "Note text", type: "string" },
];

/**
 * Lookup table keyed by React Flow node `type` (matches the `nodeTypes`
 * map in GraphEditor.tsx). Nodes not in the table are not OSC-exposable
 * via this path — pipeline nodes are handled separately because their
 * params come from the loaded pipeline's schema.
 */
export const OSC_NODE_PARAMS: Record<string, OscParamDescriptor[]> = {
  source: SOURCE_PARAMS,
  output: OUTPUT_PARAMS,
  slider: SLIDER_PARAMS,
  xypad: XYPAD_PARAMS,
  bool: BOOL_PARAMS,
  trigger: TRIGGER_PARAMS,
  tempo: TEMPO_PARAMS,
  primitive: PRIMITIVE_PARAMS,
  note: NOTE_PARAMS,
};

export function getNodeOscParams(
  nodeType: string | undefined
): OscParamDescriptor[] {
  return nodeType ? (OSC_NODE_PARAMS[nodeType] ?? []) : [];
}

/**
 * Lowercase, kebab-cased slug suitable for use as the node-namespacing
 * segment of an OSC address. Falls back to the node id when the title
 * has no usable characters (e.g. all emoji).
 */
export function slugifyForOsc(
  title: string | undefined,
  nodeId: string
): string {
  const fromTitle = (title ?? "")
    .normalize("NFKD")
    // strip combining diacritics so "café" → "cafe"
    .replace(/[̀-ͯ]/g, "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .replace(/-{2,}/g, "-");
  return fromTitle || nodeId;
}

/**
 * Read the current value of a descriptor's field off `node.data`.
 * Used to seed the "default" override field in the Configure OSC modal
 * and to populate the OSC docs when the user hasn't explicitly set one.
 */
export function readNodeParamValue(
  data: FlowNodeData | undefined,
  descriptor: OscParamDescriptor
): unknown {
  if (!data) return undefined;
  return (data as Record<string, unknown>)[descriptor.name];
}

function oscTypeFromSchema(prop: PipelineSchemaProperty): OscParamType {
  const t = String(prop.type ?? "any");
  if (t === "number") return "float";
  if (t === "integer") return "integer";
  if (t === "boolean") return "bool";
  if (t === "array") {
    const items = prop.items as { type?: string } | undefined;
    if (items?.type === "integer") return "integer_list";
  }
  return "string";
}

/**
 * Convert a pipeline's JSON-Schema config into OSC param descriptors.
 *
 * Only runtime params (`ui.is_load_param === false`) are exposed — load-time
 * params don't make sense to drive over OSC.
 */
export function pickPipelineOscParams(
  schema: PipelineConfigSchema | null | undefined
): OscParamDescriptor[] {
  if (!schema) return [];
  const out: OscParamDescriptor[] = [];
  for (const [key, rawProp] of Object.entries(schema.properties)) {
    const prop = rawProp as PipelineSchemaProperty;
    const ui = prop.ui;
    if (ui?.is_load_param !== false) continue;
    out.push({
      name: key,
      label: (ui as { label?: string } | undefined)?.label ?? key,
      type: oscTypeFromSchema(prop),
      min: prop.minimum,
      max: prop.maximum,
      enum: prop.enum as string[] | undefined,
      description: prop.description ?? "",
    });
  }
  return out;
}

/**
 * Coerce a user-typed default (always a string from the modal's text input)
 * to the descriptor's declared type. Returns `undefined` for empty or
 * un-parseable input — callers treat that as "no default set".
 *
 * For `integer_list`, accepts comma- or space-separated integers
 * (e.g. `"1, 2, 3"` or `"1 2 3"`).
 */
export function coerceDefaultForType(
  raw: unknown,
  type: OscParamType
): unknown {
  if (raw === undefined || raw === null) return undefined;
  if (typeof raw !== "string") return raw;
  const trimmed = raw.trim();
  if (trimmed === "") return undefined;
  switch (type) {
    case "float": {
      const n = Number(trimmed);
      return Number.isFinite(n) ? n : undefined;
    }
    case "integer": {
      const n = Number(trimmed);
      if (!Number.isFinite(n)) return undefined;
      const i = Math.trunc(n);
      return i === n ? i : undefined;
    }
    case "bool": {
      const lower = trimmed.toLowerCase();
      if (["true", "1", "yes", "on"].includes(lower)) return true;
      if (["false", "0", "no", "off"].includes(lower)) return false;
      return undefined;
    }
    case "integer_list": {
      const parts = trimmed.split(/[\s,]+/).filter(Boolean);
      const ints: number[] = [];
      for (const p of parts) {
        const n = Number(p);
        if (!Number.isFinite(n)) return undefined;
        const i = Math.trunc(n);
        if (i !== n) return undefined;
        ints.push(i);
      }
      return ints.length > 0 ? ints : undefined;
    }
    case "string":
    default:
      return trimmed;
  }
}
