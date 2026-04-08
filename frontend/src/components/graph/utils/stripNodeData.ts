import type { FlowNodeData } from "../../../lib/graphUtils";

/**
 * Keys that should never be persisted into subgraph serialized data.
 * These are callbacks and runtime-only refs injected by `enrichNodes`.
 */
export const STRIP_KEYS = new Set([
  "localStream",
  "remoteStream",
  "onVideoFileUpload",
  "onSourceModeChange",
  "onSpoutSourceChange",
  "onNdiSourceChange",
  "onSyphonSourceChange",
  "onPromptChange",
  "onPipelineSelect",
  "onParameterChange",
  "onPromptSubmit",
  "pipelinePortsMap",
  "onEnterSubgraph",
  "_savedWidth",
  "_savedHeight",
  "committedValue",
]);

export interface StripOptions {
  /** Additional keys to strip beyond the default blocklist. */
  extraKeys?: Set<string>;
  /** When true, strip **only** functions and non-plain objects – ignore the
   *  default key blocklist entirely.  Useful for clipboard copy where you
   *  still want to keep pipeline-specific keys. */
  skipBlocklist?: boolean;
}

/**
 * Strip non-serializable values from FlowNodeData.
 *
 * - Removes functions
 * - Removes non-plain objects (MediaStream, DOM refs, etc.)
 * - Optionally removes keys in the default blocklist
 *
 * Returns a plain record suitable for JSON serialization / storage.
 */
export function stripNonSerializable(
  data: FlowNodeData,
  opts?: StripOptions
): Record<string, unknown> {
  const result: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(data)) {
    if (!opts?.skipBlocklist && STRIP_KEYS.has(key)) continue;
    if (opts?.extraKeys?.has(key)) continue;
    if (typeof value === "function") continue;
    if (
      typeof value === "object" &&
      value !== null &&
      !Array.isArray(value) &&
      Object.getPrototypeOf(value) !== Object.prototype
    ) {
      continue; // skip MediaStream etc.
    }
    result[key] = value;
  }
  return result;
}

/**
 * Deep-clone node data while stripping non-serializable values.
 * Uses a JSON round-trip to sever all object references.
 */
export function safeCloneData(data: FlowNodeData): FlowNodeData {
  return JSON.parse(
    JSON.stringify(stripNonSerializable(data, { skipBlocklist: true }))
  ) as FlowNodeData;
}
