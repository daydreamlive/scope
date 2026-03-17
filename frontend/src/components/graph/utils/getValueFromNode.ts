import type { Node } from "@xyflow/react";
import type { FlowNodeData } from "../../../lib/graphUtils";
import { parseHandleId } from "../../../lib/graphUtils";

/**
 * Extract a numeric value from a producer node given the source handle.
 * Also handles `subgraph_input` (reads from portValues) and `subgraph` nodes.
 */
export function getNumberFromNode(
  node: Node<FlowNodeData>,
  sourceHandleId?: string | null
): number | null {
  const t = node.data.nodeType;

  if (t === "primitive" || t === "reroute") {
    const val = node.data.value;
    return typeof val === "number" ? val : null;
  }
  if (t === "control" || t === "math") {
    const val = node.data.currentValue;
    return typeof val === "number" ? val : null;
  }
  if (t === "slider") {
    const val = node.data.value;
    return typeof val === "number" ? val : null;
  }
  if (t === "knobs") {
    const knobs = node.data.knobs;
    if (!knobs || !sourceHandleId) return null;
    const parsed = parseHandleId(sourceHandleId);
    if (!parsed) return null;
    const idx = parseInt(parsed.name.replace("knob_", ""), 10);
    if (isNaN(idx) || idx >= knobs.length) return null;
    return knobs[idx].value;
  }
  if (t === "xypad") {
    if (!sourceHandleId) return null;
    const parsed = parseHandleId(sourceHandleId);
    if (!parsed) return null;
    if (parsed.name === "x") return node.data.padX ?? null;
    if (parsed.name === "y") return node.data.padY ?? null;
    return null;
  }
  if (t === "midi") {
    const channels = node.data.midiChannels;
    if (!channels || !sourceHandleId) return null;
    const parsed = parseHandleId(sourceHandleId);
    if (!parsed) return null;
    const idx = parseInt(parsed.name.replace("midi_", ""), 10);
    if (isNaN(idx) || idx >= channels.length) return null;
    return channels[idx].value;
  }
  if (t === "bool" || t === "trigger") {
    const val = node.data.value;
    if (typeof val === "boolean") return val ? 1 : 0;
    return null;
  }
  // Boundary input / subgraph — read from portValues
  if (t === "subgraph_input" || t === "subgraph") {
    const pv = node.data.portValues as Record<string, unknown> | undefined;
    if (!pv || !sourceHandleId) return null;
    const parsed = parseHandleId(sourceHandleId);
    if (!parsed) return null;
    const val = pv[parsed.name];
    return typeof val === "number" ? val : null;
  }
  return null;
}

/**
 * Extract a string value from a producer node given the source handle.
 */
export function getStringFromNode(
  node: Node<FlowNodeData>,
  sourceHandleId?: string | null
): string | null {
  const t = node.data.nodeType;
  if (t === "primitive" || t === "reroute") {
    const val = node.data.value;
    return typeof val === "string" ? val : null;
  }
  if (t === "control") {
    const val = node.data.currentValue;
    return typeof val === "string" ? val : null;
  }
  if (t === "subgraph_input" || t === "subgraph") {
    const pv = node.data.portValues as Record<string, unknown> | undefined;
    if (!pv) return null;
    if (sourceHandleId) {
      const parsed = parseHandleId(sourceHandleId);
      if (parsed) {
        const val = pv[parsed.name];
        return typeof val === "string" ? val : null;
      }
    }
    for (const v of Object.values(pv)) {
      if (typeof v === "string") return v;
    }
    return null;
  }
  return null;
}

/**
 * Extract any scalar value (number, string, boolean, etc.) from a producer
 * node given its source handle.  This is the canonical "read value from node"
 * helper – all other call-sites should use this instead of inlining their own
 * node-type switch.
 */
export function getAnyValueFromNode(
  node: Node<FlowNodeData>,
  sourceHandleId?: string | null
): unknown {
  const t = node.data.nodeType;

  if (t === "primitive" || t === "reroute") return node.data.value ?? null;
  if (t === "control" || t === "math") return node.data.currentValue ?? null;
  if (t === "slider") return node.data.value ?? null;
  if (t === "bool" || t === "trigger") {
    const v = node.data.value;
    return typeof v === "boolean" ? (v ? 1 : 0) : null;
  }
  if (t === "knobs") {
    const knobs = node.data.knobs as { value: number }[] | undefined;
    if (!knobs || !sourceHandleId) return null;
    const parsed = parseHandleId(sourceHandleId);
    if (!parsed) return null;
    const idx = parseInt(parsed.name.replace("knob_", ""), 10);
    if (isNaN(idx) || idx >= knobs.length) return null;
    return knobs[idx].value;
  }
  if (t === "xypad") {
    if (!sourceHandleId) return null;
    const parsed = parseHandleId(sourceHandleId);
    if (!parsed) return null;
    if (parsed.name === "x") return node.data.padX ?? null;
    if (parsed.name === "y") return node.data.padY ?? null;
    return null;
  }
  if (t === "midi") {
    const channels = node.data.midiChannels as { value: number }[] | undefined;
    if (!channels || !sourceHandleId) return null;
    const parsed = parseHandleId(sourceHandleId);
    if (!parsed) return null;
    const idx = parseInt(parsed.name.replace("midi_", ""), 10);
    if (isNaN(idx) || idx >= channels.length) return null;
    return channels[idx].value;
  }
  if (t === "tuple") {
    if (!sourceHandleId) return null;
    const parsed = parseHandleId(sourceHandleId);
    if (!parsed) return null;
    if (parsed.name === "value") {
      return node.data.tupleValues ?? null;
    }
    const idx = parseInt(parsed.name.replace("tuple_", ""), 10);
    const vals = node.data.tupleValues as number[] | undefined;
    if (!vals || isNaN(idx) || idx >= vals.length) return null;
    return vals[idx];
  }
  if (t === "subgraph" || t === "subgraph_input") {
    const pv = node.data.portValues as Record<string, unknown> | undefined;
    if (!pv || !sourceHandleId) return null;
    const parsed = parseHandleId(sourceHandleId);
    if (!parsed) return null;
    return pv[parsed.name] ?? null;
  }
  return null;
}
