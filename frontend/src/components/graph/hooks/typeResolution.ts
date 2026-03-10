/**
 * Shared type-resolution utilities for param-type connections.
 *
 * Extracted from useConnectionLogic.ts and useGraphState.ts to eliminate
 * duplicated `resolveSourceType`, `resolveTargetType`,
 * `resolveDownstreamType`, and `collectUpstreamChain` logic.
 */
import type { Edge, Node } from "@xyflow/react";
import { parseHandleId } from "../../../lib/graphUtils";
import type { FlowNodeData } from "../../../lib/graphUtils";

export type ResolvedType =
  | "string"
  | "number"
  | "boolean"
  | "list_number"
  | "vace"
  | undefined;

// Source type resolution
export function resolveSourceType(
  node: Node<FlowNodeData>,
  nodes: Node<FlowNodeData>[],
  edges: Edge[],
  visited = new Set<string>()
): ResolvedType {
  if (visited.has(node.id)) return undefined;
  visited.add(node.id);

  const nt = node.data.nodeType;
  if (nt === "primitive") return node.data.valueType;
  if (nt === "control") {
    return node.data.controlType === "string" ? "string" : "number";
  }
  if (nt === "math") return "number";
  if (nt === "slider" || nt === "knobs" || nt === "xypad") return "number";
  if (nt === "tuple") return "list_number";
  if (nt === "image") return "string";
  if (nt === "vace") return "vace";
  if (nt === "reroute") {
    // Walk upstream to find source
    for (const e of edges) {
      if (e.target !== node.id) continue;
      const upstream = nodes.find(n => n.id === e.source);
      if (upstream) return resolveSourceType(upstream, nodes, edges, visited);
    }
    // Fallback to stored valueType
    return node.data.valueType;
  }
  return undefined;
}

// Target type resolution
export function resolveTargetType(
  targetNode: Node<FlowNodeData>,
  targetParamName: string
): ResolvedType {
  const nt = targetNode.data.nodeType;
  if (targetParamName === "__prompt") return "string";
  if (targetParamName === "__vace") return "vace";
  if (nt === "math") return "number";
  if (nt === "slider" || nt === "knobs" || nt === "xypad") return "number";
  if (nt === "tuple") {
    if (targetParamName === "value") return "list_number";
    if (targetParamName.startsWith("row_")) return "number";
    return undefined;
  }
  if (nt === "vace") {
    if (
      targetParamName === "ref_image" ||
      targetParamName === "first_frame" ||
      targetParamName === "last_frame"
    ) {
      return "string";
    }
    return undefined;
  }
  if (nt === "reroute") return undefined; // accepts any
  if (nt === "pipeline") {
    const param = targetNode.data.parameterInputs?.find(
      p => p.name === targetParamName
    );
    return param?.type;
  }
  return undefined;
}

// ---------------------------------------------------------------------------
// Downstream type resolution
// ---------------------------------------------------------------------------

/**
 * Walk downstream from a reroute node through other reroutes until we
 * find a typed consumer. Returns the expected type or undefined.
 */
export function resolveDownstreamType(
  nodeId: string,
  nodes: Node<FlowNodeData>[],
  edges: Edge[],
  visited = new Set<string>()
): ResolvedType {
  if (visited.has(nodeId)) return undefined;
  visited.add(nodeId);

  for (const e of edges) {
    if (e.source !== nodeId) continue;
    const targetParsed = parseHandleId(e.targetHandle);
    if (!targetParsed || targetParsed.kind !== "param") continue;

    const targetNode = nodes.find(n => n.id === e.target);
    if (!targetNode) continue;

    if (targetNode.data.nodeType === "reroute") {
      const result = resolveDownstreamType(
        targetNode.id,
        nodes,
        edges,
        visited
      );
      if (result) return result;
    } else {
      const t = resolveTargetType(targetNode, targetParsed.name);
      if (t) return t;
    }
  }
  return undefined;
}

// Upstream chain collection
export function collectUpstreamChain(
  nodeId: string,
  nodes: Node<FlowNodeData>[],
  edges: Edge[],
  visited = new Set<string>()
): { rerouteIds: string[]; rootSourceId: string | null } {
  if (visited.has(nodeId)) return { rerouteIds: [], rootSourceId: null };
  visited.add(nodeId);

  const node = nodes.find(n => n.id === nodeId);
  if (!node) return { rerouteIds: [], rootSourceId: null };

  if (node.data.nodeType !== "reroute") {
    return { rerouteIds: [], rootSourceId: node.id };
  }

  const rerouteIds = [node.id];

  // Find the upstream edge feeding into this reroute
  for (const e of edges) {
    if (e.target !== nodeId) continue;
    const upstream = collectUpstreamChain(e.source, nodes, edges, visited);
    return {
      rerouteIds: [...rerouteIds, ...upstream.rerouteIds],
      rootSourceId: upstream.rootSourceId,
    };
  }

  return { rerouteIds, rootSourceId: null };
}
