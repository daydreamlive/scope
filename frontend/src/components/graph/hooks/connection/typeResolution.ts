// Type resolution utilities for param connections
import type { Edge, Node } from "@xyflow/react";
import { parseHandleId } from "../../../../lib/graphUtils";
import type { FlowNodeData } from "../../../../lib/graphUtils";

export type ResolvedType =
  | "string"
  | "number"
  | "boolean"
  | "list_number"
  | "video_path"
  | "vace"
  | undefined;

// Source types
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
  if (nt === "image") {
    return node.data.mediaType === "video" ? "video_path" : "string";
  }
  if (nt === "vace") return "vace";
  if (nt === "midi") return "number";
  if (nt === "bool") return "boolean";
  if (nt === "timeline") return "number"; // trigger outputs are number pulses (0/1)
  if (nt === "curve") return "number"; // curve node output (shape data is read directly via node data)
  if (nt === "trigger_action") {
    // Output type depends on action type
    const at = node.data.triggerActionType;
    if (at === "set_string" || at === "cycle_strings") return "string";
    if (at === "set_bool" || at === "toggle_bool") return "boolean";
    return "number"; // set_number, animate_number, default
  }
  if (nt === "reroute") {
    // Walk upstream
    for (const e of edges) {
      if (e.target !== node.id) continue;
      const upstream = nodes.find(n => n.id === e.source);
      if (upstream) return resolveSourceType(upstream, nodes, edges, visited);
    }
    // Fallback to valueType
    return node.data.valueType;
  }
  // Boundary / subgraph nodes: default to "number" so general type checks
  // don't reject them; fine-grained port-level checks happen in isValidConnection.
  if (nt === "subgraph_input" || nt === "subgraph") return "number";
  return undefined;
}

// Target types
export function resolveTargetType(
  targetNode: Node<FlowNodeData>,
  targetParamName: string
): ResolvedType {
  const nt = targetNode.data.nodeType;
  if (targetParamName === "__prompt") return "string";
  if (targetParamName === "__vace") return "vace";
  if (nt === "math") return "number";
  if (nt === "bool") return "number";
  if (
    nt === "control" &&
    targetNode.data.controlType === "string" &&
    targetNode.data.controlMode === "switch"
  ) {
    if (targetParamName.startsWith("item_")) return "number";
    if (targetParamName.startsWith("str_")) return "string";
  }
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
    if (targetParamName === "video") {
      return "video_path";
    }
    return undefined;
  }
  if (nt === "timeline") {
    if (targetParamName === "play") return "number";
    return undefined;
  }
  if (nt === "trigger_action") {
    if (targetParamName === "trigger") return "number";
    if (targetParamName === "curve") return "number";
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

// Downstream types
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

// Upstream chains
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
