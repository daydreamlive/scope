/**
 * Pure serialization / deserialization helpers for subgraph nodes & edges,
 * plus boundary-node creation / stripping.
 *
 * Extracted from useGraphNavigation so they can be reused and unit-tested
 * independently.
 */

import type { Edge, Node } from "@xyflow/react";
import type {
  FlowNodeData,
  SubgraphPort,
  SerializedSubgraphNode,
  SerializedSubgraphEdge,
} from "../../../lib/graphUtils";
import { buildHandleId } from "../../../lib/graphUtils";
import { stripNonSerializable } from "./stripNodeData";

/** Sentinel node IDs for the subgraph boundary nodes. */
export const BOUNDARY_INPUT_ID = "__sg_boundary_input__";
export const BOUNDARY_OUTPUT_ID = "__sg_boundary_output__";

// ---------------------------------------------------------------------------
// Serialize
// ---------------------------------------------------------------------------

export function serializeNodes(
  nodes: Node<FlowNodeData>[]
): SerializedSubgraphNode[] {
  return nodes.map(n => {
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
      type: n.data.nodeType || n.type || "pipeline",
      position: { x: n.position.x, y: n.position.y },
      ...(w && !Number.isNaN(w) ? { width: w } : {}),
      ...(h && !Number.isNaN(h) ? { height: h } : {}),
      data: stripNonSerializable(n.data),
    };
  });
}

export function serializeEdges(edges: Edge[]): SerializedSubgraphEdge[] {
  return edges.map(e => ({
    id: e.id,
    source: e.source,
    sourceHandle: e.sourceHandle ?? null,
    target: e.target,
    targetHandle: e.targetHandle ?? null,
  }));
}

// ---------------------------------------------------------------------------
// Deserialize
// ---------------------------------------------------------------------------

export function deserializeNodes(
  serialized: SerializedSubgraphNode[]
): Node<FlowNodeData>[] {
  return serialized.map(n => {
    const sizeProps =
      n.width != null || n.height != null
        ? {
            width: n.width ?? undefined,
            height: n.height ?? undefined,
            style: {
              width: n.width ?? undefined,
              height: n.height ?? undefined,
            },
          }
        : {};
    return {
      id: n.id,
      type: n.type,
      position: { x: n.position.x, y: n.position.y },
      ...sizeProps,
      data: { ...n.data, label: n.data.label ?? n.id } as FlowNodeData,
    };
  });
}

export function deserializeEdges(serialized: SerializedSubgraphEdge[]): Edge[] {
  return serialized.map(e => ({
    id: e.id,
    source: e.source,
    sourceHandle: e.sourceHandle ?? undefined,
    target: e.target,
    targetHandle: e.targetHandle ?? undefined,
  }));
}

// ---------------------------------------------------------------------------
// Boundary helpers
// ---------------------------------------------------------------------------

function isBoundaryNode(n: Node<FlowNodeData>): boolean {
  return n.id === BOUNDARY_INPUT_ID || n.id === BOUNDARY_OUTPUT_ID;
}

function isBoundaryEdge(e: Edge): boolean {
  return e.source === BOUNDARY_INPUT_ID || e.target === BOUNDARY_OUTPUT_ID;
}

export function stripBoundary(
  nodes: Node<FlowNodeData>[],
  edges: Edge[]
): { nodes: Node<FlowNodeData>[]; edges: Edge[] } {
  return {
    nodes: nodes.filter(n => !isBoundaryNode(n)),
    edges: edges.filter(e => !isBoundaryEdge(e)),
  };
}

export function createBoundaryNodesAndEdges(
  subgraphInputs: SubgraphPort[],
  subgraphOutputs: SubgraphPort[],
  innerNodes: Node<FlowNodeData>[]
): { boundaryNodes: Node<FlowNodeData>[]; boundaryEdges: Edge[] } {
  const boundaryNodes: Node<FlowNodeData>[] = [];
  const boundaryEdges: Edge[] = [];

  const innerNodeIds = new Set(innerNodes.map(n => n.id));

  // When innerNodeId references a deeply-nested node (inside a child
  // subgraph), resolve to the child subgraph node with a matching port.
  function resolvePort(
    port: SubgraphPort,
    direction: "input" | "output"
  ): { nodeId: string; handleId: string } {
    if (innerNodeIds.has(port.innerNodeId)) {
      return { nodeId: port.innerNodeId, handleId: port.innerHandleId };
    }
    for (const n of innerNodes) {
      if (n.data.nodeType !== "subgraph") continue;
      const ports =
        direction === "input" ? n.data.subgraphInputs : n.data.subgraphOutputs;
      const matching = ports?.find(p => p.name === port.name);
      if (matching) {
        return {
          nodeId: n.id,
          handleId: buildHandleId(port.portType, port.name),
        };
      }
    }
    return { nodeId: port.innerNodeId, handleId: port.innerHandleId };
  }

  let minX = Infinity,
    maxX = -Infinity,
    minY = Infinity,
    maxY = -Infinity;
  for (const n of innerNodes) {
    const w =
      n.width ?? (typeof n.style?.width === "number" ? n.style.width : 200);
    if (n.position.x < minX) minX = n.position.x;
    if (n.position.x + (w as number) > maxX)
      maxX = n.position.x + (w as number);
    if (n.position.y < minY) minY = n.position.y;
    if (n.position.y > maxY) maxY = n.position.y;
  }
  if (!isFinite(minX)) {
    minX = 0;
    maxX = 400;
    minY = 0;
    maxY = 200;
  }
  const centerY = (minY + maxY) / 2;

  boundaryNodes.push({
    id: BOUNDARY_INPUT_ID,
    type: "subgraph_input",
    position: { x: minX - 200, y: centerY - 40 },
    deletable: false,
    data: {
      label: "Subgraph Inputs",
      nodeType: "subgraph_input",
      subgraphInputs,
    } as FlowNodeData,
  });
  for (const port of subgraphInputs) {
    const resolved = resolvePort(port, "input");
    boundaryEdges.push({
      id: `__sg_boundary_in_${port.name}`,
      source: BOUNDARY_INPUT_ID,
      sourceHandle: buildHandleId(port.portType, port.name),
      target: resolved.nodeId,
      targetHandle: resolved.handleId,
    });
  }

  boundaryNodes.push({
    id: BOUNDARY_OUTPUT_ID,
    type: "subgraph_output",
    position: { x: maxX + 80, y: centerY - 40 },
    deletable: false,
    data: {
      label: "Subgraph Outputs",
      nodeType: "subgraph_output",
      subgraphOutputs,
    } as FlowNodeData,
  });
  for (const port of subgraphOutputs) {
    const resolved = resolvePort(port, "output");
    boundaryEdges.push({
      id: `__sg_boundary_out_${port.name}`,
      source: resolved.nodeId,
      sourceHandle: resolved.handleId,
      target: BOUNDARY_OUTPUT_ID,
      targetHandle: buildHandleId(port.portType, port.name),
    });
  }

  return { boundaryNodes, boundaryEdges };
}
