import { useEdges, useNodes } from "@xyflow/react";
import type { Node } from "@xyflow/react";
import type { FlowNodeData } from "../../../../lib/graphUtils";
import { buildHandleId } from "../../../../lib/graphUtils";
import { getNumberFromNode } from "../../utils/getValueFromNode";

interface ConnectedNumber {
  value: number;
  connected: boolean;
}

/** Numeric param value when connected, else fallback. */
export function useConnectedNumber(
  nodeId: string,
  paramName: string,
  fallback: number
): ConnectedNumber {
  const edges = useEdges();
  const allNodes = useNodes() as Node<FlowNodeData>[];

  const handleId = buildHandleId("param", paramName);
  const edge = edges.find(
    e => e.target === nodeId && e.targetHandle === handleId
  );
  if (!edge) return { value: fallback, connected: false };

  const sourceNode = allNodes.find(n => n.id === edge.source);
  if (!sourceNode) return { value: fallback, connected: false };

  const v = getNumberFromNode(sourceNode, edge.sourceHandle);
  return v !== null
    ? { value: v, connected: true }
    : { value: fallback, connected: false };
}
