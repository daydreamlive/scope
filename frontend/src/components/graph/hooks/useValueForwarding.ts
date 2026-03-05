import { useEffect, useRef } from "react";
import type { Edge, Node } from "@xyflow/react";
import type { FlowNodeData } from "../../../lib/graphUtils";

export function useValueForwarding(
  nodes: Node<FlowNodeData>[],
  edges: Edge[],
  findConnectedPipelineParams: (
    sourceNodeId: string,
    edges: Edge[],
    nodes: Node<FlowNodeData>[]
  ) => Array<{ nodeId: string; paramName: string }>,
  resolveBackendId: (nodeId: string) => string,
  isStreamingRef: React.RefObject<boolean>,
  onNodeParamChangeRef: React.RefObject<
    ((nodeId: string, key: string, value: unknown) => void) | undefined
  >
) {
  const lastForwardTimeRef = useRef<Record<string, number>>({});

  useEffect(() => {
    if (!isStreamingRef.current || !onNodeParamChangeRef.current) return;

    const throttleMs = 100;

    for (const node of nodes) {
      if (
        node.data.nodeType !== "value" &&
        node.data.nodeType !== "control" &&
        node.data.nodeType !== "math"
      )
        continue;

      const connected = findConnectedPipelineParams(node.id, edges, nodes);
      if (connected.length === 0) continue;

      let value: unknown;
      if (node.data.nodeType === "value") {
        value = node.data.value;
      } else if (
        node.data.nodeType === "control" ||
        node.data.nodeType === "math"
      ) {
        value = node.data.currentValue;
      }

      if (value === undefined) continue;

      if (node.data.nodeType === "control" || node.data.nodeType === "math") {
        const now = Date.now();
        const lastTime = lastForwardTimeRef.current[node.id] || 0;
        if (now - lastTime < throttleMs) continue;
        lastForwardTimeRef.current[node.id] = now;
      }

      for (const { nodeId, paramName } of connected) {
        const backendId = resolveBackendId(nodeId);
        if (paramName === "__prompt") {
          onNodeParamChangeRef.current(backendId, "prompts", [
            { text: String(value), weight: 100 },
          ]);
        } else {
          onNodeParamChangeRef.current(backendId, paramName, value);
        }
      }
    }
  }, [
    nodes,
    edges,
    findConnectedPipelineParams,
    resolveBackendId,
    isStreamingRef,
    onNodeParamChangeRef,
  ]);
}

