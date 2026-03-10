import { useCallback } from "react";
import { useReactFlow } from "@xyflow/react";
import type { Node } from "@xyflow/react";
import type { FlowNodeData } from "../../../lib/graphUtils";

// Helper to update node data without boilerplate
export function useNodeData(nodeId: string) {
  const { setNodes } = useReactFlow<Node<FlowNodeData>>();

  const updateData = useCallback(
    (fields: Partial<FlowNodeData>) => {
      setNodes(nds =>
        nds.map(n =>
          n.id === nodeId ? { ...n, data: { ...n.data, ...fields } } : n
        )
      );
    },
    [nodeId, setNodes]
  );

  return { updateData };
}
