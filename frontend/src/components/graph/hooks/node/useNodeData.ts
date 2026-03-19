import { useCallback } from "react";
import { useReactFlow } from "@xyflow/react";
import type { Node } from "@xyflow/react";
import type { FlowNodeData } from "../../../../lib/graphUtils";

// Updates node data, bailing out if no fields changed
export function useNodeData(nodeId: string) {
  const { setNodes } = useReactFlow<Node<FlowNodeData>>();

  const updateData = useCallback(
    (fields: Partial<FlowNodeData>) => {
      setNodes(nds => {
        let anyChanged = false;
        const result = nds.map(n => {
          if (n.id !== nodeId) return n;
          let changed = false;
          for (const [key, val] of Object.entries(fields)) {
            if (!Object.is((n.data as Record<string, unknown>)[key], val)) {
              changed = true;
              break;
            }
          }
          if (!changed) return n;
          anyChanged = true;
          return { ...n, data: { ...n.data, ...fields } };
        });
        return anyChanged ? result : nds;
      });
    },
    [nodeId, setNodes]
  );

  return { updateData };
}
