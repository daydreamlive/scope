import { useCallback } from "react";
import { useStore, useNodeId, useReactFlow } from "@xyflow/react";
import type { Node } from "@xyflow/react";
import type { FlowNodeData } from "../../../lib/graphUtils";

interface NodeFlags {
  locked: boolean;
  pinned: boolean;
  selected: boolean;
}

// Read lock/pin/selected state (single selector to minimize re-renders)
export function useNodeFlags(): NodeFlags {
  const nodeId = useNodeId();

  return useStore(s => {
    if (!nodeId) return { locked: false, pinned: false, selected: false };
    const node = s.nodeLookup?.get(nodeId);
    if (!node) return { locked: false, pinned: false, selected: false };
    const data = node.data as FlowNodeData;
    return {
      locked: !!data.locked,
      pinned: !!data.pinned,
      selected: !!node.selected,
    };
  });
}

// Returns toggle callbacks for lock/pin
export function useNodeFlagToggle() {
  const nodeId = useNodeId();
  const { setNodes } = useReactFlow<Node<FlowNodeData>>();

  const toggleLock = useCallback(() => {
    if (!nodeId) return;
    setNodes(nds =>
      nds.map(n =>
        n.id === nodeId
          ? { ...n, data: { ...n.data, locked: !n.data.locked } }
          : n
      )
    );
  }, [nodeId, setNodes]);

  const togglePin = useCallback(() => {
    if (!nodeId) return;
    setNodes(nds =>
      nds.map(n => {
        if (n.id !== nodeId) return n;
        const newPinned = !n.data.pinned;
        return {
          ...n,
          draggable: !newPinned,
          data: { ...n.data, pinned: newPinned },
        };
      })
    );
  }, [nodeId, setNodes]);

  return { toggleLock, togglePin };
}
