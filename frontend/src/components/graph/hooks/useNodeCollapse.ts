import { useCallback } from "react";
import { useStore, useNodeId, useReactFlow } from "@xyflow/react";
import type { Node } from "@xyflow/react";
import type { FlowNodeData } from "../../../lib/graphUtils";

/**
 * Read collapsed state and provide a toggle callback.
 *
 * When collapsing, the current node dimensions are saved into
 * `_savedWidth` / `_savedHeight` and then cleared so React Flow
 * auto-measures the (smaller) collapsed DOM.  When expanding,
 * the saved dimensions are restored.
 */
export function useNodeCollapse() {
  const nodeId = useNodeId();

  const collapsed = useStore(s => {
    if (!nodeId) return false;
    const node = s.nodeLookup?.get(nodeId);
    if (!node) return false;
    return !!(node.data as FlowNodeData).collapsed;
  });

  const { setNodes } = useReactFlow<Node<FlowNodeData>>();

  const toggleCollapse = useCallback(() => {
    if (!nodeId) return;
    setNodes(nds =>
      nds.map(n => {
        if (n.id !== nodeId) return n;
        const willCollapse = !n.data.collapsed;

        if (willCollapse) {
          // Save current dimensions before collapsing
          const currentW =
            n.width ??
            n.measured?.width ??
            (typeof n.style?.width === "number"
              ? (n.style.width as number)
              : undefined);
          const currentH =
            n.height ??
            n.measured?.height ??
            (typeof n.style?.height === "number"
              ? (n.style.height as number)
              : undefined);

          return {
            ...n,
            data: {
              ...n.data,
              collapsed: true,
              _savedWidth: currentW,
              _savedHeight: currentH,
            },
            width: undefined,
            height: undefined,
            style: {},
          };
        }

        // Expanding — restore saved dimensions
        const savedW = n.data._savedWidth as number | undefined;
        const savedH = n.data._savedHeight as number | undefined;

        return {
          ...n,
          data: {
            ...n.data,
            collapsed: false,
            _savedWidth: undefined,
            _savedHeight: undefined,
          },
          width: savedW,
          height: savedH,
          style: {
            width: savedW,
            height: savedH,
          },
        };
      })
    );
  }, [nodeId, setNodes]);

  return { collapsed, toggleCollapse };
}
