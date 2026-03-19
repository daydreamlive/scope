import { useCallback, useState } from "react";
import type { ReactFlowInstance, Edge, Node } from "@xyflow/react";
import type { FlowNodeData } from "../../../../lib/graphUtils";

export interface SelectionRect {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface ContextMenuState {
  x: number;
  y: number;
  type: "pane" | "node";
  nodeId?: string;
}

/**
 * Handles right-click interaction on the ReactFlow pane:
 * - Drag → rectangle selection
 * - Click on node → node context menu
 * - Click on pane → pane context menu
 */
export function useRightClickSelect(
  reactFlowInstanceRef: React.RefObject<ReactFlowInstance<
    Node<FlowNodeData>,
    Edge
  > | null>,
  setNodes: React.Dispatch<React.SetStateAction<Node<FlowNodeData>[]>>,
  setPendingNodePosition: (pos: { x: number; y: number } | null) => void
) {
  const [selectionRect, setSelectionRect] = useState<SelectionRect | null>(
    null
  );
  const [contextMenu, setContextMenu] = useState<ContextMenuState | null>(null);

  const handleRightMouseDown = useCallback(
    (e: React.MouseEvent) => {
      if (e.button !== 2) return; // only right-click

      const startX = e.clientX;
      const startY = e.clientY;
      const startTarget = e.target as HTMLElement;
      let isDrag = false;

      setContextMenu(null);

      const handleMove = (me: MouseEvent) => {
        const dx = me.clientX - startX;
        const dy = me.clientY - startY;
        if (!isDrag && Math.sqrt(dx * dx + dy * dy) > 5) {
          isDrag = true;
        }
        if (isDrag) {
          setSelectionRect({
            x1: startX,
            y1: startY,
            x2: me.clientX,
            y2: me.clientY,
          });
        }
      };

      const handleUp = (me: MouseEvent) => {
        window.removeEventListener("mousemove", handleMove);
        window.removeEventListener("mouseup", handleUp);

        if (isDrag) {
          const rf = reactFlowInstanceRef.current;
          if (rf) {
            const start = rf.screenToFlowPosition({ x: startX, y: startY });
            const end = rf.screenToFlowPosition({
              x: me.clientX,
              y: me.clientY,
            });

            const minX = Math.min(start.x, end.x);
            const maxX = Math.max(start.x, end.x);
            const minY = Math.min(start.y, end.y);
            const maxY = Math.max(start.y, end.y);

            setNodes(nds =>
              nds.map(n => {
                const w = n.measured?.width ?? n.width ?? 200;
                const h = n.measured?.height ?? n.height ?? 100;
                const overlaps =
                  n.position.x < maxX &&
                  n.position.x + w > minX &&
                  n.position.y < maxY &&
                  n.position.y + h > minY;
                return n.selected === overlaps
                  ? n
                  : { ...n, selected: overlaps };
              })
            );
          }
        } else {
          const nodeEl = startTarget.closest(".react-flow__node");
          if (nodeEl) {
            const nodeId = nodeEl.getAttribute("data-id");
            if (nodeId) {
              setContextMenu({
                x: startX,
                y: startY,
                type: "node",
                nodeId,
              });
            }
          } else {
            const rf = reactFlowInstanceRef.current;
            if (rf) {
              const position = rf.screenToFlowPosition({
                x: startX,
                y: startY,
              });
              setPendingNodePosition(position);
              setContextMenu({
                x: startX,
                y: startY,
                type: "pane",
              });
            }
          }
        }

        setSelectionRect(null);
      };

      window.addEventListener("mousemove", handleMove);
      window.addEventListener("mouseup", handleUp);
    },
    [reactFlowInstanceRef, setNodes, setPendingNodePosition]
  );

  return { selectionRect, contextMenu, setContextMenu, handleRightMouseDown };
}
