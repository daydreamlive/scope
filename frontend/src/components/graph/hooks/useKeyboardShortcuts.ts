import { useEffect } from "react";
import type { ReactFlowInstance, Edge, Node } from "@xyflow/react";
import type { FlowNodeData } from "../../../lib/graphUtils";

export function useKeyboardShortcuts(
  reactFlowInstanceRef: React.RefObject<ReactFlowInstance<
    Node<FlowNodeData>,
    Edge
  > | null>,
  setPendingNodePosition: (pos: { x: number; y: number }) => void,
  setShowAddNodeModal: (show: boolean) => void
) {
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key !== "Tab") return;

      const activeElement = document.activeElement;
      const isInputElement =
        activeElement?.tagName === "INPUT" ||
        activeElement?.tagName === "TEXTAREA" ||
        activeElement?.tagName === "SELECT";

      if (isInputElement) return;

      e.preventDefault();

      if (!reactFlowInstanceRef.current) return;

      const centerX = window.innerWidth / 2;
      const centerY = window.innerHeight / 2;

      const flowPosition = reactFlowInstanceRef.current.screenToFlowPosition({
        x: centerX,
        y: centerY,
      });

      setPendingNodePosition(flowPosition);
      setShowAddNodeModal(true);
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [reactFlowInstanceRef, setPendingNodePosition, setShowAddNodeModal]);
}
