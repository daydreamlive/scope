import { useEffect, useRef } from "react";
import type { ReactFlowInstance, Edge, Node } from "@xyflow/react";
import type { FlowNodeData } from "../../../lib/graphUtils";
import { generateNodeId } from "../../../lib/graphUtils";

interface ClipboardData {
  nodes: Node<FlowNodeData>[];
  edges: Edge[];
}

const PASTE_OFFSET = 30;

/**
 * Deep-clone node data while stripping non-serializable values
 * (functions, MediaStream, DOM elements, etc.).  These are re-added
 * by `enrichNodes` in useGraphState after the next enrichment pass.
 */
function safeCloneData(data: FlowNodeData): FlowNodeData {
  const cleaned: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(data)) {
    if (typeof value === "function") continue;
    // Keep primitives/arrays/objects, skip browser APIs
    if (
      typeof value === "object" &&
      value !== null &&
      !Array.isArray(value) &&
      Object.getPrototypeOf(value) !== Object.prototype
    ) {
      continue;
    }
    cleaned[key] = value;
  }
  // JSON round-trip to sever references
  return JSON.parse(JSON.stringify(cleaned)) as FlowNodeData;
}

export function useKeyboardShortcuts(
  reactFlowInstanceRef: React.RefObject<ReactFlowInstance<
    Node<FlowNodeData>,
    Edge
  > | null>,
  setPendingNodePosition: (pos: { x: number; y: number }) => void,
  setShowAddNodeModal: (show: boolean) => void,
  nodes: Node<FlowNodeData>[],
  edges: Edge[],
  setNodes: React.Dispatch<React.SetStateAction<Node<FlowNodeData>[]>>,
  setEdges: React.Dispatch<React.SetStateAction<Edge[]>>,
  handleSave?: () => void
) {
  const clipboardRef = useRef<ClipboardData | null>(null);
  const pasteCountRef = useRef(0);

  // Keep latest nodes/edges in refs
  const nodesRef = useRef(nodes);
  nodesRef.current = nodes;
  const edgesRef = useRef(edges);
  edgesRef.current = edges;

  const handleSaveRef = useRef(handleSave);
  handleSaveRef.current = handleSave;

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Skip if focus is in form elements
      const activeElement = document.activeElement;
      const isInputElement =
        activeElement?.tagName === "INPUT" ||
        activeElement?.tagName === "TEXTAREA" ||
        activeElement?.tagName === "SELECT";

      // Cmd/Ctrl+S → Save
      if ((e.metaKey || e.ctrlKey) && e.key === "s") {
        e.preventDefault();
        handleSaveRef.current?.();
        return;
      }

      // Tab → open Add Node modal
      if (e.key === "Tab" && !isInputElement) {
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
        return;
      }

      // Cmd/Ctrl+C → Copy
      if ((e.metaKey || e.ctrlKey) && e.key === "c" && !isInputElement) {
        const currentNodes = nodesRef.current;
        const currentEdges = edgesRef.current;

        const selectedNodes = currentNodes.filter(n => n.selected);
        if (selectedNodes.length === 0) return;

        const selectedIds = new Set(selectedNodes.map(n => n.id));

        // Capture inter-edges
        const interEdges = currentEdges.filter(
          edge => selectedIds.has(edge.source) && selectedIds.has(edge.target)
        );

        // Deep-clone (strip MediaStream, callbacks)
        clipboardRef.current = {
          nodes: selectedNodes.map(n => ({
            ...n,
            data: safeCloneData(n.data),
            position: { ...n.position },
          })),
          edges: interEdges.map(e => ({ ...e })),
        };
        pasteCountRef.current = 0;
        return;
      }

      // Cmd/Ctrl+V → Paste
      if ((e.metaKey || e.ctrlKey) && e.key === "v" && !isInputElement) {
        if (!clipboardRef.current || clipboardRef.current.nodes.length === 0)
          return;

        e.preventDefault();

        pasteCountRef.current += 1;
        const offset = PASTE_OFFSET * pasteCountRef.current;

        const clipboard = clipboardRef.current;

        // Build existing IDs set
        const existingIds = new Set(nodesRef.current.map(n => n.id));

        // Map old → new IDs
        const idMap = new Map<string, string>();
        for (const node of clipboard.nodes) {
          const prefix = node.data.nodeType || node.type || "node";
          const newId = generateNodeId(prefix, existingIds);
          idMap.set(node.id, newId);
          existingIds.add(newId); // prevent collisions within the batch
        }

        // Create new nodes
        const newNodes: Node<FlowNodeData>[] = clipboard.nodes.map(node => ({
          ...node,
          id: idMap.get(node.id)!,
          data: safeCloneData(node.data),
          position: {
            x: (node.position?.x ?? 0) + offset,
            y: (node.position?.y ?? 0) + offset,
          },
          selected: true,
        }));

        // Create new edges
        const newEdges: Edge[] = clipboard.edges.map((edge, idx) => ({
          ...edge,
          id: `paste_${Date.now()}_${idx}`,
          source: idMap.get(edge.source) ?? edge.source,
          target: idMap.get(edge.target) ?? edge.target,
        }));

        // Deselect existing, append pasted
        setNodes(nds => [
          ...nds.map(n => (n.selected ? { ...n, selected: false } : n)),
          ...newNodes,
        ]);

        if (newEdges.length > 0) {
          setEdges(eds => [...eds, ...newEdges]);
        }

        return;
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [
    reactFlowInstanceRef,
    setPendingNodePosition,
    setShowAddNodeModal,
    setNodes,
    setEdges,
  ]);
}
