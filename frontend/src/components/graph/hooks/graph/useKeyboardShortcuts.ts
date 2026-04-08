import { useEffect, useRef, useCallback, useMemo } from "react";
import type { Edge, Node } from "@xyflow/react";
import type { FlowNodeData } from "../../../../lib/graphUtils";
import { generateNodeId } from "../../../../lib/graphUtils";
import { safeCloneData } from "../../utils/stripNodeData";
import { matchesShortcut } from "../../../../lib/shortcuts";
import { getEffectiveShortcuts } from "../../../../lib/shortcutOverrides";

interface ClipboardData {
  nodes: Node<FlowNodeData>[];
  edges: Edge[];
}

const PASTE_OFFSET = 30;

export interface KeyboardShortcutHandlers {
  [id: string]: (() => void) | undefined;
}

interface UseKeyboardShortcutsOptions {
  nodes: Node<FlowNodeData>[];
  edges: Edge[];
  setNodes: React.Dispatch<React.SetStateAction<Node<FlowNodeData>[]>>;
  setEdges: React.Dispatch<React.SetStateAction<Edge[]>>;
  isStreaming: boolean;
  handlers: KeyboardShortcutHandlers;
}

export function useKeyboardShortcuts({
  nodes,
  edges,
  setNodes,
  setEdges,
  isStreaming,
  handlers,
}: UseKeyboardShortcutsOptions) {
  const clipboardRef = useRef<ClipboardData | null>(null);
  const pasteCountRef = useRef(0);

  // Keep latest values in refs for the keydown closure
  const nodesRef = useRef(nodes);
  nodesRef.current = nodes;
  const edgesRef = useRef(edges);
  edgesRef.current = edges;
  const isStreamingRef = useRef(isStreaming);
  isStreamingRef.current = isStreaming;
  const handlersRef = useRef(handlers);
  handlersRef.current = handlers;

  // ── Internal clipboard operations ──────────────────────────────────

  const doCopy = useCallback(() => {
    const currentNodes = nodesRef.current;
    const currentEdges = edgesRef.current;

    const selectedNodes = currentNodes.filter(n => n.selected);
    if (selectedNodes.length === 0) return false;

    const selectedIds = new Set(selectedNodes.map(n => n.id));
    const interEdges = currentEdges.filter(
      edge => selectedIds.has(edge.source) && selectedIds.has(edge.target)
    );

    clipboardRef.current = {
      nodes: selectedNodes.map(n => ({
        ...n,
        data: safeCloneData(n.data),
        position: { ...n.position },
      })),
      edges: interEdges.map(e => ({ ...e })),
    };
    pasteCountRef.current = 0;
    return true;
  }, []);

  const doPaste = useCallback(() => {
    if (!clipboardRef.current || clipboardRef.current.nodes.length === 0)
      return;

    pasteCountRef.current += 1;
    const offset = PASTE_OFFSET * pasteCountRef.current;
    const clipboard = clipboardRef.current;

    const existingIds = new Set(nodesRef.current.map(n => n.id));
    const idMap = new Map<string, string>();
    for (const node of clipboard.nodes) {
      const prefix = node.data.nodeType || node.type || "node";
      const newId = generateNodeId(prefix, existingIds);
      idMap.set(node.id, newId);
      existingIds.add(newId);
    }

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

    const newEdges: Edge[] = clipboard.edges.map((edge, idx) => ({
      ...edge,
      id: `paste_${Date.now()}_${idx}`,
      source: idMap.get(edge.source) ?? edge.source,
      target: idMap.get(edge.target) ?? edge.target,
    }));

    setNodes(nds => [
      ...nds.map(n => (n.selected ? { ...n, selected: false } : n)),
      ...newNodes,
    ]);

    if (newEdges.length > 0) {
      setEdges(eds => [...eds, ...newEdges]);
    }
  }, [setNodes, setEdges]);

  // Build the full handler map including internal clipboard operations
  const resolveHandler = useCallback(
    (id: string): (() => void) | undefined => {
      // Internal handlers for clipboard operations
      switch (id) {
        case "copy":
          return () => doCopy();
        case "paste":
          return () => doPaste();
        case "cut":
          return () => {
            if (doCopy()) {
              // Delete selected nodes
              const selected = nodesRef.current.filter(n => n.selected);
              const selectedIds = new Set(selected.map(n => n.id));
              setNodes(nds => nds.filter(n => !selectedIds.has(n.id)));
              setEdges(eds =>
                eds.filter(
                  e => !selectedIds.has(e.source) && !selectedIds.has(e.target)
                )
              );
            }
          };
        case "duplicate":
          return () => {
            if (doCopy()) {
              doPaste();
            }
          };
        default:
          return handlersRef.current[id];
      }
    },
    [doCopy, doPaste, setNodes, setEdges]
  );

  // Get effective shortcuts (with user overrides)
  const shortcuts = useMemo(() => getEffectiveShortcuts(), []);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const activeElement = document.activeElement;
      const isInputElement =
        activeElement?.tagName === "INPUT" ||
        activeElement?.tagName === "TEXTAREA" ||
        activeElement?.tagName === "SELECT";

      for (const shortcut of shortcuts) {
        if (shortcut.builtIn) continue;
        if (!matchesShortcut(e, shortcut)) continue;
        if (!shortcut.allowInInput && isInputElement) continue;
        if (shortcut.disabledWhileStreaming && isStreamingRef.current) continue;

        const handler = resolveHandler(shortcut.id);
        if (handler) {
          e.preventDefault();
          handler();
          return;
        }
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [shortcuts, resolveHandler]);
}
