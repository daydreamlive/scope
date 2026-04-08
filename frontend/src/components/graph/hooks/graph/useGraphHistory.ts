import { useRef, useCallback, useEffect } from "react";
import type { Edge, Node } from "@xyflow/react";
import type { FlowNodeData } from "../../../../lib/graphUtils";
import { safeCloneData } from "../../utils/stripNodeData";
import { enrichNodes, colorEdges } from "../../utils/nodeEnrichment";
import type { EnrichNodesDeps } from "../../utils/nodeEnrichment";

interface Snapshot {
  nodes: Node<FlowNodeData>[];
  edges: Edge[];
}

const MAX_HISTORY = 50;
const DEBOUNCE_MS = 500;

function cloneNodes(nodes: Node<FlowNodeData>[]): Node<FlowNodeData>[] {
  return nodes.map(n => ({
    ...n,
    data: safeCloneData(n.data),
    position: { ...n.position },
    measured: n.measured ? { ...n.measured } : undefined,
  }));
}

function cloneEdges(edges: Edge[]): Edge[] {
  return edges.map(e => ({ ...e }));
}

/**
 * Structural fingerprint of the graph. Captures node IDs, rounded positions,
 * and edge connections so we can detect meaningful changes while ignoring
 * sub-pixel position jitter.
 */
function computeFingerprint(
  nodes: Node<FlowNodeData>[],
  edges: Edge[]
): string {
  const ns = nodes
    .map(
      n =>
        `${n.id}|${n.type}|${Math.round(n.position.x)}|${Math.round(n.position.y)}`
    )
    .sort()
    .join(",");
  const es = edges
    .map(e => `${e.source}:${e.sourceHandle}->${e.target}:${e.targetHandle}`)
    .sort()
    .join(",");
  return `${ns}##${es}`;
}

export function useGraphHistory(
  nodes: Node<FlowNodeData>[],
  edges: Edge[],
  setNodes: React.Dispatch<React.SetStateAction<Node<FlowNodeData>[]>>,
  setEdges: React.Dispatch<React.SetStateAction<Edge[]>>,
  enrichDepsRef: React.RefObject<EnrichNodesDeps>,
  handleEdgeDelete: (edgeId: string) => void
) {
  const undoStackRef = useRef<Snapshot[]>([]);
  const redoStackRef = useRef<Snapshot[]>([]);
  const isRestoringRef = useRef(false);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const lastFingerprintRef = useRef("");
  const lastSnapshotRef = useRef<Snapshot | null>(null);
  const initializedRef = useRef(false);

  const nodesRef = useRef(nodes);
  nodesRef.current = nodes;
  const edgesRef = useRef(edges);
  edgesRef.current = edges;

  // Capture initial state once the graph has nodes
  useEffect(() => {
    if (initializedRef.current || nodes.length === 0) return;
    initializedRef.current = true;
    lastSnapshotRef.current = {
      nodes: cloneNodes(nodes),
      edges: cloneEdges(edges),
    };
    lastFingerprintRef.current = computeFingerprint(nodes, edges);
  }, [nodes, edges]);

  // Debounced change detection — push to undo stack when graph settles
  useEffect(() => {
    if (!initializedRef.current) return;
    if (isRestoringRef.current) return;

    if (debounceRef.current) clearTimeout(debounceRef.current);

    debounceRef.current = setTimeout(() => {
      const fp = computeFingerprint(nodes, edges);
      if (fp === lastFingerprintRef.current) return;

      // Push previous state to undo stack
      if (lastSnapshotRef.current) {
        undoStackRef.current.push(lastSnapshotRef.current);
        if (undoStackRef.current.length > MAX_HISTORY) {
          undoStackRef.current.shift();
        }
        redoStackRef.current = [];
      }

      lastSnapshotRef.current = {
        nodes: cloneNodes(nodes),
        edges: cloneEdges(edges),
      };
      lastFingerprintRef.current = fp;
    }, DEBOUNCE_MS);

    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, [nodes, edges]);

  const restoreSnapshot = useCallback(
    (snapshot: Snapshot) => {
      isRestoringRef.current = true;

      // Re-enrich the stripped nodes with current callbacks/streams
      let restoredNodes = snapshot.nodes.map(n => ({ ...n }));
      restoredNodes = enrichNodes(restoredNodes, enrichDepsRef.current);
      const restoredEdges = colorEdges(
        snapshot.edges.map(e => ({ ...e })),
        restoredNodes,
        handleEdgeDelete
      );

      setNodes(restoredNodes);
      setEdges(restoredEdges);

      lastFingerprintRef.current = computeFingerprint(
        snapshot.nodes,
        snapshot.edges
      );
      lastSnapshotRef.current = {
        nodes: cloneNodes(snapshot.nodes),
        edges: cloneEdges(snapshot.edges),
      };

      // Clear restoring flag after React processes the update
      requestAnimationFrame(() => {
        isRestoringRef.current = false;
      });
    },
    [setNodes, setEdges, enrichDepsRef, handleEdgeDelete]
  );

  const undo = useCallback(() => {
    if (undoStackRef.current.length === 0) return;

    // Push current state to redo stack
    redoStackRef.current.push({
      nodes: cloneNodes(nodesRef.current),
      edges: cloneEdges(edgesRef.current),
    });

    const snapshot = undoStackRef.current.pop()!;
    restoreSnapshot(snapshot);
  }, [restoreSnapshot]);

  const redo = useCallback(() => {
    if (redoStackRef.current.length === 0) return;

    // Push current state to undo stack
    undoStackRef.current.push({
      nodes: cloneNodes(nodesRef.current),
      edges: cloneEdges(edgesRef.current),
    });

    const snapshot = redoStackRef.current.pop()!;
    restoreSnapshot(snapshot);
  }, [restoreSnapshot]);

  return { undo, redo };
}
