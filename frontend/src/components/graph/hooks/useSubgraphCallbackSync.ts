import { useCallback, useEffect, useRef } from "react";
import type { Edge, Node } from "@xyflow/react";
import type { FlowNodeData } from "../../../lib/graphUtils";
import { parseHandleId } from "../../../lib/graphUtils";
import { BOUNDARY_INPUT_ID, BOUNDARY_OUTPUT_ID } from "./useGraphNavigation";

type SetNodes = React.Dispatch<React.SetStateAction<Node<FlowNodeData>[]>>;
type SetEdges = React.Dispatch<React.SetStateAction<Edge[]>>;

/**
 * Keeps subgraph and boundary nodes in sync with the latest
 * `onEnterSubgraph` / `onPortRename` callbacks, and cleans up
 * orphaned boundary ports when their edges are removed.
 */
export function useSubgraphCallbackSync(deps: {
  nodes: Node<FlowNodeData>[];
  edges: Edge[];
  setNodes: SetNodes;
  setEdges: SetEdges;
  handleEnterSubgraph: (nodeId: string) => void;
  renameSubgraphPort: (
    side: "input" | "output",
    oldName: string,
    newName: string,
    portType: string,
    setNodes: SetNodes,
    setEdges: SetEdges
  ) => void;
  removeSubgraphPort: (
    side: "input" | "output",
    portName: string,
    setNodes: SetNodes
  ) => void;
  hasExternalConnection: (
    side: "input" | "output",
    portName: string,
    portType: string
  ) => boolean;
}) {
  const {
    nodes,
    edges,
    setNodes,
    setEdges,
    handleEnterSubgraph,
    renameSubgraphPort,
    removeSubgraphPort,
    hasExternalConnection,
  } = deps;

  // ── Stable onEnterSubgraph callback ───────────────────────────────────
  const enterSubgraphRef = useRef(handleEnterSubgraph);
  enterSubgraphRef.current = handleEnterSubgraph;

  const stableEnterSubgraph = useCallback(
    (nodeId: string) => enterSubgraphRef.current(nodeId),
    []
  );

  const hasSubgraphNeedingCallback = nodes.some(
    n =>
      n.data.nodeType === "subgraph" &&
      n.data.onEnterSubgraph !== stableEnterSubgraph
  );
  useEffect(() => {
    if (!hasSubgraphNeedingCallback) return;
    setNodes(nds =>
      nds.map(n => {
        if (n.data.nodeType !== "subgraph") return n;
        if (n.data.onEnterSubgraph === stableEnterSubgraph) return n;
        return {
          ...n,
          data: { ...n.data, onEnterSubgraph: stableEnterSubgraph },
        };
      })
    );
  }, [hasSubgraphNeedingCallback, stableEnterSubgraph, setNodes]);

  // ── Stable onPortRename callbacks ─────────────────────────────────────
  const renameInputRef = useRef(
    (oldName: string, newName: string, portType: string) =>
      renameSubgraphPort("input", oldName, newName, portType, setNodes, setEdges)
  );
  renameInputRef.current = (oldName, newName, portType) =>
    renameSubgraphPort("input", oldName, newName, portType, setNodes, setEdges);

  const renameOutputRef = useRef(
    (oldName: string, newName: string, portType: string) =>
      renameSubgraphPort(
        "output",
        oldName,
        newName,
        portType,
        setNodes,
        setEdges
      )
  );
  renameOutputRef.current = (oldName, newName, portType) =>
    renameSubgraphPort(
      "output",
      oldName,
      newName,
      portType,
      setNodes,
      setEdges
    );

  const stableRenameInput = useCallback(
    (oldName: string, newName: string, portType: string) =>
      renameInputRef.current(oldName, newName, portType),
    []
  );
  const stableRenameOutput = useCallback(
    (oldName: string, newName: string, portType: string) =>
      renameOutputRef.current(oldName, newName, portType),
    []
  );

  const hasBoundaryNeedingRename = nodes.some(
    n =>
      (n.id === BOUNDARY_INPUT_ID &&
        n.data.onPortRename !== stableRenameInput) ||
      (n.id === BOUNDARY_OUTPUT_ID &&
        n.data.onPortRename !== stableRenameOutput)
  );
  useEffect(() => {
    if (!hasBoundaryNeedingRename) return;
    setNodes(nds =>
      nds.map(n => {
        if (
          n.id === BOUNDARY_INPUT_ID &&
          n.data.onPortRename !== stableRenameInput
        ) {
          return {
            ...n,
            data: { ...n.data, onPortRename: stableRenameInput },
          };
        }
        if (
          n.id === BOUNDARY_OUTPUT_ID &&
          n.data.onPortRename !== stableRenameOutput
        ) {
          return {
            ...n,
            data: { ...n.data, onPortRename: stableRenameOutput },
          };
        }
        return n;
      })
    );
  }, [hasBoundaryNeedingRename, stableRenameInput, stableRenameOutput, setNodes]);

  // ── Orphaned boundary-port cleanup ────────────────────────────────────
  useEffect(() => {
    const currentInputHandles = new Set<string>();
    const currentOutputHandles = new Set<string>();
    for (const e of edges) {
      if (e.source === BOUNDARY_INPUT_ID) {
        const parsed = parseHandleId(e.sourceHandle);
        if (parsed && parsed.name !== "__add__")
          currentInputHandles.add(parsed.name);
      }
      if (e.target === BOUNDARY_OUTPUT_ID) {
        const parsed = parseHandleId(e.targetHandle);
        if (parsed && parsed.name !== "__add__")
          currentOutputHandles.add(parsed.name);
      }
    }

    const inputBoundary = nodes.find(n => n.id === BOUNDARY_INPUT_ID);
    const outputBoundary = nodes.find(n => n.id === BOUNDARY_OUTPUT_ID);

    if (inputBoundary) {
      const ports = inputBoundary.data.subgraphInputs ?? [];
      for (const port of ports) {
        if (
          !currentInputHandles.has(port.name) &&
          !hasExternalConnection("input", port.name, port.portType)
        ) {
          removeSubgraphPort("input", port.name, setNodes);
        }
      }
    }
    if (outputBoundary) {
      const ports = outputBoundary.data.subgraphOutputs ?? [];
      for (const port of ports) {
        if (
          !currentOutputHandles.has(port.name) &&
          !hasExternalConnection("output", port.name, port.portType)
        ) {
          removeSubgraphPort("output", port.name, setNodes);
        }
      }
    }
  }, [edges, nodes, removeSubgraphPort, hasExternalConnection, setNodes]);
}
