import { useCallback, useRef, useState } from "react";
import type { Edge, Node } from "@xyflow/react";
import type { FlowNodeData, SubgraphPort } from "../../../../lib/graphUtils";
import { buildHandleId } from "../../../../lib/graphUtils";
import type { EnrichNodesDeps } from "../graph/useGraphPersistence";
import { enrichNodes, colorEdges } from "../graph/useGraphPersistence";
import { getAnyValueFromNode } from "../../utils/getValueFromNode";
import {
  serializeNodes,
  serializeEdges,
  deserializeNodes,
  deserializeEdges,
  stripBoundary,
  createBoundaryNodesAndEdges,
  BOUNDARY_INPUT_ID,
  BOUNDARY_OUTPUT_ID,
} from "../../utils/subgraphSerialization";

export interface Viewport {
  x: number;
  y: number;
  zoom: number;
}

export interface GraphLevel {
  subgraphNodeId: string;
  label: string;
  nodes: Node<FlowNodeData>[];
  edges: Edge[];
  viewport?: Viewport;
}

export interface UseGraphNavigationReturn {
  depth: number;
  breadcrumbPath: string[];
  enterSubgraph: (
    nodeId: string,
    currentNodes: Node<FlowNodeData>[],
    currentEdges: Edge[],
    setNodes: (nodes: Node<FlowNodeData>[]) => void,
    setEdges: (edges: Edge[]) => void,
    enrichDeps: EnrichNodesDeps,
    handleEdgeDelete: (edgeId: string) => void,
    currentViewport?: Viewport
  ) => Viewport | null;
  navigateTo: (
    targetDepth: number,
    currentNodes: Node<FlowNodeData>[],
    currentEdges: Edge[],
    setNodes: (nodes: Node<FlowNodeData>[]) => void,
    setEdges: (edges: Edge[]) => void,
    enrichDeps: EnrichNodesDeps,
    handleEdgeDelete: (edgeId: string) => void,
    currentViewport?: Viewport
  ) => Viewport | null;
  addSubgraphPort: (
    side: "input" | "output",
    port: SubgraphPort,
    setNodes: (
      updater: (nds: Node<FlowNodeData>[]) => Node<FlowNodeData>[]
    ) => void
  ) => string | null;
  removeSubgraphPort: (
    side: "input" | "output",
    portName: string,
    setNodes: (
      updater: (nds: Node<FlowNodeData>[]) => Node<FlowNodeData>[]
    ) => void
  ) => void;
  renameSubgraphPort: (
    side: "input" | "output",
    oldName: string,
    newName: string,
    portType: string,
    setNodes: (
      updater: (nds: Node<FlowNodeData>[]) => Node<FlowNodeData>[]
    ) => void,
    setEdges: (updater: (eds: Edge[]) => Edge[]) => void
  ) => void;
  getRootGraph: (
    currentNodes: Node<FlowNodeData>[],
    currentEdges: Edge[]
  ) => { nodes: Node<FlowNodeData>[]; edges: Edge[] };
  hasExternalConnection: (
    side: "input" | "output",
    portName: string,
    portType: string
  ) => boolean;
  resetStack: () => void;
  stackRef: { readonly current: GraphLevel[] };
}

export function useGraphNavigation(): UseGraphNavigationReturn {
  const [stack, setStack] = useState<GraphLevel[]>([]);
  const stackRef = useRef(stack);
  stackRef.current = stack;

  const subgraphViewportCache = useRef(new Map<string, Viewport>());

  const depth = stack.length;
  const breadcrumbPath = ["Root", ...stack.map(l => l.label)];

  const packCurrentIntoParent = useCallback(
    (
      currentNodes: Node<FlowNodeData>[],
      currentEdges: Edge[],
      parentNodes: Node<FlowNodeData>[],
      subgraphNodeId: string
    ): Node<FlowNodeData>[] => {
      const outputPortValues: Record<string, unknown> = {};
      const outBoundary = currentNodes.find(n => n.id === BOUNDARY_OUTPUT_ID);
      if (outBoundary) {
        const outPorts: SubgraphPort[] = outBoundary.data.subgraphOutputs ?? [];
        for (const port of outPorts) {
          if (port.portType !== "param") continue;
          const hid = buildHandleId("param", port.name);
          const edge = currentEdges.find(
            e => e.target === BOUNDARY_OUTPUT_ID && e.targetHandle === hid
          );
          if (!edge) continue;
          const srcNode = currentNodes.find(n => n.id === edge.source);
          if (!srcNode) continue;
          const val = getAnyValueFromNode(srcNode, edge.sourceHandle);
          if (val !== null && val !== undefined) {
            outputPortValues[port.name] = val;
          }
        }
      }

      const { nodes: cleanNodes, edges: cleanEdges } = stripBoundary(
        currentNodes,
        currentEdges
      );
      return parentNodes.map(n =>
        n.id !== subgraphNodeId
          ? n
          : {
              ...n,
              data: {
                ...n.data,
                subgraphNodes: serializeNodes(cleanNodes),
                subgraphEdges: serializeEdges(cleanEdges),
                portValues: {
                  ...((n.data.portValues ?? {}) as Record<string, unknown>),
                  ...outputPortValues,
                },
              },
            }
      );
    },
    []
  );

  const enterSubgraph = useCallback(
    (
      nodeId: string,
      currentNodes: Node<FlowNodeData>[],
      currentEdges: Edge[],
      setNodes: (nodes: Node<FlowNodeData>[]) => void,
      setEdges: (edges: Edge[]) => void,
      enrichDeps: EnrichNodesDeps,
      handleEdgeDelete: (edgeId: string) => void,
      currentViewport?: Viewport
    ): Viewport | null => {
      const targetNode = currentNodes.find(n => n.id === nodeId);
      if (!targetNode || targetNode.data.nodeType !== "subgraph") return null;

      setStack(prev => [
        ...prev,
        {
          subgraphNodeId: nodeId,
          label:
            targetNode.data.customTitle || targetNode.data.label || "Subgraph",
          nodes: currentNodes,
          edges: currentEdges,
          viewport: currentViewport,
        },
      ]);

      let desNodes = deserializeNodes(targetNode.data.subgraphNodes ?? []);
      let desEdges = deserializeEdges(targetNode.data.subgraphEdges ?? []);

      const { boundaryNodes, boundaryEdges } = createBoundaryNodesAndEdges(
        targetNode.data.subgraphInputs ?? [],
        targetNode.data.subgraphOutputs ?? [],
        desNodes
      );
      desNodes = [...desNodes, ...boundaryNodes];
      desEdges = [...desEdges, ...boundaryEdges];

      setNodes(enrichNodes(desNodes, enrichDeps));
      setEdges(
        colorEdges(
          desEdges,
          enrichNodes(desNodes, enrichDeps),
          handleEdgeDelete
        )
      );

      return subgraphViewportCache.current.get(nodeId) ?? null;
    },
    []
  );

  const navigateTo = useCallback(
    (
      targetDepth: number,
      currentNodes: Node<FlowNodeData>[],
      currentEdges: Edge[],
      setNodes: (nodes: Node<FlowNodeData>[]) => void,
      setEdges: (edges: Edge[]) => void,
      enrichDeps: EnrichNodesDeps,
      handleEdgeDelete: (edgeId: string) => void,
      currentViewport?: Viewport
    ): Viewport | null => {
      const currentStack = stackRef.current;
      if (targetDepth < 0 || targetDepth >= currentStack.length) return null;

      if (currentViewport && currentStack.length > 0) {
        subgraphViewportCache.current.set(
          currentStack[currentStack.length - 1].subgraphNodeId,
          currentViewport
        );
      }

      let nodes = currentNodes;
      let edges = currentEdges;
      for (let i = currentStack.length - 1; i >= targetDepth; i--) {
        const level = currentStack[i];
        nodes = packCurrentIntoParent(
          nodes,
          edges,
          level.nodes,
          level.subgraphNodeId
        );
        edges = level.edges;
      }

      const sgNodeIds = new Set(
        currentStack.slice(targetDepth).map(l => l.subgraphNodeId)
      );

      const targetViewport = currentStack[targetDepth]?.viewport ?? null;
      setStack(prev => prev.slice(0, targetDepth));

      const enriched = enrichNodes(nodes, enrichDeps).map(n =>
        sgNodeIds.has(n.id)
          ? { ...n, measured: undefined, width: undefined, height: undefined }
          : n
      );
      setNodes(enriched);
      setEdges(colorEdges(edges, enriched, handleEdgeDelete));

      return targetViewport;
    },
    [packCurrentIntoParent]
  );

  const addSubgraphPort = useCallback(
    (
      side: "input" | "output",
      port: SubgraphPort,
      setNodes: (
        updater: (nds: Node<FlowNodeData>[]) => Node<FlowNodeData>[]
      ) => void
    ): string | null => {
      const currentStack = stackRef.current;
      if (currentStack.length === 0) return null;

      const boundaryId =
        side === "input" ? BOUNDARY_INPUT_ID : BOUNDARY_OUTPUT_ID;
      const portListKey =
        side === "input" ? "subgraphInputs" : "subgraphOutputs";
      const newHandleId = buildHandleId(port.portType, port.name);

      const patchPorts = (n: Node<FlowNodeData>) => {
        const existing =
          (n.data[portListKey] as SubgraphPort[] | undefined) ?? [];
        return {
          ...n,
          data: { ...n.data, [portListKey]: [...existing, port] },
        };
      };

      setNodes(nds => nds.map(n => (n.id === boundaryId ? patchPorts(n) : n)));

      const top = currentStack[currentStack.length - 1];
      setStack(prev =>
        prev.map((level, i) =>
          i !== prev.length - 1
            ? level
            : {
                ...level,
                nodes: level.nodes.map(n =>
                  n.id === top.subgraphNodeId ? patchPorts(n) : n
                ),
              }
        )
      );

      return newHandleId;
    },
    []
  );

  const removeSubgraphPort = useCallback(
    (
      side: "input" | "output",
      portName: string,
      setNodes: (
        updater: (nds: Node<FlowNodeData>[]) => Node<FlowNodeData>[]
      ) => void
    ): void => {
      const currentStack = stackRef.current;
      if (currentStack.length === 0) return;

      const boundaryId =
        side === "input" ? BOUNDARY_INPUT_ID : BOUNDARY_OUTPUT_ID;
      const portListKey =
        side === "input" ? "subgraphInputs" : "subgraphOutputs";

      const patchPorts = (n: Node<FlowNodeData>) => {
        const existing =
          (n.data[portListKey] as SubgraphPort[] | undefined) ?? [];
        return {
          ...n,
          data: {
            ...n.data,
            [portListKey]: existing.filter(p => p.name !== portName),
          },
        };
      };

      setNodes(nds => nds.map(n => (n.id === boundaryId ? patchPorts(n) : n)));

      const top = currentStack[currentStack.length - 1];
      setStack(prev =>
        prev.map((level, i) =>
          i !== prev.length - 1
            ? level
            : {
                ...level,
                nodes: level.nodes.map(n =>
                  n.id === top.subgraphNodeId ? patchPorts(n) : n
                ),
              }
        )
      );
    },
    []
  );

  const renameSubgraphPort = useCallback(
    (
      side: "input" | "output",
      oldName: string,
      newName: string,
      portType: string,
      setNodes: (
        updater: (nds: Node<FlowNodeData>[]) => Node<FlowNodeData>[]
      ) => void,
      setEdges: (updater: (eds: Edge[]) => Edge[]) => void
    ): void => {
      const currentStack = stackRef.current;
      if (currentStack.length === 0 || oldName === newName || !newName.trim())
        return;

      const boundaryId =
        side === "input" ? BOUNDARY_INPUT_ID : BOUNDARY_OUTPUT_ID;
      const portListKey =
        side === "input" ? "subgraphInputs" : "subgraphOutputs";
      const oldHandleId = buildHandleId(
        portType as "stream" | "param",
        oldName
      );
      const newHandleId = buildHandleId(
        portType as "stream" | "param",
        newName
      );

      const patchPorts = (n: Node<FlowNodeData>) => {
        const existing =
          (n.data[portListKey] as SubgraphPort[] | undefined) ?? [];
        return {
          ...n,
          data: {
            ...n.data,
            [portListKey]: existing.map(p =>
              p.name === oldName ? { ...p, name: newName } : p
            ),
          },
        };
      };

      setNodes(nds => nds.map(n => (n.id === boundaryId ? patchPorts(n) : n)));

      setEdges(eds =>
        eds.map(e => {
          let next = e;
          if (e.source === boundaryId && e.sourceHandle === oldHandleId)
            next = {
              ...next,
              sourceHandle: newHandleId,
              id: e.id.replace(oldName, newName),
            };
          if (e.target === boundaryId && e.targetHandle === oldHandleId)
            next = {
              ...next,
              targetHandle: newHandleId,
              id: e.id.replace(oldName, newName),
            };
          return next;
        })
      );

      const top = currentStack[currentStack.length - 1];
      const sgId = top.subgraphNodeId;

      const updatedParentNodes = top.nodes.map(n =>
        n.id === sgId ? patchPorts(n) : n
      );
      const updatedParentEdges = top.edges.map(e => {
        if (
          side === "input" &&
          e.target === sgId &&
          e.targetHandle === oldHandleId
        )
          return { ...e, targetHandle: newHandleId };
        if (
          side === "output" &&
          e.source === sgId &&
          e.sourceHandle === oldHandleId
        )
          return { ...e, sourceHandle: newHandleId };
        return e;
      });

      setStack(prev =>
        prev.map((level, i) =>
          i !== prev.length - 1
            ? level
            : { ...level, nodes: updatedParentNodes, edges: updatedParentEdges }
        )
      );
    },
    []
  );

  const getRootGraph = useCallback(
    (
      currentNodes: Node<FlowNodeData>[],
      currentEdges: Edge[]
    ): { nodes: Node<FlowNodeData>[]; edges: Edge[] } => {
      const currentStack = stackRef.current;
      if (currentStack.length === 0)
        return { nodes: currentNodes, edges: currentEdges };

      let nodes = currentNodes;
      let edges = currentEdges;
      for (let i = currentStack.length - 1; i >= 0; i--) {
        const level = currentStack[i];
        nodes = packCurrentIntoParent(
          nodes,
          edges,
          level.nodes,
          level.subgraphNodeId
        );
        edges = level.edges;
      }
      return { nodes, edges };
    },
    [packCurrentIntoParent]
  );

  const hasExternalConnection = useCallback(
    (side: "input" | "output", portName: string, portType: string): boolean => {
      const currentStack = stackRef.current;
      if (currentStack.length === 0) return false;

      const top = currentStack[currentStack.length - 1];
      const sgId = top.subgraphNodeId;
      const handleId = buildHandleId(portType as "stream" | "param", portName);

      return top.edges.some(e => {
        if (side === "input")
          return e.target === sgId && e.targetHandle === handleId;
        return e.source === sgId && e.sourceHandle === handleId;
      });
    },
    []
  );

  const resetStack = useCallback(() => setStack([]), []);

  return {
    depth,
    breadcrumbPath,
    enterSubgraph,
    navigateTo,
    addSubgraphPort,
    removeSubgraphPort,
    renameSubgraphPort,
    hasExternalConnection,
    getRootGraph,
    resetStack,
    stackRef,
  };
}
