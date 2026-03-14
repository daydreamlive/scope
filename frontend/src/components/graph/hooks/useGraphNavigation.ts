import { useCallback, useRef, useState } from "react";
import type { Edge, Node } from "@xyflow/react";
import type {
  FlowNodeData,
  SubgraphPort,
  SerializedSubgraphNode,
  SerializedSubgraphEdge,
} from "../../../lib/graphUtils";
import { buildHandleId, parseHandleId } from "../../../lib/graphUtils";
import type { EnrichNodesDeps } from "./useGraphPersistence";
import { enrichNodes, colorEdges } from "./useGraphPersistence";

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

export const BOUNDARY_INPUT_ID = "__sg_boundary_input__";
export const BOUNDARY_OUTPUT_ID = "__sg_boundary_output__";

const STRIP_KEYS = new Set([
  "localStream",
  "remoteStream",
  "onVideoFileUpload",
  "onSourceModeChange",
  "onSpoutSourceChange",
  "onNdiSourceChange",
  "onSyphonSourceChange",
  "onPromptChange",
  "onPipelineSelect",
  "onParameterChange",
  "onPromptSubmit",
  "pipelinePortsMap",
  "onEnterSubgraph",
  "_savedWidth",
  "_savedHeight",
]);

function stripNonSerializable(data: FlowNodeData): Record<string, unknown> {
  const result: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(data)) {
    if (STRIP_KEYS.has(key)) continue;
    if (typeof value === "function") continue;
    if (
      typeof value === "object" &&
      value !== null &&
      !Array.isArray(value) &&
      Object.getPrototypeOf(value) !== Object.prototype
    )
      continue; // skip MediaStream etc.
    result[key] = value;
  }
  return result;
}

function serializeNodes(nodes: Node<FlowNodeData>[]): SerializedSubgraphNode[] {
  return nodes.map(n => {
    const w =
      n.width ??
      n.measured?.width ??
      (typeof n.style?.width === "number" ? n.style.width : undefined);
    const h =
      n.height ??
      n.measured?.height ??
      (typeof n.style?.height === "number" ? n.style.height : undefined);
    return {
      id: n.id,
      type: n.data.nodeType || n.type || "pipeline",
      position: { x: n.position.x, y: n.position.y },
      ...(w && !Number.isNaN(w) ? { width: w } : {}),
      ...(h && !Number.isNaN(h) ? { height: h } : {}),
      data: stripNonSerializable(n.data),
    };
  });
}

function serializeEdges(edges: Edge[]): SerializedSubgraphEdge[] {
  return edges.map(e => ({
    id: e.id,
    source: e.source,
    sourceHandle: e.sourceHandle ?? null,
    target: e.target,
    targetHandle: e.targetHandle ?? null,
  }));
}

function deserializeNodes(
  serialized: SerializedSubgraphNode[]
): Node<FlowNodeData>[] {
  return serialized.map(n => {
    const sizeProps =
      n.width != null || n.height != null
        ? {
            width: n.width ?? undefined,
            height: n.height ?? undefined,
            style: {
              width: n.width ?? undefined,
              height: n.height ?? undefined,
            },
          }
        : {};
    return {
      id: n.id,
      type: n.type,
      position: { x: n.position.x, y: n.position.y },
      ...sizeProps,
      data: { ...n.data, label: n.data.label ?? n.id } as FlowNodeData,
    };
  });
}

function deserializeEdges(serialized: SerializedSubgraphEdge[]): Edge[] {
  return serialized.map(e => ({
    id: e.id,
    source: e.source,
    sourceHandle: e.sourceHandle ?? undefined,
    target: e.target,
    targetHandle: e.targetHandle ?? undefined,
  }));
}

function isBoundaryNode(n: Node<FlowNodeData>): boolean {
  return n.id === BOUNDARY_INPUT_ID || n.id === BOUNDARY_OUTPUT_ID;
}

function isBoundaryEdge(e: Edge): boolean {
  return e.source === BOUNDARY_INPUT_ID || e.target === BOUNDARY_OUTPUT_ID;
}

function stripBoundary(
  nodes: Node<FlowNodeData>[],
  edges: Edge[]
): { nodes: Node<FlowNodeData>[]; edges: Edge[] } {
  return {
    nodes: nodes.filter(n => !isBoundaryNode(n)),
    edges: edges.filter(e => !isBoundaryEdge(e)),
  };
}

function createBoundaryNodesAndEdges(
  subgraphInputs: SubgraphPort[],
  subgraphOutputs: SubgraphPort[],
  innerNodes: Node<FlowNodeData>[]
): { boundaryNodes: Node<FlowNodeData>[]; boundaryEdges: Edge[] } {
  const boundaryNodes: Node<FlowNodeData>[] = [];
  const boundaryEdges: Edge[] = [];

  let minX = Infinity,
    maxX = -Infinity,
    minY = Infinity,
    maxY = -Infinity;
  for (const n of innerNodes) {
    const w =
      n.width ?? (typeof n.style?.width === "number" ? n.style.width : 200);
    if (n.position.x < minX) minX = n.position.x;
    if (n.position.x + (w as number) > maxX)
      maxX = n.position.x + (w as number);
    if (n.position.y < minY) minY = n.position.y;
    if (n.position.y > maxY) maxY = n.position.y;
  }
  if (!isFinite(minX)) {
    minX = 0;
    maxX = 400;
    minY = 0;
    maxY = 200;
  }
  const centerY = (minY + maxY) / 2;

  boundaryNodes.push({
    id: BOUNDARY_INPUT_ID,
    type: "subgraph_input",
    position: { x: minX - 200, y: centerY - 40 },
    deletable: false,
    data: {
      label: "Subgraph Inputs",
      nodeType: "subgraph_input",
      subgraphInputs,
    } as FlowNodeData,
  });
  for (const port of subgraphInputs) {
    boundaryEdges.push({
      id: `__sg_boundary_in_${port.name}`,
      source: BOUNDARY_INPUT_ID,
      sourceHandle: buildHandleId(port.portType, port.name),
      target: port.innerNodeId,
      targetHandle: port.innerHandleId,
    });
  }

  boundaryNodes.push({
    id: BOUNDARY_OUTPUT_ID,
    type: "subgraph_output",
    position: { x: maxX + 80, y: centerY - 40 },
    deletable: false,
    data: {
      label: "Subgraph Outputs",
      nodeType: "subgraph_output",
      subgraphOutputs,
    } as FlowNodeData,
  });
  for (const port of subgraphOutputs) {
    boundaryEdges.push({
      id: `__sg_boundary_out_${port.name}`,
      source: port.innerNodeId,
      sourceHandle: port.innerHandleId,
      target: BOUNDARY_OUTPUT_ID,
      targetHandle: buildHandleId(port.portType, port.name),
    });
  }

  return { boundaryNodes, boundaryEdges };
}

function readValueFromNode(
  node: Node<FlowNodeData>,
  sourceHandle?: string | null
): unknown {
  const t = node.data.nodeType;
  if (t === "primitive" || t === "reroute") return node.data.value ?? null;
  if (t === "control" || t === "math") return node.data.currentValue ?? null;
  if (t === "slider") return node.data.value ?? null;
  if (t === "bool") {
    const v = node.data.value;
    return typeof v === "boolean" ? (v ? 1 : 0) : null;
  }
  if (t === "knobs") {
    const knobs = node.data.knobs as { value: number }[] | undefined;
    if (!knobs || !sourceHandle) return null;
    const parsed = parseHandleId(sourceHandle);
    if (!parsed) return null;
    const idx = parseInt(parsed.name.replace("knob_", ""), 10);
    if (isNaN(idx) || idx >= knobs.length) return null;
    return knobs[idx].value;
  }
  if (t === "xypad") {
    if (!sourceHandle) return null;
    const parsed = parseHandleId(sourceHandle);
    if (!parsed) return null;
    if (parsed.name === "x") return node.data.padX ?? null;
    if (parsed.name === "y") return node.data.padY ?? null;
    return null;
  }
  if (t === "midi") {
    const channels = node.data.midiChannels as { value: number }[] | undefined;
    if (!channels || !sourceHandle) return null;
    const parsed = parseHandleId(sourceHandle);
    if (!parsed) return null;
    const idx = parseInt(parsed.name.replace("midi_", ""), 10);
    if (isNaN(idx) || idx >= channels.length) return null;
    return channels[idx].value;
  }
  if (t === "subgraph_input" || t === "subgraph") {
    const pv = node.data.portValues as Record<string, unknown> | undefined;
    if (!pv || !sourceHandle) return null;
    const parsed = parseHandleId(sourceHandle);
    if (!parsed) return null;
    return pv[parsed.name] ?? null;
  }
  return null;
}

export interface UseGraphNavigationReturn {
  depth: number;
  breadcrumbPath: string[];
  isInsideSubgraph: boolean;
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
  exitSubgraph: (
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
  const isInsideSubgraph = depth > 0;
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
          const val = readValueFromNode(srcNode, edge.sourceHandle);
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

  const exitSubgraph = useCallback(
    (
      currentNodes: Node<FlowNodeData>[],
      currentEdges: Edge[],
      setNodes: (nodes: Node<FlowNodeData>[]) => void,
      setEdges: (edges: Edge[]) => void,
      enrichDeps: EnrichNodesDeps,
      handleEdgeDelete: (edgeId: string) => void,
      currentViewport?: Viewport
    ): Viewport | null => {
      const currentStack = stackRef.current;
      if (currentStack.length === 0) return null;

      const top = currentStack[currentStack.length - 1];
      if (currentViewport)
        subgraphViewportCache.current.set(top.subgraphNodeId, currentViewport);

      const updatedParent = packCurrentIntoParent(
        currentNodes,
        currentEdges,
        top.nodes,
        top.subgraphNodeId
      );
      setStack(prev => prev.slice(0, -1));

      const enriched = enrichNodes(updatedParent, enrichDeps).map(n =>
        n.id === top.subgraphNodeId
          ? { ...n, measured: undefined, width: undefined, height: undefined }
          : n
      );
      setNodes(enriched);
      setEdges(colorEdges(top.edges, enriched, handleEdgeDelete));

      return top.viewport ?? null;
    },
    [packCurrentIntoParent]
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
    exitSubgraph,
    isInsideSubgraph,
    addSubgraphPort,
    removeSubgraphPort,
    renameSubgraphPort,
    hasExternalConnection,
    getRootGraph,
    resetStack,
    stackRef,
  };
}
