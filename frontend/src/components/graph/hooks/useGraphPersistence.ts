import { useCallback, useEffect, useRef, useState } from "react";
import type { Edge, Node } from "@xyflow/react";
import {
  graphConfigToFlow,
  flowToGraphConfig,
  extractParameterPorts,
} from "../../../lib/graphUtils";
import type { FlowNodeData } from "../../../lib/graphUtils";
import type { PipelineSchemaInfo } from "../../../lib/api";
import { buildEdgeStyle } from "../constants";

// localStorage helpers
const LS_GRAPH_KEY = "scope:graph:backup";

function saveGraphToLocalStorage(graphJson: string): void {
  try {
    localStorage.setItem(LS_GRAPH_KEY, graphJson);
  } catch {
    // Storage full or unavailable – silently ignore
  }
}

function loadGraphFromLocalStorage(): unknown | null {
  try {
    const raw = localStorage.getItem(LS_GRAPH_KEY);
    if (!raw) return null;
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

function clearGraphFromLocalStorage(): void {
  try {
    localStorage.removeItem(LS_GRAPH_KEY);
  } catch {
    // ignore
  }
}

// Node-param attachment helpers

/** Attach pipeline node parameter values into ui_state for persistence. */
export function attachNodeParams(
  config: ReturnType<typeof flowToGraphConfig>,
  params: Record<string, Record<string, unknown>>
): ReturnType<typeof flowToGraphConfig> {
  const filtered: Record<string, Record<string, unknown>> = {};
  for (const [nodeId, bag] of Object.entries(params)) {
    if (bag && Object.keys(bag).length > 0) {
      filtered[nodeId] = bag;
    }
  }
  if (Object.keys(filtered).length === 0) return config;
  return {
    ...config,
    ui_state: {
      ...(config.ui_state ?? {}),
      node_params: filtered,
    },
  };
}

/** Extract pipeline node parameter values from ui_state. */
export function extractNodeParams(
  uiState: Record<string, unknown> | null | undefined
): Record<string, Record<string, unknown>> {
  if (!uiState || typeof uiState !== "object") return {};
  const raw = (uiState as Record<string, unknown>).node_params;
  if (!raw || typeof raw !== "object") return {};
  return raw as Record<string, Record<string, unknown>>;
}

// Enrichment helpers

export interface EnrichNodesDeps {
  availablePipelineIds: string[];
  portsMap: Record<string, { inputs: string[]; outputs: string[] }>;
  pipelineSchemas: Record<string, PipelineSchemaInfo>;
  handlePipelineSelect: (nodeId: string, newPipelineId: string | null) => void;
  handleNodeParameterChange: (
    nodeId: string,
    key: string,
    value: unknown
  ) => void;
  handlePromptChange: (nodeId: string, text: string) => void;
  handlePromptSubmit: (nodeId: string) => void;
  nodeParamsRef: React.RefObject<Record<string, Record<string, unknown>>>;
  localStream?: MediaStream | null;
  remoteStream?: MediaStream | null;
  onVideoFileUploadRef: React.RefObject<
    ((file: File) => Promise<boolean>) | undefined
  >;
  onSourceModeChangeRef: React.RefObject<((mode: string) => void) | undefined>;
  onSpoutSourceChangeRef: React.RefObject<((name: string) => void) | undefined>;
  onNdiSourceChangeRef: React.RefObject<
    ((identifier: string) => void) | undefined
  >;
  onSyphonSourceChangeRef: React.RefObject<
    ((identifier: string) => void) | undefined
  >;
  spoutAvailable: boolean;
  ndiAvailable: boolean;
  syphonAvailable: boolean;
  spoutOutputAvailable: boolean;
  ndiOutputAvailable: boolean;
  syphonOutputAvailable: boolean;
  handleEdgeDelete: (edgeId: string) => void;
}

export function enrichNodes(
  flowNodes: Node<FlowNodeData>[],
  deps: EnrichNodesDeps
): Node<FlowNodeData>[] {
  return flowNodes.map(n => {
    if (n.data.nodeType === "pipeline") {
      const pipelineId = n.data.pipelineId;
      const schema = pipelineId ? deps.pipelineSchemas[pipelineId] : null;
      const parameterInputs = schema ? extractParameterPorts(schema) : [];
      const supportsPrompts = schema?.supports_prompts ?? false;
      const supportsCacheManagement =
        schema?.supports_cache_management ?? false;
      const supportsVace = schema?.supports_vace ?? false;
      const nodeParamValues = deps.nodeParamsRef.current?.[n.id] || {};
      const pipelineAvailable = pipelineId
        ? deps.availablePipelineIds.includes(pipelineId)
        : true;
      const ports =
        pipelineId && deps.portsMap ? deps.portsMap[pipelineId] : null;
      return {
        ...n,
        data: {
          ...n.data,
          availablePipelineIds: deps.availablePipelineIds,
          pipelinePortsMap: deps.portsMap,
          onPipelineSelect: deps.handlePipelineSelect,
          parameterInputs,
          parameterValues: nodeParamValues,
          onParameterChange: deps.handleNodeParameterChange,
          supportsPrompts,
          supportsCacheManagement,
          supportsVace,
          promptText: (nodeParamValues.__prompt as string) || "",
          onPromptChange: deps.handlePromptChange,
          onPromptSubmit: deps.handlePromptSubmit,
          pipelineAvailable,
          ...(ports
            ? {
                streamInputs: ports.inputs,
                streamOutputs: ports.outputs,
              }
            : {}),
        },
      };
    }
    if (n.data.nodeType === "source") {
      return {
        ...n,
        data: {
          ...n.data,
          localStream: deps.localStream,
          onVideoFileUpload: deps.onVideoFileUploadRef.current,
          onSourceModeChange: deps.onSourceModeChangeRef.current,
          spoutAvailable: deps.spoutAvailable,
          ndiAvailable: deps.ndiAvailable,
          syphonAvailable: deps.syphonAvailable,
          onSpoutSourceChange: deps.onSpoutSourceChangeRef.current,
          onNdiSourceChange: deps.onNdiSourceChangeRef.current,
          onSyphonSourceChange: deps.onSyphonSourceChangeRef.current,
        },
      };
    }
    if (n.data.nodeType === "sink") {
      return { ...n, data: { ...n.data, remoteStream: deps.remoteStream } };
    }
    if (n.data.nodeType === "output") {
      return {
        ...n,
        data: {
          ...n.data,
          spoutAvailable: deps.spoutOutputAvailable,
          ndiAvailable: deps.ndiOutputAvailable,
          syphonAvailable: deps.syphonOutputAvailable,
        },
      };
    }
    return n;
  });
}

export function colorEdges(
  flowEdges: Edge[],
  enrichedNodes: Node<FlowNodeData>[],
  handleEdgeDelete: (edgeId: string) => void
): Edge[] {
  return flowEdges.map(edge => {
    const sourceNode = enrichedNodes.find(n => n.id === edge.source);
    const style = buildEdgeStyle(sourceNode, edge.sourceHandle);
    return {
      ...edge,
      type: "default",
      reconnectable: "target" as const,
      style,
      animated: false,
      data: { onDelete: handleEdgeDelete },
    };
  });
}

interface UseGraphPersistenceArgs {
  nodes: Node<FlowNodeData>[];
  edges: Edge[];
  setNodes: React.Dispatch<React.SetStateAction<Node<FlowNodeData>[]>>;
  setEdges: React.Dispatch<React.SetStateAction<Edge[]>>;
  portsMap: Record<string, { inputs: string[]; outputs: string[] }>;
  nodeParamsRef: React.RefObject<Record<string, Record<string, unknown>>>;
  setNodeParams: React.Dispatch<
    React.SetStateAction<Record<string, Record<string, unknown>>>
  >;
  enrichDepsRef: React.RefObject<EnrichNodesDeps>;
  handleEdgeDelete: (edgeId: string) => void;
  onGraphChange?: () => void;
  onGraphClear?: () => void;
}

export function useGraphPersistence({
  nodes,
  edges,
  setNodes,
  setEdges,
  portsMap,
  nodeParamsRef,
  setNodeParams,
  enrichDepsRef,
  handleEdgeDelete,
  onGraphChange,
  onGraphClear,
}: UseGraphPersistenceArgs) {
  const [status, setStatus] = useState<string>("");
  const [fitViewTrigger, setFitViewTrigger] = useState(0);

  const nodesRef = useRef(nodes);
  nodesRef.current = nodes;
  const edgesRef = useRef(edges);
  edgesRef.current = edges;

  const onGraphChangeRef = useRef(onGraphChange);
  onGraphChangeRef.current = onGraphChange;

  const initialLoadDone = useRef(false);

  // Load graph from localStorage

  const loadGraph = useCallback(() => {
    if (Object.keys(portsMap).length === 0) return;

    const backup = loadGraphFromLocalStorage();
    if (
      backup &&
      typeof backup === "object" &&
      backup !== null &&
      "nodes" in backup &&
      "edges" in backup
    ) {
      try {
        const graphConfig = backup as Parameters<typeof graphConfigToFlow>[0];
        const { nodes: flowNodes, edges: flowEdges } = graphConfigToFlow(
          graphConfig,
          portsMap
        );
        const restoredParams = extractNodeParams(
          (backup as Record<string, unknown>).ui_state as
            | Record<string, unknown>
            | null
            | undefined
        );
        setNodeParams(restoredParams);
        const enriched = enrichNodes(flowNodes, enrichDepsRef.current);
        setNodes(enriched);
        setEdges(colorEdges(flowEdges, enriched, handleEdgeDelete));
        setStatus("Restored from local storage");
      } catch {
        setStatus("No graph configured");
      }
    } else {
      setStatus("No graph configured");
    }
    initialLoadDone.current = true;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [portsMap]);

  // Load graph on mount
  useEffect(() => {
    loadGraph();
  }, [loadGraph]);

  // Notify parent
  useEffect(() => {
    if (!initialLoadDone.current) return;
    if (nodes.length === 0 && edges.length === 0) return;
    onGraphChangeRef.current?.();
  }, [nodes, edges]);

  // Auto-save graph to localStorage (debounced)
  useEffect(() => {
    if (!initialLoadDone.current) return;
    if (nodes.length === 0 && edges.length === 0) return;

    const timer = setTimeout(() => {
      try {
        const graphConfig = attachNodeParams(
          flowToGraphConfig(nodes, edges),
          nodeParamsRef.current
        );
        const graphJson = JSON.stringify(graphConfig);
        saveGraphToLocalStorage(graphJson);
        setStatus(
          `Saved: ${graphConfig.nodes.length} nodes, ${graphConfig.edges.length} edges`
        );
      } catch (err) {
        const message = err instanceof Error ? err.message : "Unknown error";
        setStatus(`Save failed: ${message}`);
      }
    }, 500);

    return () => clearTimeout(timer);
  }, [nodes, edges, nodeParamsRef]);

  // Manual save (immediate, no debounce)
  const handleSave = useCallback(() => {
    if (nodes.length === 0 && edges.length === 0) {
      setStatus("Nothing to save");
      return;
    }
    try {
      const graphConfig = attachNodeParams(
        flowToGraphConfig(nodes, edges),
        nodeParamsRef.current
      );
      const graphJson = JSON.stringify(graphConfig);
      saveGraphToLocalStorage(graphJson);
      setStatus(
        `Saved: ${graphConfig.nodes.length} nodes, ${graphConfig.edges.length} edges`
      );
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unknown error";
      setStatus(`Save failed: ${message}`);
    }
  }, [nodes, edges, nodeParamsRef]);

  // beforeunload: sync-save to localStorage
  useEffect(() => {
    const handler = () => {
      try {
        const currentNodes = nodesRef.current;
        const currentEdges = edgesRef.current;
        if (currentNodes.length > 0 || currentEdges.length > 0) {
          const graphConfig = attachNodeParams(
            flowToGraphConfig(currentNodes, currentEdges),
            nodeParamsRef.current
          );
          saveGraphToLocalStorage(JSON.stringify(graphConfig));
        }
      } catch {
        // best effort
      }
    };

    window.addEventListener("beforeunload", handler);
    return () => window.removeEventListener("beforeunload", handler);
  }, [nodeParamsRef]);

  // Clear graph
  const handleClear = useCallback(() => {
    clearGraphFromLocalStorage();
    setNodes([]);
    setEdges([]);
    setStatus("Graph cleared");
    onGraphClear?.();
  }, [setNodes, setEdges, onGraphClear]);

  // Import graph from JSON file
  const handleImport = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      if (!file) return;

      const reader = new FileReader();
      reader.onload = e => {
        try {
          const graphConfig = JSON.parse(e.target?.result as string);
          if (!graphConfig.nodes || !graphConfig.edges) {
            setStatus("Import failed: invalid graph format");
            return;
          }
          const { nodes: flowNodes, edges: flowEdges } = graphConfigToFlow(
            graphConfig,
            portsMap
          );
          const restoredParams = extractNodeParams(graphConfig.ui_state);
          setNodeParams(restoredParams);
          const enriched = enrichNodes(flowNodes, enrichDepsRef.current);
          setNodes(enriched);
          setEdges(colorEdges(flowEdges, enriched, handleEdgeDelete));
          setStatus(`Imported from ${file.name}`);
          setFitViewTrigger(c => c + 1);
        } catch {
          setStatus("Import failed: invalid JSON");
        }
      };
      reader.readAsText(file);
      event.target.value = "";
    },

    [
      portsMap,
      setNodes,
      setEdges,
      handleEdgeDelete,
      setNodeParams,
      enrichDepsRef,
    ]
  );

  // Export graph as JSON file
  const handleExport = useCallback(() => {
    const graphConfig = attachNodeParams(
      flowToGraphConfig(nodes, edges),
      nodeParamsRef.current
    );
    const dataStr = JSON.stringify(graphConfig, null, 2);
    const blob = new Blob([dataStr], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `graph-${new Date().toISOString().split("T")[0]}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
    setStatus("Graph exported");
  }, [nodes, edges, nodeParamsRef]);

  // Get current graph config
  const getCurrentGraphConfig = useCallback(
    () =>
      attachNodeParams(
        flowToGraphConfig(nodesRef.current, edgesRef.current),
        nodeParamsRef.current
      ),
    [nodeParamsRef]
  );

  const getGraphNodePrompts = useCallback((): Array<{
    nodeId: string;
    text: string;
  }> => {
    const results: Array<{ nodeId: string; text: string }> = [];
    for (const node of nodesRef.current) {
      if (node.data.nodeType !== "pipeline") continue;
      const text = (nodeParamsRef.current[node.id]?.__prompt as string) || "";
      if (text) results.push({ nodeId: node.id, text });
    }
    return results;
  }, [nodeParamsRef]);

  return {
    status,
    fitViewTrigger,
    handleSave,
    handleClear,
    handleImport,
    handleExport,
    refreshGraph: loadGraph,
    getCurrentGraphConfig,
    getGraphNodePrompts,
    initialLoadDone,
  };
}
