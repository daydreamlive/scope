import { useCallback, useEffect, useRef, useState } from "react";
import type { Edge, Node } from "@xyflow/react";
import {
  graphConfigToFlow,
  flowToGraphConfig,
  extractParameterPorts,
  workflowToGraphConfig,
} from "../../../../lib/graphUtils";
import type { FlowNodeData } from "../../../../lib/graphUtils";
import type { PipelineSchemaInfo, PluginInfo } from "../../../../lib/api";
import type { ScopeWorkflow } from "../../../../lib/workflowApi";
import { buildGraphWorkflow } from "../../../../lib/workflowSettings";
import { usePipelinesContext } from "../../../../contexts/PipelinesContext";
import { usePluginsContext } from "../../../../contexts/PluginsContext";
import { useServerInfoContext } from "../../../../contexts/ServerInfoContext";
import { buildEdgeStyle } from "../../constants";

const LS_GRAPH_KEY = "scope:graph:backup";

function saveGraphToLocalStorage(graphJson: string): void {
  try {
    localStorage.setItem(LS_GRAPH_KEY, graphJson);
  } catch {
    // ignore
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

export function extractNodeParams(
  uiState: Record<string, unknown> | null | undefined
): Record<string, Record<string, unknown>> {
  if (!uiState || typeof uiState !== "object") return {};
  const raw = (uiState as Record<string, unknown>).node_params;
  if (!raw || typeof raw !== "object") return {};
  return raw as Record<string, Record<string, unknown>>;
}

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
  isStreaming: boolean;
  onStartRecordingRef: React.RefObject<(() => void) | undefined>;
  onStopRecordingRef: React.RefObject<(() => void) | undefined>;
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
          isStreaming: deps.isStreaming,
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
          onVideoFileUpload: (file: File) =>
            deps.onVideoFileUploadRef.current?.(file) ?? Promise.resolve(false),
          onSourceModeChange: (mode: string) =>
            deps.onSourceModeChangeRef.current?.(mode),
          spoutAvailable: deps.spoutAvailable,
          ndiAvailable: deps.ndiAvailable,
          syphonAvailable: deps.syphonAvailable,
          onSpoutSourceChange: (name: string) =>
            deps.onSpoutSourceChangeRef.current?.(name),
          onNdiSourceChange: (identifier: string) =>
            deps.onNdiSourceChangeRef.current?.(identifier),
          onSyphonSourceChange: (identifier: string) =>
            deps.onSyphonSourceChangeRef.current?.(identifier),
        },
      };
    }
    if (n.data.nodeType === "sink") {
      return { ...n, data: { ...n.data, remoteStream: deps.remoteStream } };
    }
    if (n.data.nodeType === "record") {
      return {
        ...n,
        data: {
          ...n.data,
          isStreaming: deps.isStreaming,
          onStartRecording: deps.onStartRecordingRef.current,
          onStopRecording: deps.onStopRecordingRef.current,
        },
      };
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
  resolveRootGraphRef: React.RefObject<
    (
      nodes: Node<FlowNodeData>[],
      edges: Edge[]
    ) => { nodes: Node<FlowNodeData>[]; edges: Edge[] }
  >;
  resetNavigationRef: React.RefObject<() => void>;
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
  resolveRootGraphRef,
  resetNavigationRef,
}: UseGraphPersistenceArgs) {
  const [status, setStatus] = useState<string>("");
  const [fitViewTrigger, setFitViewTrigger] = useState(0);

  const { pipelines: pipelineInfoMap } = usePipelinesContext();
  const { plugins } = usePluginsContext();
  const { version: scopeVersion } = useServerInfoContext();

  const nodesRef = useRef(nodes);
  nodesRef.current = nodes;
  const edgesRef = useRef(edges);
  edgesRef.current = edges;

  const onGraphChangeRef = useRef(onGraphChange);
  onGraphChangeRef.current = onGraphChange;

  const initialLoadDone = useRef(false);

  const loadGraph = useCallback(() => {
    if (Object.keys(portsMap).length === 0) return;
    resetNavigationRef.current?.();

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

        const sourceNode = flowNodes.find(n => n.data.nodeType === "source");
        const restoredMode = sourceNode?.data.sourceMode as string | undefined;
        if (restoredMode && restoredMode !== "video") {
          setTimeout(() => {
            enrichDepsRef.current.onSourceModeChangeRef.current?.(restoredMode);
          }, 0);
        }
      } catch {
        setStatus("No graph configured");
      }
    } else {
      setStatus("No graph configured");
    }
    initialLoadDone.current = true;
  }, [portsMap]);

  useEffect(() => {
    loadGraph();
  }, [loadGraph]);

  useEffect(() => {
    if (!initialLoadDone.current) return;
    if (nodes.length === 0 && edges.length === 0) return;
    onGraphChangeRef.current?.();
  }, [nodes, edges]);

  useEffect(() => {
    if (!initialLoadDone.current) return;
    if (nodes.length === 0 && edges.length === 0) return;

    const timer = setTimeout(() => {
      try {
        const root = resolveRootGraphRef.current(nodes, edges);
        const graphConfig = attachNodeParams(
          flowToGraphConfig(root.nodes, root.edges),
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
  }, [nodes, edges, nodeParamsRef, resolveRootGraphRef]);

  const handleSave = useCallback(() => {
    if (nodes.length === 0 && edges.length === 0) {
      setStatus("Nothing to save");
      return;
    }
    try {
      const root = resolveRootGraphRef.current(nodes, edges);
      const graphConfig = attachNodeParams(
        flowToGraphConfig(root.nodes, root.edges),
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
  }, [nodes, edges, nodeParamsRef, resolveRootGraphRef]);

  useEffect(() => {
    const handler = () => {
      try {
        const currentNodes = nodesRef.current;
        const currentEdges = edgesRef.current;
        if (currentNodes.length > 0 || currentEdges.length > 0) {
          const root = resolveRootGraphRef.current(currentNodes, currentEdges);
          const graphConfig = attachNodeParams(
            flowToGraphConfig(root.nodes, root.edges),
            nodeParamsRef.current
          );
          saveGraphToLocalStorage(JSON.stringify(graphConfig));
        }
      } catch {
        // ignore
      }
    };

    window.addEventListener("beforeunload", handler);
    return () => window.removeEventListener("beforeunload", handler);
  }, [nodeParamsRef, resolveRootGraphRef]);

  const handleClear = useCallback(() => {
    resetNavigationRef.current?.();
    clearGraphFromLocalStorage();
    setNodes([]);
    setEdges([]);
    setStatus("Graph cleared");
    onGraphClear?.();
  }, [setNodes, setEdges, onGraphClear, resetNavigationRef]);

  const handleImport = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      if (!file) return;

      const reader = new FileReader();
      reader.onload = e => {
        try {
          const parsed = JSON.parse(e.target?.result as string);

          let graphConfig: Parameters<typeof graphConfigToFlow>[0];
          let importedParams: Record<string, Record<string, unknown>> | null =
            null;

          if (
            parsed.format === "scope-workflow" &&
            Array.isArray(parsed.pipelines)
          ) {
            const workflow = parsed as ScopeWorkflow;
            if (workflow.graph?.nodes && workflow.graph?.edges) {
              graphConfig = workflow.graph as Parameters<
                typeof graphConfigToFlow
              >[0];
            } else {
              const result = workflowToGraphConfig(workflow);
              graphConfig = result.graphConfig as Parameters<
                typeof graphConfigToFlow
              >[0];
              importedParams = result.nodeParams;
            }
          } else if (parsed.nodes && parsed.edges) {
            graphConfig = parsed;
          } else {
            setStatus("Import failed: unrecognized format");
            return;
          }

          resetNavigationRef.current?.();
          const { nodes: flowNodes, edges: flowEdges } = graphConfigToFlow(
            graphConfig,
            portsMap
          );
          const restoredParams =
            importedParams ?? extractNodeParams(graphConfig.ui_state);
          setNodeParams(restoredParams);
          const enriched = enrichNodes(flowNodes, enrichDepsRef.current);
          setNodes(enriched);
          setEdges(colorEdges(flowEdges, enriched, handleEdgeDelete));
          setStatus(`Imported from ${file.name}`);
          setFitViewTrigger(c => c + 1);

          const sourceNode = flowNodes.find(n => n.data.nodeType === "source");
          const importedMode = sourceNode?.data.sourceMode as
            | string
            | undefined;
          if (importedMode) {
            setTimeout(() => {
              enrichDepsRef.current.onSourceModeChangeRef.current?.(
                importedMode
              );
            }, 0);
          }
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
      resetNavigationRef,
    ]
  );

  const handleExport = useCallback(() => {
    const root = resolveRootGraphRef.current(nodes, edges);
    const graphConfig = attachNodeParams(
      flowToGraphConfig(root.nodes, root.edges),
      nodeParamsRef.current
    );

    const pluginInfoMap = new Map<string, PluginInfo>(
      plugins.map(p => [p.name, p])
    );

    const workflow = buildGraphWorkflow({
      name: `Graph Export ${new Date().toISOString().split("T")[0]}`,
      graphConfig,
      pipelineInfoMap: pipelineInfoMap ?? {},
      pluginInfoMap,
      scopeVersion: scopeVersion ?? "unknown",
    });

    const dataStr = JSON.stringify(workflow, null, 2);
    const blob = new Blob([dataStr], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    const safeName = workflow.metadata.name
      .replace(/[^a-zA-Z0-9_-]/g, "_")
      .toLowerCase();
    link.download = `${safeName}.scope-workflow.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
    setStatus("Graph exported");
  }, [
    nodes,
    edges,
    nodeParamsRef,
    resolveRootGraphRef,
    pipelineInfoMap,
    plugins,
    scopeVersion,
  ]);

  const getCurrentGraphConfig = useCallback(() => {
    const root = resolveRootGraphRef.current(
      nodesRef.current,
      edgesRef.current
    );
    return attachNodeParams(
      flowToGraphConfig(root.nodes, root.edges),
      nodeParamsRef.current
    );
  }, [nodeParamsRef, resolveRootGraphRef]);

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
