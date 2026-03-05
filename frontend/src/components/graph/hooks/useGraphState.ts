import { useCallback, useEffect, useRef, useState } from "react";
import { useNodesState, useEdgesState } from "@xyflow/react";
import type { Edge, Node } from "@xyflow/react";
import {
  graphConfigToFlow,
  flowToGraphConfig,
  buildPipelinePortsMap,
  extractParameterPorts,
} from "../../../lib/graphUtils";
import type { FlowNodeData } from "../../../lib/graphUtils";
import type { PipelineSchemaInfo } from "../../../lib/api";
import {
  getGraph,
  setGraph,
  clearGraph,
  getPipelineSchemas,
} from "../../../lib/api";
import { getEdgeColor } from "../constants";

interface EnrichNodesDeps {
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

function enrichNodes(
  flowNodes: Node<FlowNodeData>[],
  deps: EnrichNodesDeps
): Node<FlowNodeData>[] {
  return flowNodes.map(n => {
    if (n.data.nodeType === "pipeline") {
      const pipelineId = n.data.pipelineId;
      const schema = pipelineId ? deps.pipelineSchemas[pipelineId] : null;
      const parameterInputs = schema ? extractParameterPorts(schema) : [];
      const supportsPrompts = schema?.supports_prompts ?? false;
      const nodeParamValues = deps.nodeParamsRef.current?.[n.id] || {};
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
          promptText: (nodeParamValues.__prompt as string) || "",
          onPromptChange: deps.handlePromptChange,
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

function colorEdges(
  flowEdges: Edge[],
  enrichedNodes: Node<FlowNodeData>[],
  handleEdgeDelete: (edgeId: string) => void
): Edge[] {
  return flowEdges.map(edge => {
    const sourceNode = enrichedNodes.find(n => n.id === edge.source);
    const edgeColor = getEdgeColor(sourceNode, edge.sourceHandle);
    return {
      ...edge,
      type: "default",
      reconnectable: "target" as const,
      style: { stroke: edgeColor, strokeWidth: 2 },
      animated: false,
      data: { onDelete: handleEdgeDelete },
    };
  });
}

export interface GraphEditorCallbacks {
  onNodeParameterChange?: (nodeId: string, key: string, value: unknown) => void;
  onGraphChange?: () => void;
  onGraphClear?: () => void;
  onVideoFileUpload?: (file: File) => Promise<boolean>;
  onSourceModeChange?: (mode: string) => void;
  onSpoutSourceChange?: (name: string) => void;
  onNdiSourceChange?: (identifier: string) => void;
  onSyphonSourceChange?: (identifier: string) => void;
  onOutputSinkChange?: (
    sinkType: string,
    config: { enabled: boolean; name: string }
  ) => void;
}

export interface GraphEditorStreams {
  localStream?: MediaStream | null;
  remoteStream?: MediaStream | null;
  isStreaming: boolean;
}

export interface GraphEditorAvailability {
  spoutAvailable: boolean;
  ndiAvailable: boolean;
  syphonAvailable: boolean;
  spoutOutputAvailable: boolean;
  ndiOutputAvailable: boolean;
  syphonOutputAvailable: boolean;
}

export function useGraphState(
  callbacks: GraphEditorCallbacks,
  streams: GraphEditorStreams,
  availability: GraphEditorAvailability
) {
  const [nodes, setNodes, onNodesChange] = useNodesState<Node<FlowNodeData>>(
    []
  );
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
  const [status, setStatus] = useState<string>("");
  const [graphSource, setGraphSource] = useState<string | null>(null);
  const [availablePipelineIds, setAvailablePipelineIds] = useState<string[]>(
    []
  );
  const [portsMap, setPortsMap] = useState<
    Record<string, { inputs: string[]; outputs: string[] }>
  >({});
  const [pipelineSchemas, setPipelineSchemas] = useState<
    Record<string, PipelineSchemaInfo>
  >({});
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [nodeParams, setNodeParams] = useState<
    Record<string, Record<string, unknown>>
  >({});

  // Stable refs for callbacks
  const nodesRef = useRef(nodes);
  nodesRef.current = nodes;

  const onNodeParamChangeRef = useRef(callbacks.onNodeParameterChange);
  onNodeParamChangeRef.current = callbacks.onNodeParameterChange;

  const isStreamingRef = useRef(streams.isStreaming);
  isStreamingRef.current = streams.isStreaming;

  const onVideoFileUploadRef = useRef(callbacks.onVideoFileUpload);
  onVideoFileUploadRef.current = callbacks.onVideoFileUpload;

  const onSourceModeChangeRef = useRef(callbacks.onSourceModeChange);
  onSourceModeChangeRef.current = callbacks.onSourceModeChange;

  const onSpoutSourceChangeRef = useRef(callbacks.onSpoutSourceChange);
  onSpoutSourceChangeRef.current = callbacks.onSpoutSourceChange;

  const onNdiSourceChangeRef = useRef(callbacks.onNdiSourceChange);
  onNdiSourceChangeRef.current = callbacks.onNdiSourceChange;

  const onSyphonSourceChangeRef = useRef(callbacks.onSyphonSourceChange);
  onSyphonSourceChangeRef.current = callbacks.onSyphonSourceChange;

  const onOutputSinkChangeRef = useRef(callbacks.onOutputSinkChange);
  onOutputSinkChangeRef.current = callbacks.onOutputSinkChange;

  const onGraphChangeRef = useRef(callbacks.onGraphChange);
  onGraphChangeRef.current = callbacks.onGraphChange;

  const nodeParamsRef = useRef(nodeParams);
  nodeParamsRef.current = nodeParams;

  const initialLoadDone = useRef(false);

  // Resolve node ID for backend communication
  const resolveBackendId = useCallback((nodeId: string): string => {
    return nodeId;
  }, []);

  const handlePipelineSelect = useCallback(
    (nodeId: string, newPipelineId: string | null) => {
      setNodes(nds =>
        nds.map(n => {
          if (n.id !== nodeId) return n;
          const ports =
            newPipelineId && portsMap ? portsMap[newPipelineId] : null;
          const schema = newPipelineId ? pipelineSchemas[newPipelineId] : null;
          const parameterInputs = schema ? extractParameterPorts(schema) : [];
          const supportsPrompts = schema?.supports_prompts ?? false;
          const newStyle = { ...n.style };
          delete newStyle.height;
          return {
            ...n,
            style: newStyle,
            height: undefined,
            measured: undefined,
            data: {
              ...n.data,
              pipelineId: newPipelineId,
              label: newPipelineId || n.id,
              streamInputs: ports?.inputs ?? ["video"],
              streamOutputs: ports?.outputs ?? ["video"],
              parameterInputs,
              supportsPrompts,
            },
          };
        })
      );
    },
    [setNodes, portsMap, pipelineSchemas]
  );

  const handleNodeParameterChange = useCallback(
    (nodeId: string, key: string, value: unknown) => {
      setNodeParams(prev => ({
        ...prev,
        [nodeId]: { ...(prev[nodeId] || {}), [key]: value },
      }));
      onNodeParamChangeRef.current?.(resolveBackendId(nodeId), key, value);
    },
    [resolveBackendId]
  );

  const handlePromptChange = useCallback(
    (nodeId: string, text: string) => {
      setNodeParams(prev => ({
        ...prev,
        [nodeId]: { ...(prev[nodeId] || {}), __prompt: text },
      }));
      if (isStreamingRef.current) {
        onNodeParamChangeRef.current?.(resolveBackendId(nodeId), "prompts", [
          { text, weight: 100 },
        ]);
      }
    },
    [resolveBackendId]
  );

  const handleEdgeDelete = useCallback(
    (edgeId: string) => {
      setEdges(eds => eds.filter(e => e.id !== edgeId));
    },
    [setEdges]
  );

  // Fetch pipeline schemas on mount
  useEffect(() => {
    getPipelineSchemas()
      .then(schemas => {
        setAvailablePipelineIds(Object.keys(schemas.pipelines));
        setPortsMap(buildPipelinePortsMap(schemas.pipelines));
        setPipelineSchemas(schemas.pipelines);
      })
      .catch(err => {
        console.error("Failed to fetch pipeline schemas:", err);
      });
  }, []);

  // Build enrichment deps object
  const enrichDeps: EnrichNodesDeps = {
    availablePipelineIds,
    portsMap,
    pipelineSchemas,
    handlePipelineSelect,
    handleNodeParameterChange,
    handlePromptChange,
    nodeParamsRef,
    localStream: streams.localStream,
    remoteStream: streams.remoteStream,
    onVideoFileUploadRef,
    onSourceModeChangeRef,
    onSpoutSourceChangeRef,
    onNdiSourceChangeRef,
    onSyphonSourceChangeRef,
    spoutAvailable: availability.spoutAvailable,
    ndiAvailable: availability.ndiAvailable,
    syphonAvailable: availability.syphonAvailable,
    spoutOutputAvailable: availability.spoutOutputAvailable,
    ndiOutputAvailable: availability.ndiOutputAvailable,
    syphonOutputAvailable: availability.syphonOutputAvailable,
    handleEdgeDelete,
  };

  // Enrich nodes when dependencies change
  useEffect(() => {
    if (availablePipelineIds.length === 0) return;
    setNodes(nds => enrichNodes(nds, enrichDeps));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    availablePipelineIds,
    portsMap,
    handlePipelineSelect,
    setNodes,
    pipelineSchemas,
    handleNodeParameterChange,
    streams.localStream,
    streams.remoteStream,
    availability.spoutAvailable,
    availability.ndiAvailable,
    availability.syphonAvailable,
    availability.spoutOutputAvailable,
    availability.ndiOutputAvailable,
    availability.syphonOutputAvailable,
  ]);

  // Sync nodeParams to pipeline node parameterValues
  useEffect(() => {
    setNodes(nds =>
      nds.map(n => {
        if (n.data.nodeType === "pipeline") {
          const vals = nodeParams[n.id] || {};
          if (n.data.parameterValues === vals) return n;
          return {
            ...n,
            data: {
              ...n.data,
              parameterValues: vals,
              promptText: (vals.__prompt as string) || "",
            },
          };
        }
        return n;
      })
    );
  }, [nodeParams, setNodes]);

  // Reload graph from backend (shared by initial load and manual refresh)
  const loadGraphFromBackend = useCallback(() => {
    if (Object.keys(portsMap).length === 0) return;

    getGraph()
      .then(response => {
        if (response.graph) {
          const { nodes: flowNodes, edges: flowEdges } = graphConfigToFlow(
            response.graph,
            portsMap
          );
          const enriched = enrichNodes(flowNodes, enrichDeps);
          setNodes(enriched);
          setEdges(colorEdges(flowEdges, enriched, handleEdgeDelete));
          setGraphSource(response.source);
          setStatus(`Loaded from ${response.source}`);
        } else {
          setStatus("No graph configured");
        }
        initialLoadDone.current = true;
      })
      .catch(err => {
        console.error("Failed to load graph:", err);
        setStatus("Failed to load graph");
        initialLoadDone.current = true;
      });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [portsMap]);

  // Load graph from backend on mount / when portsMap changes
  useEffect(() => {
    loadGraphFromBackend();
  }, [loadGraphFromBackend]);

  // Notify parent on graph changes
  useEffect(() => {
    if (!initialLoadDone.current) return;
    if (nodes.length === 0 && edges.length === 0) return;
    onGraphChangeRef.current?.();
  }, [nodes, edges]);

  // Auto-save graph to backend (debounced)
  useEffect(() => {
    if (!initialLoadDone.current) return;
    if (nodes.length === 0 && edges.length === 0) return;

    const timer = setTimeout(async () => {
      try {
        const graphConfig = flowToGraphConfig(nodes, edges);
        const result = await setGraph(graphConfig);
        setGraphSource("api");
        setStatus(`Saved: ${result.nodes} nodes, ${result.edges} edges`);
      } catch (err) {
        const message = err instanceof Error ? err.message : "Unknown error";
        setStatus(`Save failed: ${message}`);
      }
    }, 500);

    return () => clearTimeout(timer);
  }, [nodes, edges]);

  // Clear graph
  const handleClear = useCallback(async () => {
    try {
      await clearGraph();
      setNodes([]);
      setEdges([]);
      setGraphSource(null);
      setSelectedNodeId(null);
      setStatus("Graph cleared");
      callbacks.onGraphClear?.();
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unknown error";
      setStatus(`Clear failed: ${message}`);
    }
  }, [setNodes, setEdges, callbacks.onGraphClear]);

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
          const enriched = enrichNodes(flowNodes, enrichDeps);
          setNodes(enriched);
          setEdges(colorEdges(flowEdges, enriched, handleEdgeDelete));
          setGraphSource(null);
          setStatus(`Imported from ${file.name}`);
        } catch {
          setStatus("Import failed: invalid JSON");
        }
      };
      reader.readAsText(file);
      event.target.value = "";
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [portsMap, availablePipelineIds, handlePipelineSelect, setNodes, setEdges]
  );

  // Export graph as JSON file
  const handleExport = useCallback(() => {
    const graphConfig = flowToGraphConfig(nodes, edges);
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
  }, [nodes, edges]);

  return {
    nodes,
    setNodes,
    onNodesChange,
    edges,
    setEdges,
    onEdgesChange,
    status,
    graphSource,
    availablePipelineIds,
    portsMap,
    pipelineSchemas,
    selectedNodeId,
    setSelectedNodeId,
    nodeParams,
    handlePipelineSelect,
    handleNodeParameterChange,
    handlePromptChange,
    handleEdgeDelete,
    resolveBackendId,
    isStreamingRef,
    onNodeParamChangeRef,
    onOutputSinkChangeRef,
    handleClear,
    handleImport,
    handleExport,
    refreshGraph: loadGraphFromBackend,
  };
}
