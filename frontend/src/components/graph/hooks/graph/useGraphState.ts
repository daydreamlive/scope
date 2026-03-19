import { useCallback, useEffect, useMemo, useRef } from "react";
import { useNodesState, useEdgesState } from "@xyflow/react";
import type { Edge, Node } from "@xyflow/react";
import { buildPipelinePortsMap } from "../../../../lib/graphUtils";
import type { FlowNodeData } from "../../../../lib/graphUtils";
import type { PipelineSchemaInfo } from "../../../../lib/api";
import { getPipelineSchemas } from "../../../../lib/api";
import { useState } from "react";

import { usePipelineParams } from "../node/usePipelineParams";
import {
  useGraphPersistence,
  enrichNodes,
  colorEdges,
  type EnrichNodesDeps,
} from "./useGraphPersistence";
import { useRerouteTypeSync } from "../value/useRerouteTypeSync";

type NodesSetter = React.Dispatch<React.SetStateAction<Node<FlowNodeData>[]>>;
type EdgesSetter = React.Dispatch<React.SetStateAction<Edge[]>>;

function useStableNodesSetter(rawSet: NodesSetter): NodesSetter {
  return useCallback<NodesSetter>(
    update => {
      if (typeof update === "function") {
        rawSet(prev => {
          const next = update(prev);
          if (next !== prev && next.length === 0 && prev.length === 0)
            return prev;
          return next;
        });
      } else {
        rawSet(update);
      }
    },
    [rawSet]
  );
}

function useStableEdgesSetter(rawSet: EdgesSetter): EdgesSetter {
  return useCallback<EdgesSetter>(
    update => {
      if (typeof update === "function") {
        rawSet(prev => {
          const next = update(prev);
          if (next !== prev && next.length === 0 && prev.length === 0)
            return prev;
          return next;
        });
      } else {
        rawSet(update);
      }
    },
    [rawSet]
  );
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
  onOutputSinkBulkChange?: (
    sinks: Record<string, { enabled: boolean; name: string }>
  ) => void;
  onStartRecording?: () => void;
  onStopRecording?: () => void;
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
  availability: GraphEditorAvailability,
  resolveRootGraphRef: React.RefObject<
    (
      nodes: Node<FlowNodeData>[],
      edges: Edge[]
    ) => { nodes: Node<FlowNodeData>[]; edges: Edge[] }
  >,
  resetNavigationRef: React.RefObject<() => void>
) {
  const [nodes, rawSetNodes, onNodesChange] = useNodesState<Node<FlowNodeData>>(
    []
  );
  const [edges, rawSetEdges, onEdgesChange] = useEdgesState<Edge>([]);

  const setNodes = useStableNodesSetter(rawSetNodes);
  const setEdges = useStableEdgesSetter(rawSetEdges);

  const [selectedNodeIds, setSelectedNodeIds] = useState<string[]>([]);

  const [availablePipelineIds, setAvailablePipelineIds] = useState<string[]>(
    []
  );
  const [portsMap, setPortsMap] = useState<
    Record<string, { inputs: string[]; outputs: string[] }>
  >({});
  const [pipelineSchemas, setPipelineSchemas] = useState<
    Record<string, PipelineSchemaInfo>
  >({});

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

  const nodesRef = useRef(nodes);
  nodesRef.current = nodes;

  const edgesRef = useRef(edges);
  edgesRef.current = edges;

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

  const onOutputSinkBulkChangeRef = useRef(callbacks.onOutputSinkBulkChange);
  onOutputSinkBulkChangeRef.current = callbacks.onOutputSinkBulkChange;

  const onStartRecordingRef = useRef(callbacks.onStartRecording);
  onStartRecordingRef.current = callbacks.onStartRecording;

  const onStopRecordingRef = useRef(callbacks.onStopRecording);
  onStopRecordingRef.current = callbacks.onStopRecording;

  const handleEdgeDelete = useCallback(
    (edgeId: string) => {
      setEdges(eds => eds.filter(e => e.id !== edgeId));
    },
    [setEdges]
  );

  const params = usePipelineParams({
    setNodes,
    portsMap,
    pipelineSchemas,
    isStreamingRef,
    nodesRef,
    onNodeParameterChange: callbacks.onNodeParameterChange,
  });

  const enrichDeps: EnrichNodesDeps = {
    availablePipelineIds,
    portsMap,
    pipelineSchemas,
    handlePipelineSelect: params.handlePipelineSelect,
    handleNodeParameterChange: params.handleNodeParameterChange,
    handlePromptChange: params.handlePromptChange,
    handlePromptSubmit: params.handlePromptSubmit,
    nodeParamsRef: params.nodeParamsRef,
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
    isStreaming: streams.isStreaming,
    onStartRecordingRef,
    onStopRecordingRef,
  };

  const enrichDepsRef = useRef(enrichDeps);
  enrichDepsRef.current = enrichDeps;

  useEffect(() => {
    if (availablePipelineIds.length === 0) return;
    setNodes(nds => {
      if (nds.length === 0) return nds;
      return enrichNodes(nds, enrichDeps);
    });
  }, [
    availablePipelineIds,
    portsMap,
    params.handlePipelineSelect,
    setNodes,
    pipelineSchemas,
    params.handleNodeParameterChange,
    streams.localStream,
    streams.remoteStream,
    streams.isStreaming,
    availability.spoutAvailable,
    availability.ndiAvailable,
    availability.syphonAvailable,
    availability.spoutOutputAvailable,
    availability.ndiOutputAvailable,
    availability.syphonOutputAvailable,
  ]);

  // Re-color edges only when node types change (not positions/selections)
  const nodeTypeFingerprint = useMemo(
    () => nodes.map(n => `${n.id}:${n.data.nodeType}`).join(","),
    [nodes]
  );
  const prevNodeTypesRef = useRef("");
  useEffect(() => {
    if (nodes.length === 0) return;
    if (nodeTypeFingerprint === prevNodeTypesRef.current) return;
    prevNodeTypesRef.current = nodeTypeFingerprint;
    setEdges(eds => colorEdges(eds, nodes, handleEdgeDelete));
  }, [nodes, nodeTypeFingerprint, setEdges, handleEdgeDelete]);

  useRerouteTypeSync(edges, nodesRef, setNodes, setEdges);

  const persistence = useGraphPersistence({
    nodes,
    edges,
    setNodes,
    setEdges,
    portsMap,
    nodeParamsRef: params.nodeParamsRef,
    setNodeParams: params.setNodeParams,
    enrichDepsRef,
    handleEdgeDelete,
    onGraphChange: callbacks.onGraphChange,
    onGraphClear: callbacks.onGraphClear,
    resolveRootGraphRef,
    resetNavigationRef,
  });

  return {
    nodes,
    setNodes,
    onNodesChange,
    edges,
    setEdges,
    onEdgesChange,
    selectedNodeIds,
    setSelectedNodeIds,
    availablePipelineIds,
    portsMap,
    pipelineSchemas,
    nodeParams: params.nodeParams,
    handlePipelineSelect: params.handlePipelineSelect,
    handleNodeParameterChange: params.handleNodeParameterChange,
    handlePromptChange: params.handlePromptChange,
    handlePromptSubmit: params.handlePromptSubmit,
    resolveBackendId: params.resolveBackendId,
    isStreamingRef,
    onNodeParamChangeRef: params.onNodeParamChangeRef,
    onOutputSinkChangeRef,
    onOutputSinkBulkChangeRef,
    enrichDepsRef,
    handleEdgeDelete,
    status: persistence.status,
    fitViewTrigger: persistence.fitViewTrigger,
    handleSave: persistence.handleSave,
    handleClear: persistence.handleClear,
    handleImport: persistence.handleImport,
    handleExport: persistence.handleExport,
    refreshGraph: persistence.refreshGraph,
    getCurrentGraphConfig: persistence.getCurrentGraphConfig,
    getGraphNodePrompts: persistence.getGraphNodePrompts,
    getGraphVaceSettings: persistence.getGraphVaceSettings,
    pendingImportWorkflow: persistence.pendingImportWorkflow,
    pendingResolutionPlan: persistence.pendingResolutionPlan,
    pendingImportResolving: persistence.pendingImportResolving,
    confirmImport: persistence.confirmImport,
    cancelImport: persistence.cancelImport,
    reResolveImport: persistence.reResolveImport,
  };
}
