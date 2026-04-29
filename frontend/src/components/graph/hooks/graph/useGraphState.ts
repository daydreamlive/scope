import { useCallback, useEffect, useMemo, useRef } from "react";
import { useNodesState, useEdgesState } from "@xyflow/react";
import type { Edge, Node } from "@xyflow/react";
import { buildPipelinePortsMap } from "../../../../lib/graphUtils";
import type { FlowNodeData } from "../../../../lib/graphUtils";
import type {
  PipelineSchemaInfo,
  TempoEnableRequest,
} from "../../../../lib/api";
import { useState } from "react";
import { useApi } from "../../../../hooks/useApi";
import { useCloudStatus } from "../../../../hooks/useCloudStatus";
import { usePipelinesContext } from "../../../../contexts/PipelinesContext";
import type { HardwareInfoResponse } from "../../../../lib/api";

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
  onVideoFileUpload?: (file: File, nodeId?: string) => Promise<boolean>;
  onSourceModeChange?: (mode: string, nodeId?: string) => void;
  onSpoutSourceChange?: (name: string) => void;
  onNdiSourceChange?: (identifier: string) => void;
  onSyphonSourceChange?: (identifier: string) => void;
  onCycleSampleVideo?: (nodeId?: string) => void;
  onInitSampleVideo?: (nodeId?: string) => void;
  onOutputSinkChange?: (
    sinkType: string,
    config: { enabled: boolean; name: string }
  ) => void;

  onStartRecording?: (nodeId?: string) => void;
  onStopRecording?: (nodeId?: string) => void;
  onEnableTempo?: (req: TempoEnableRequest) => void;
  onDisableTempo?: () => void;
  onSetTempo?: (bpm: number) => void;
  onRefreshTempoSources?: () => void;
}

export interface GraphEditorStreams {
  localStream?: MediaStream | null;
  localStreams?: Record<string, MediaStream>;
  remoteStream?: MediaStream | null;
  remoteStreams?: Record<string, MediaStream>;
  sinkStats?: Record<string, { fps: number; bitrate: number }>;
  isStreaming: boolean;
  isLoading?: boolean;
  loadingStage?: string | null;
  isPlaying?: boolean;
  onPlayPauseToggle?: () => void;
}

export interface GraphEditorTempo {
  tempoState?: {
    enabled: boolean;
    bpm: number | null;
    beatPhase: number;
    barPosition: number;
    beatCount: number;
    isPlaying: boolean;
    sourceType: string | null;
    numPeers: number | null;
    beatsPerBar: number;
  };
  tempoSources?: unknown;
  tempoLoading?: boolean;
  tempoError?: string | null;
}

export interface GraphEditorAvailability {
  spoutAvailable: boolean;
  ndiAvailable: boolean;
  syphonAvailable: boolean;
  spoutOutputAvailable: boolean;
  ndiOutputAvailable: boolean;
  syphonOutputAvailable: boolean;
  availableInputSources: import("../../../../lib/api").InputSourceType[];
}

export function useGraphState(
  callbacks: GraphEditorCallbacks,
  streams: GraphEditorStreams,
  availability: GraphEditorAvailability,
  tempo: GraphEditorTempo,
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

  const { getPipelineSchemas, getHardwareInfo } = useApi();
  const { isConnected: isCloudConnected } = useCloudStatus();
  const { pipelinesVersion } = usePipelinesContext();
  const [hardwareInfo, setHardwareInfo] = useState<HardwareInfoResponse | null>(
    null
  );

  useEffect(() => {
    let mounted = true;
    getPipelineSchemas()
      .then(schemas => {
        if (!mounted) return;
        setAvailablePipelineIds(Object.keys(schemas.pipelines));
        setPortsMap(buildPipelinePortsMap(schemas.pipelines));
        setPipelineSchemas(schemas.pipelines);
      })
      .catch(err => {
        if (!mounted) return;
        console.error("Failed to fetch pipeline schemas:", err);
      });
    getHardwareInfo()
      .then(info => {
        if (!mounted) return;
        setHardwareInfo(info);
      })
      .catch(err => {
        if (!mounted) return;
        console.error("Failed to fetch hardware info:", err);
      });
    return () => {
      mounted = false;
    };
  }, [getPipelineSchemas, getHardwareInfo, isCloudConnected, pipelinesVersion]);

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

  const onCycleSampleVideoRef = useRef(callbacks.onCycleSampleVideo);
  onCycleSampleVideoRef.current = callbacks.onCycleSampleVideo;

  const onInitSampleVideoRef = useRef(callbacks.onInitSampleVideo);
  onInitSampleVideoRef.current = callbacks.onInitSampleVideo;

  const onOutputSinkChangeRef = useRef(callbacks.onOutputSinkChange);
  onOutputSinkChangeRef.current = callbacks.onOutputSinkChange;

  const onStartRecordingRef = useRef(callbacks.onStartRecording);
  onStartRecordingRef.current = callbacks.onStartRecording;

  const onStopRecordingRef = useRef(callbacks.onStopRecording);
  onStopRecordingRef.current = callbacks.onStopRecording;

  const onEnableTempoRef = useRef(callbacks.onEnableTempo);
  onEnableTempoRef.current = callbacks.onEnableTempo;

  const onDisableTempoRef = useRef(callbacks.onDisableTempo);
  onDisableTempoRef.current = callbacks.onDisableTempo;

  const onSetTempoRef = useRef(callbacks.onSetTempo);
  onSetTempoRef.current = callbacks.onSetTempo;

  const onRefreshTempoSourcesRef = useRef(callbacks.onRefreshTempoSources);
  onRefreshTempoSourcesRef.current = callbacks.onRefreshTempoSources;

  const onPlayPauseToggleRef = useRef(streams.onPlayPauseToggle);
  onPlayPauseToggleRef.current = streams.onPlayPauseToggle;

  const handleEdgeDelete = useCallback(
    (edgeId: string) => {
      setEdges(eds => eds.filter(e => e.id !== edgeId));
    },
    [setEdges]
  );

  const params = usePipelineParams({
    setNodes,
    setEdges,
    portsMap,
    pipelineSchemas,
    isStreamingRef,
    nodesRef,
    onNodeParameterChange: callbacks.onNodeParameterChange,
    hardwareInfo,
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
    localStreams: streams.localStreams,
    remoteStream: streams.remoteStream,
    remoteStreams: streams.remoteStreams,
    sinkStats: streams.sinkStats,
    onVideoFileUploadRef,
    onSourceModeChangeRef,
    onSpoutSourceChangeRef,
    onNdiSourceChangeRef,
    onSyphonSourceChangeRef,
    onCycleSampleVideoRef,
    onInitSampleVideoRef,
    spoutAvailable: availability.spoutAvailable,
    ndiAvailable: availability.ndiAvailable,
    syphonAvailable: availability.syphonAvailable,
    spoutOutputAvailable: availability.spoutOutputAvailable,
    ndiOutputAvailable: availability.ndiOutputAvailable,
    syphonOutputAvailable: availability.syphonOutputAvailable,
    availableInputSources: availability.availableInputSources,
    handleEdgeDelete,
    isStreaming: streams.isStreaming,
    isLoading: streams.isLoading ?? false,
    loadingStage: streams.loadingStage,
    isPlaying: streams.isPlaying,
    onPlayPauseToggleRef,
    onStartRecordingRef,
    onStopRecordingRef,
    tempoState: tempo.tempoState,
    tempoSources: tempo.tempoSources,
    tempoLoading: tempo.tempoLoading,
    tempoError: tempo.tempoError,
    onEnableTempoRef,
    onDisableTempoRef,
    onSetTempoRef,
    onRefreshTempoSourcesRef,
  };

  const enrichDepsRef = useRef(enrichDeps);
  enrichDepsRef.current = enrichDeps;

  useEffect(() => {
    if (availablePipelineIds.length === 0) return;
    setNodes(nds => {
      if (nds.length === 0) return nds;
      return enrichNodes(nds, enrichDepsRef.current);
    });
  }, [
    availablePipelineIds,
    portsMap,
    params.handlePipelineSelect,
    setNodes,
    pipelineSchemas,
    params.handleNodeParameterChange,
    streams.localStream,
    streams.localStreams,
    streams.remoteStream,
    streams.remoteStreams,
    streams.sinkStats,
    streams.isStreaming,
    streams.isLoading,
    streams.loadingStage,
    streams.isPlaying,
    availability.spoutAvailable,
    availability.ndiAvailable,
    availability.syphonAvailable,
    availability.spoutOutputAvailable,
    availability.ndiOutputAvailable,
    availability.syphonOutputAvailable,
    availability.availableInputSources,
    tempo.tempoState,
    tempo.tempoSources,
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
    applyExternalNodeParams: params.applyExternalNodeParams,
    isStreamingRef,
    onNodeParamChangeRef: params.onNodeParamChangeRef,
    onOutputSinkChangeRef,
    enrichDepsRef,
    handleEdgeDelete,
    status: persistence.status,
    fitViewTrigger: persistence.fitViewTrigger,
    handleSave: persistence.handleSave,
    handleClear: persistence.handleClear,
    handleImport: persistence.handleImport,
    handleExport: persistence.handleExport,
    buildCurrentWorkflow: persistence.buildCurrentWorkflow,
    refreshGraph: persistence.refreshGraph,
    getCurrentGraphConfig: persistence.getCurrentGraphConfig,
    getGraphNodePrompts: persistence.getGraphNodePrompts,
    getGraphVaceSettings: persistence.getGraphVaceSettings,
    getGraphLoRASettings: persistence.getGraphLoRASettings,
    pendingImportWorkflow: persistence.pendingImportWorkflow,
    pendingResolutionPlan: persistence.pendingResolutionPlan,
    pendingImportResolving: persistence.pendingImportResolving,
    confirmImport: persistence.confirmImport,
    cancelImport: persistence.cancelImport,
    reResolveImport: persistence.reResolveImport,
    loadGraphFromParsed: persistence.loadGraphFromParsed,
    initialLoadDone: persistence.initialLoadDone,
  };
}
