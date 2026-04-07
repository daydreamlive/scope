import type { Edge, Node } from "@xyflow/react";
import { extractParameterPorts } from "../../../lib/graphUtils";
import type { FlowNodeData } from "../../../lib/graphUtils";
import type { GraphConfig, PipelineSchemaInfo } from "../../../lib/api";
import { buildEdgeStyle } from "../constants";

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
  /** Per-source-node local streams (multi-source) */
  localStreams?: Record<string, MediaStream>;
  remoteStream?: MediaStream | null;
  /** Per-sink-node remote streams (multi-sink) */
  remoteStreams?: Record<string, MediaStream>;
  /** Per-sink-node WebRTC stats */
  sinkStats?: Record<string, { fps: number; bitrate: number }>;
  onVideoFileUploadRef: React.RefObject<
    ((file: File, nodeId?: string) => Promise<boolean>) | undefined
  >;
  onSourceModeChangeRef: React.RefObject<
    ((mode: string, nodeId?: string) => void) | undefined
  >;
  onSpoutSourceChangeRef: React.RefObject<((name: string) => void) | undefined>;
  onNdiSourceChangeRef: React.RefObject<
    ((identifier: string) => void) | undefined
  >;
  onSyphonSourceChangeRef: React.RefObject<
    ((identifier: string) => void) | undefined
  >;
  onCycleSampleVideoRef: React.RefObject<(() => void) | undefined>;
  spoutAvailable: boolean;
  ndiAvailable: boolean;
  syphonAvailable: boolean;
  spoutOutputAvailable: boolean;
  ndiOutputAvailable: boolean;
  syphonOutputAvailable: boolean;
  handleEdgeDelete: (edgeId: string) => void;
  isStreaming: boolean;
  isLoading: boolean;
  loadingStage?: string | null;
  isPlaying?: boolean;
  onPlayPauseToggleRef: React.RefObject<(() => void) | undefined>;
  onStartRecordingRef: React.RefObject<((nodeId?: string) => void) | undefined>;
  onStopRecordingRef: React.RefObject<((nodeId?: string) => void) | undefined>;
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
  onEnableTempoRef: React.RefObject<
    ((req: import("../../../lib/api").TempoEnableRequest) => void) | undefined
  >;
  onDisableTempoRef: React.RefObject<(() => void) | undefined>;
  onSetTempoRef: React.RefObject<((bpm: number) => void) | undefined>;
  onRefreshTempoSourcesRef: React.RefObject<(() => void) | undefined>;
}

const FIXED_SIZE_NODE_TYPES = new Set(["source", "sink", "image"]);

/**
 * Clear saved height from nodes that use autoMinHeight so NodeCard's
 * ResizeObserver can recalculate the proper minimum. Width is preserved.
 */
export function resetAutoHeightNodes(
  nodes: Node<FlowNodeData>[]
): Node<FlowNodeData>[] {
  return nodes.map(n => {
    if (FIXED_SIZE_NODE_TYPES.has(n.data.nodeType as string)) return n;
    if (n.height == null && n.style?.height == null) return n;
    const rest = Object.fromEntries(
      Object.entries(n).filter(([k]) => k !== "height" && k !== "measured")
    );
    const restStyle = Object.fromEntries(
      Object.entries((n.style ?? {}) as Record<string, unknown>).filter(
        ([k]) => k !== "height"
      )
    );
    return {
      ...rest,
      style: Object.keys(restStyle).length > 0 ? restStyle : undefined,
    } as Node<FlowNodeData>;
  });
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
      const supportsLoRA = schema?.supports_lora ?? false;
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
          supportsLoRA,
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
      // Per-node stream if available (multi-source), else fall back to global
      const nodeStream = deps.localStreams?.[n.id] ?? deps.localStream;
      return {
        ...n,
        data: {
          ...n.data,
          localStream: nodeStream,
          onVideoFileUpload: (file: File) =>
            deps.onVideoFileUploadRef.current?.(file, n.id) ??
            Promise.resolve(false),
          onSourceModeChange: (mode: string) =>
            deps.onSourceModeChangeRef.current?.(mode, n.id),
          spoutAvailable: deps.spoutAvailable,
          ndiAvailable: deps.ndiAvailable,
          syphonAvailable: deps.syphonAvailable,
          onSpoutSourceChange: (name: string) =>
            deps.onSpoutSourceChangeRef.current?.(name),
          onNdiSourceChange: (identifier: string) =>
            deps.onNdiSourceChangeRef.current?.(identifier),
          onSyphonSourceChange: (identifier: string) =>
            deps.onSyphonSourceChangeRef.current?.(identifier),
          onCycleSampleVideo: () => deps.onCycleSampleVideoRef.current?.(),
          isStreaming: deps.isStreaming,
        },
      };
    }
    if (n.data.nodeType === "sink") {
      // Per-node remote stream and stats (multi-sink)
      const nodeRemoteStream = deps.remoteStreams?.[n.id] ?? deps.remoteStream;
      const nodeStats = deps.sinkStats?.[n.id];
      return {
        ...n,
        data: {
          ...n.data,
          remoteStream: nodeRemoteStream,
          sinkStats: nodeStats,
          isPlaying: deps.isPlaying,
          isLoading: deps.isLoading,
          loadingStage: deps.loadingStage,
          onPlayPauseToggle: () => deps.onPlayPauseToggleRef.current?.(),
        },
      };
    }
    if (n.data.nodeType === "record") {
      const nodeId = n.id;
      return {
        ...n,
        data: {
          ...n.data,
          isStreaming: deps.isStreaming,
          onStartRecording: () => deps.onStartRecordingRef.current?.(nodeId),
          onStopRecording: () => deps.onStopRecordingRef.current?.(nodeId),
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
    if (n.data.nodeType === "tempo") {
      const ts = deps.tempoState;
      return {
        ...n,
        data: {
          ...n.data,
          tempoEnabled: ts?.enabled ?? false,
          tempoBpm: ts?.bpm ?? null,
          tempoBeatPhase: ts?.beatPhase ?? 0,
          tempoBeatCount: ts?.beatCount ?? 0,
          tempoBarPosition: ts?.barPosition ?? 0,
          tempoIsPlaying: ts?.isPlaying ?? false,
          tempoSourceType: ts?.sourceType ?? null,
          tempoNumPeers: ts?.numPeers ?? null,
          tempoBeatsPerBar: ts?.beatsPerBar ?? 4,
          tempoSources: deps.tempoSources,
          isStreaming: deps.isStreaming,
          tempoLoading: deps.tempoLoading ?? false,
          tempoError: deps.tempoError ?? null,
          onEnableTempo: deps.onEnableTempoRef.current,
          onDisableTempo: () => deps.onDisableTempoRef.current?.(),
          onSetTempo: (bpm: number) => deps.onSetTempoRef.current?.(bpm),
          onRefreshTempoSources: () =>
            deps.onRefreshTempoSourcesRef.current?.(),
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

export function attachNodeParams(
  config: GraphConfig,
  params: Record<string, Record<string, unknown>>
): GraphConfig {
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
