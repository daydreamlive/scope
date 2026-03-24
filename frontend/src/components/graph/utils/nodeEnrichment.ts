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
