import { useCallback, useEffect, useRef } from "react";
import { useNodesState, useEdgesState } from "@xyflow/react";
import type { Edge, Node } from "@xyflow/react";
import { buildPipelinePortsMap } from "../../../lib/graphUtils";
import type { FlowNodeData } from "../../../lib/graphUtils";
import type { PipelineSchemaInfo } from "../../../lib/api";
import { getPipelineSchemas } from "../../../lib/api";
import { useState } from "react";

import { usePipelineParams } from "./usePipelineParams";
import {
  useGraphPersistence,
  enrichNodes,
  type EnrichNodesDeps,
} from "./useGraphPersistence";
import { useRerouteTypeSync } from "./useRerouteTypeSync";

// Stable setters: prevent empty→empty array reference churn that causes render loops

type NodesSetter = React.Dispatch<React.SetStateAction<Node<FlowNodeData>[]>>;
type EdgesSetter = React.Dispatch<React.SetStateAction<Edge[]>>;

function useStableNodesSetter(rawSet: NodesSetter): NodesSetter {
  return useCallback<NodesSetter>(
    update => {
      if (typeof update === "function") {
        rawSet(prev => {
          const next = update(prev);
          // Return same ref when both empty to prevent render loop
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

// Types

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
  // Core state
  const [nodes, rawSetNodes, onNodesChange] = useNodesState<Node<FlowNodeData>>(
    []
  );
  const [edges, rawSetEdges, onEdgesChange] = useEdgesState<Edge>([]);

  // Wrap setters to prevent render loops
  const setNodes = useStableNodesSetter(rawSetNodes);
  const setEdges = useStableEdgesSetter(rawSetEdges);

  const [selectedNodeIds, setSelectedNodeIds] = useState<string[]>([]);

  // Pipeline schemas
  const [availablePipelineIds, setAvailablePipelineIds] = useState<string[]>(
    []
  );
  const [portsMap, setPortsMap] = useState<
    Record<string, { inputs: string[]; outputs: string[] }>
  >({});
  const [pipelineSchemas, setPipelineSchemas] = useState<
    Record<string, PipelineSchemaInfo>
  >({});

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

  // Refs
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

  // Edge deletion
  const handleEdgeDelete = useCallback(
    (edgeId: string) => {
      setEdges(eds => eds.filter(e => e.id !== edgeId));
    },
    [setEdges]
  );

  // Pipeline params
  const params = usePipelineParams({
    setNodes,
    portsMap,
    pipelineSchemas,
    isStreamingRef,
    nodesRef,
    onNodeParameterChange: callbacks.onNodeParameterChange,
  });

  // Enrichment deps
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
  };

  const enrichDepsRef = useRef(enrichDeps);
  enrichDepsRef.current = enrichDeps;

  // Enrich nodes on data changes
  useEffect(() => {
    if (availablePipelineIds.length === 0) return;
    setNodes(nds => {
      if (nds.length === 0) return nds; // nothing to enrich
      return enrichNodes(nds, enrichDeps);
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    availablePipelineIds,
    portsMap,
    params.handlePipelineSelect,
    setNodes,
    pipelineSchemas,
    params.handleNodeParameterChange,
    streams.localStream,
    streams.remoteStream,
    availability.spoutAvailable,
    availability.ndiAvailable,
    availability.syphonAvailable,
    availability.spoutOutputAvailable,
    availability.ndiOutputAvailable,
    availability.syphonOutputAvailable,
  ]);

  // Reroute type sync
  useRerouteTypeSync(edges, nodesRef, setNodes, setEdges);

  // Persistence
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
  });

  return {
    // Core state
    nodes,
    setNodes,
    onNodesChange,
    edges,
    setEdges,
    onEdgesChange,
    selectedNodeIds,
    setSelectedNodeIds,

    // Pipeline data
    availablePipelineIds,
    portsMap,
    pipelineSchemas,

    // Params
    nodeParams: params.nodeParams,
    handlePipelineSelect: params.handlePipelineSelect,
    handleNodeParameterChange: params.handleNodeParameterChange,
    handlePromptChange: params.handlePromptChange,
    handlePromptSubmit: params.handlePromptSubmit,
    resolveBackendId: params.resolveBackendId,
    isStreamingRef,
    onNodeParamChangeRef: params.onNodeParamChangeRef,
    onOutputSinkChangeRef,

    // Edge management
    handleEdgeDelete,

    // Persistence
    status: persistence.status,
    graphSource: persistence.graphSource,
    fitViewTrigger: persistence.fitViewTrigger,
    handleSave: persistence.handleSave,
    handleClear: persistence.handleClear,
    handleImport: persistence.handleImport,
    handleExport: persistence.handleExport,
    refreshGraph: persistence.refreshGraph,
    getCurrentGraphConfig: persistence.getCurrentGraphConfig,
    getGraphNodePrompts: persistence.getGraphNodePrompts,
  };
}
