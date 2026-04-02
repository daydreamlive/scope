import {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
} from "react";
import { toast } from "sonner";
import { Plus } from "lucide-react";
import {
  ReactFlow,
  Controls,
  Background,
  BackgroundVariant,
  SelectionMode,
} from "@xyflow/react";
import type {
  Edge,
  Node,
  NodeChange,
  EdgeChange,
  ReactFlowInstance,
  FinalConnectionState,
  HandleType,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";

import { SourceNode } from "./nodes/SourceNode";
import { PipelineNode } from "./nodes/PipelineNode";
import { SinkNode } from "./nodes/SinkNode";
import { PrimitiveNode } from "./nodes/PrimitiveNode";
import { RerouteNode } from "./nodes/RerouteNode";
import { ControlNode } from "./nodes/ControlNode";
import { MathNode } from "./nodes/MathNode";
import { NoteNode } from "./nodes/NoteNode";
import { OutputNode } from "./nodes/OutputNode";
import { SliderNode } from "./nodes/SliderNode";
import { KnobsNode } from "./nodes/KnobsNode";
import { XYPadNode } from "./nodes/XYPadNode";
import { TupleNode } from "./nodes/TupleNode";
import { ImageNode } from "./nodes/ImageNode";
import { VaceNode } from "./nodes/VaceNode";
import { LoraNode } from "./nodes/LoraNode";
import { MidiNode } from "./nodes/MidiNode";
import { BoolNode } from "./nodes/BoolNode";
import { TriggerNode } from "./nodes/TriggerNode";
import { SubgraphNode } from "./nodes/SubgraphNode";
import { SubgraphInputNode } from "./nodes/SubgraphInputNode";
import { SubgraphOutputNode } from "./nodes/SubgraphOutputNode";
import { RecordNode } from "./nodes/RecordNode";
import { TempoNode } from "./nodes/TempoNode";
import { PromptListNode } from "./nodes/PromptListNode";
import { PromptBlendNode } from "./nodes/PromptBlendNode";
import { SchedulerNode } from "./nodes/SchedulerNode";
import { CustomEdge } from "./CustomEdge";
import { ContextMenu } from "./ContextMenu";
import { AddNodeModal } from "./AddNodeModal";
import { BlueprintBrowserModal } from "./BlueprintBrowserModal";
import { BreadcrumbNav } from "./BreadcrumbNav";
import { GraphToolbar } from "./GraphToolbar";
import { GraphWorkflowImportDialog } from "./GraphWorkflowImportDialog";
import { GraphWorkflowExportDialog } from "./GraphWorkflowExportDialog";
import { ExportDialog } from "../ExportDialog";
import {
  isAuthenticated as checkIsAuthenticated,
  getDaydreamAPIKey,
  redirectToSignIn,
} from "../../lib/auth";
import { createDaydreamImportSession } from "../../lib/daydreamExport";
import { openExternalUrl } from "../../lib/openExternal";
import { buildPaneMenuItems, buildNodeMenuItems } from "./contextMenuItems";
import type { FlowNodeData } from "../../lib/graphUtils";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "../ui/alert-dialog";

import { useRightClickSelect } from "./hooks/ui/useRightClickSelect";
import { useGraphState } from "./hooks/graph/useGraphState";
import { useConnectionLogic } from "./hooks/connection/useConnectionLogic";
import { useNodeFactories } from "./hooks/node/useNodeFactories";
import { useValueForwarding } from "./hooks/value/useValueForwarding";
import { useOutputSinkSync } from "./hooks/value/useOutputSinkSync";
import { useKeyboardShortcuts } from "./hooks/graph/useKeyboardShortcuts";
import { useGraphNavigation } from "./hooks/subgraph/useGraphNavigation";
import { useParentValueBridge } from "./hooks/value/useParentValueBridge";
import { useSubgraphEval } from "./hooks/subgraph/useSubgraphEval";
import { useSubgraphCallbackSync } from "./hooks/subgraph/useSubgraphCallbackSync";
import { useSubgraphOperations } from "./hooks/subgraph/useSubgraphOperations";

const nodeTypes = {
  source: SourceNode,
  pipeline: PipelineNode,
  sink: SinkNode,
  primitive: PrimitiveNode,
  control: ControlNode,
  math: MathNode,
  note: NoteNode,
  output: OutputNode,
  slider: SliderNode,
  knobs: KnobsNode,
  xypad: XYPadNode,
  tuple: TupleNode,
  reroute: RerouteNode,
  image: ImageNode,
  vace: VaceNode,
  lora: LoraNode,
  midi: MidiNode,
  bool: BoolNode,
  trigger: TriggerNode,
  subgraph: SubgraphNode,
  subgraph_input: SubgraphInputNode,
  subgraph_output: SubgraphOutputNode,
  record: RecordNode,
  tempo: TempoNode,
  prompt_list: PromptListNode,
  prompt_blend: PromptBlendNode,
  scheduler: SchedulerNode,
};

const edgeTypes = {
  default: CustomEdge,
};

export interface GraphEditorHandle {
  refreshGraph: () => void;
  getCurrentGraphConfig: () => import("../../lib/api").GraphConfig;
  getGraphNodePrompts: () => Array<{ nodeId: string; text: string }>;
  getGraphVaceSettings: () => Array<{
    pipelineNodeId: string;
    vace_context_scale: number;
    vace_use_input_video: boolean;
    vace_ref_images?: string[];
    first_frame_image?: string;
    last_frame_image?: string;
  }>;
  getGraphLoRASettings: () => Array<{
    pipelineNodeId: string;
    loras: Array<{ path: string; scale: number; merge_mode?: string }>;
    lora_merge_mode: string;
  }>;
  loadWorkflow: (
    workflow: import("../../lib/workflowApi").ScopeWorkflow
  ) => void;
  updateNodeParam: (nodeId: string, key: string, value: unknown) => void;
  clearGraph: () => void;
}

interface GraphEditorProps {
  visible?: boolean;
  isStreaming?: boolean;
  isConnecting?: boolean;
  isLoading?: boolean;
  loadingStage?: string | null;
  onNodeParameterChange?: (nodeId: string, key: string, value: unknown) => void;
  onGraphChange?: () => void;
  onGraphClear?: () => void;
  localStream?: MediaStream | null;
  remoteStream?: MediaStream | null;
  onVideoFileUpload?: (file: File) => Promise<boolean>;
  onCycleSampleVideo?: () => void;
  isPlaying?: boolean;
  onStartStream?: () => void;
  onStopStream?: () => void;
  onPlayPauseToggle?: () => void;
  onSourceModeChange?: (mode: string) => void;
  spoutAvailable?: boolean;
  ndiAvailable?: boolean;
  syphonAvailable?: boolean;
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
  spoutOutputAvailable?: boolean;
  ndiOutputAvailable?: boolean;
  syphonOutputAvailable?: boolean;
  onStartRecording?: () => void;
  onStopRecording?: () => void;
  onOpenSettings?: () => void;
  onOpenPlugins?: () => void;
  tempoState?: import("../../hooks/useTempoSync").TempoState;
  tempoSources?: import("../../lib/api").TempoSourcesResponse | null;
  tempoLoading?: boolean;
  tempoError?: string | null;
  onEnableTempo?: (req: import("../../lib/api").TempoEnableRequest) => void;
  onDisableTempo?: () => void;
  onSetTempo?: (bpm: number) => void;
  onRefreshTempoSources?: () => void;
}

export const GraphEditor = forwardRef<GraphEditorHandle, GraphEditorProps>(
  function GraphEditor(
    {
      visible = true,
      isStreaming = false,
      isConnecting = false,
      isLoading = false,
      loadingStage = null,
      onNodeParameterChange,
      onGraphChange,
      onGraphClear,
      localStream,
      remoteStream,
      onVideoFileUpload,
      onCycleSampleVideo,
      isPlaying = true,
      onStartStream,
      onStopStream,
      onPlayPauseToggle,
      onSourceModeChange,
      spoutAvailable = false,
      ndiAvailable = false,
      syphonAvailable = false,
      onSpoutSourceChange,
      onNdiSourceChange,
      onSyphonSourceChange,
      onOutputSinkChange,
      onOutputSinkBulkChange,
      spoutOutputAvailable = false,
      ndiOutputAvailable = false,
      syphonOutputAvailable = false,
      onStartRecording,
      onStopRecording,
      onOpenSettings,
      onOpenPlugins,
      tempoState,
      tempoSources,
      tempoLoading,
      tempoError,
      onEnableTempo,
      onDisableTempo,
      onSetTempo,
      onRefreshTempoSources,
    },
    ref
  ) {
    const resolveRootGraphRef = useRef<
      (
        nodes: Node<FlowNodeData>[],
        edges: Edge[]
      ) => { nodes: Node<FlowNodeData>[]; edges: Edge[] }
    >((n, e) => ({ nodes: n, edges: e }));
    const resetNavigationRef = useRef<() => void>(() => {});
    const {
      nodes,
      setNodes,
      onNodesChange,
      edges,
      setEdges,
      onEdgesChange,
      status,
      availablePipelineIds,
      portsMap,
      selectedNodeIds,
      setSelectedNodeIds,
      handlePipelineSelect,
      handleNodeParameterChange,
      enrichDepsRef,
      handleEdgeDelete,
      resolveBackendId,
      onNodeParamChangeRef,
      onOutputSinkBulkChangeRef,
      handleClear,
      handleSave,
      handleImport,
      buildCurrentWorkflow,
      refreshGraph,
      getCurrentGraphConfig,
      getGraphNodePrompts,
      getGraphVaceSettings,
      getGraphLoRASettings,
      fitViewTrigger,
      pendingImportWorkflow,
      pendingResolutionPlan,
      pendingImportResolving,
      confirmImport,
      cancelImport,
      reResolveImport,
      loadGraphFromParsed,
      initialLoadDone,
    } = useGraphState(
      {
        onNodeParameterChange,
        onGraphChange,
        onGraphClear,
        onVideoFileUpload,
        onCycleSampleVideo,
        onSourceModeChange,
        onSpoutSourceChange,
        onNdiSourceChange,
        onSyphonSourceChange,
        onOutputSinkChange,
        onOutputSinkBulkChange,
        onStartRecording,
        onStopRecording,
        onEnableTempo,
        onDisableTempo,
        onSetTempo,
        onRefreshTempoSources,
      },
      {
        localStream,
        remoteStream,
        isStreaming,
        isLoading,
        loadingStage,
        isPlaying,
        onPlayPauseToggle,
      },
      {
        spoutAvailable,
        ndiAvailable,
        syphonAvailable,
        spoutOutputAvailable,
        ndiOutputAvailable,
        syphonOutputAvailable,
      },
      {
        tempoState,
        tempoSources,
        tempoLoading,
        tempoError,
      },
      resolveRootGraphRef,
      resetNavigationRef
    );

    useImperativeHandle(
      ref,
      () => ({
        refreshGraph,
        getCurrentGraphConfig,
        getGraphNodePrompts,
        getGraphVaceSettings,
        getGraphLoRASettings,
        loadWorkflow: (
          workflow: import("../../lib/workflowApi").ScopeWorkflow
        ) => {
          loadGraphFromParsed(
            workflow as unknown as Record<string, unknown>,
            workflow.metadata?.name ?? "workflow"
          );
        },
        updateNodeParam: handleNodeParameterChange,
        clearGraph: handleClear,
      }),
      [
        refreshGraph,
        getCurrentGraphConfig,
        getGraphNodePrompts,
        getGraphVaceSettings,
        getGraphLoRASettings,
        loadGraphFromParsed,
        handleNodeParameterChange,
        handleClear,
      ]
    );

    const [showAddNodeModal, setShowAddNodeModal] = useState(false);
    const [showBlueprintModal, setShowBlueprintModal] = useState(false);
    const [showClearConfirm, setShowClearConfirm] = useState(false);
    const [showDefaultConfirm, setShowDefaultConfirm] = useState(false);
    const [showExportDialog, setShowExportDialog] = useState(false);
    const [showWorkflowExport, setShowWorkflowExport] = useState(false);
    const [isDaydreamAuthenticated, setIsDaydreamAuthenticated] = useState(
      checkIsAuthenticated()
    );
    const [isExportingToDaydream, setIsExportingToDaydream] = useState(false);

    useEffect(() => {
      const handleAuthChange = () => {
        setIsDaydreamAuthenticated(checkIsAuthenticated());
      };
      window.addEventListener("daydream-auth-change", handleAuthChange);
      return () => {
        window.removeEventListener("daydream-auth-change", handleAuthChange);
      };
    }, []);

    const handleExportToDaydream = useCallback(async () => {
      if (!isDaydreamAuthenticated) {
        redirectToSignIn();
        return;
      }

      const apiKey = getDaydreamAPIKey();
      if (!apiKey) {
        toast.error("Not authenticated with Daydream");
        return;
      }

      const isElectron = Boolean(
        (window as unknown as { scope?: { openExternal?: unknown } }).scope
          ?.openExternal
      );
      const pendingTab = isElectron
        ? null
        : window.open("about:blank", "_blank");

      setIsExportingToDaydream(true);
      try {
        const workflow = buildCurrentWorkflow("Untitled Workflow");

        const result = await createDaydreamImportSession(
          apiKey,
          workflow,
          workflow.metadata.name
        );

        if (pendingTab) {
          pendingTab.location.href = result.createUrl;
        } else {
          openExternalUrl(result.createUrl);
        }
        toast.success("Opening daydream.live...", {
          description:
            "Your workflow has been sent to daydream.live for publishing.",
        });
        setShowExportDialog(false);
      } catch (err) {
        pendingTab?.close();
        console.error("Export to daydream.live failed:", err);
        toast.error("Export failed", {
          description: err instanceof Error ? err.message : String(err),
        });
      } finally {
        setIsExportingToDaydream(false);
      }
    }, [isDaydreamAuthenticated, buildCurrentWorkflow]);
    const [pendingNodePosition, setPendingNodePosition] = useState<{
      x: number;
      y: number;
    } | null>(null);

    const reactFlowInstanceRef = useRef<ReactFlowInstance<
      Node<FlowNodeData>,
      Edge
    > | null>(null);

    const { selectionRect, contextMenu, setContextMenu, handleRightMouseDown } =
      useRightClickSelect(
        reactFlowInstanceRef,
        setNodes,
        setPendingNodePosition
      );

    const handleOpenCreateMenu = useCallback(
      (screenX: number, screenY: number) => {
        const rf = reactFlowInstanceRef.current;
        if (!rf) return;
        const flowPosition = rf.screenToFlowPosition({
          x: screenX,
          y: screenY,
        });
        setPendingNodePosition(flowPosition);
        setContextMenu({ x: screenX, y: screenY, type: "pane" });
      },
      [setContextMenu]
    );

    const addSubgraphPortRef = useRef<
      | ((
          side: "input" | "output",
          port: import("../../lib/graphUtils").SubgraphPort,
          setNodes: (
            updater: (nds: Node<FlowNodeData>[]) => Node<FlowNodeData>[]
          ) => void
        ) => string | null)
      | null
    >(null);

    const {
      isValidConnection,
      onConnect: rawOnConnect,
      onReconnect: rawOnReconnect,
      findConnectedPipelineParams,
    } = useConnectionLogic(
      nodes,
      setNodes,
      setEdges,
      handleEdgeDelete,
      addSubgraphPortRef
    );

    const filteredOnNodesChange = useCallback(
      (changes: NodeChange<Node<FlowNodeData>>[]) => {
        if (isStreaming) {
          changes = changes.filter(
            c => c.type !== "remove" && c.type !== "add"
          );
        }
        onNodesChange(changes);
      },
      [isStreaming, onNodesChange]
    );

    const filteredOnEdgesChange = useCallback(
      (changes: EdgeChange<Edge>[]) => {
        if (isStreaming) {
          changes = changes.filter(
            c => c.type !== "remove" && c.type !== "add"
          );
        }
        onEdgesChange(changes);
      },
      [isStreaming, onEdgesChange]
    );

    const onConnect = useCallback(
      (...args: Parameters<typeof rawOnConnect>) => {
        if (isStreaming) return;
        rawOnConnect(...args);
      },
      [isStreaming, rawOnConnect]
    );

    const onReconnect = useCallback(
      (...args: Parameters<typeof rawOnReconnect>) => {
        if (isStreaming) return;
        rawOnReconnect(...args);
      },
      [isStreaming, rawOnReconnect]
    );

    const { handleNodeTypeSelect, handleDeleteNodes, insertBlueprint } =
      useNodeFactories({
        nodes,
        setNodes,
        setEdges,
        availablePipelineIds,
        portsMap,
        handlePipelineSelect,
        setSelectedNodeIds,
        spoutOutputAvailable,
        ndiOutputAvailable,
        syphonOutputAvailable,
        pendingNodePosition,
        setPendingNodePosition,
        handleEdgeDelete,
        enrichDepsRef,
      });

    const { createSubgraphFromSelection, unpackSubgraph } =
      useSubgraphOperations({
        nodes,
        setNodes,
        setEdges,
        setSelectedNodeIds,
      });

    useValueForwarding(
      nodes,
      edges,
      findConnectedPipelineParams,
      resolveBackendId,
      isStreaming,
      onNodeParamChangeRef,
      setNodes
    );

    useOutputSinkSync(nodes, onOutputSinkBulkChangeRef);
    useKeyboardShortcuts(
      reactFlowInstanceRef,
      setPendingNodePosition,
      setShowAddNodeModal,
      nodes,
      edges,
      setNodes,
      setEdges,
      handleSave
    );

    const {
      depth: navDepth,
      breadcrumbPath,
      enterSubgraph,
      navigateTo: navNavigateTo,
      addSubgraphPort,
      removeSubgraphPort,
      renameSubgraphPort,
      hasExternalConnection,
      getRootGraph,
      resetStack,
      stackRef: navStackRef,
    } = useGraphNavigation();

    useParentValueBridge(navStackRef, navDepth, setNodes);
    useSubgraphEval(nodes, edges, setNodes, visible);

    resolveRootGraphRef.current = getRootGraph;
    resetNavigationRef.current = resetStack;
    addSubgraphPortRef.current = addSubgraphPort;
    const nodesRef = useRef(nodes);
    const edgesRef = useRef(edges);
    nodesRef.current = nodes;
    edgesRef.current = edges;

    const applyViewport = useCallback(
      (viewport: ReturnType<typeof enterSubgraph>) => {
        setTimeout(() => {
          const rf = reactFlowInstanceRef.current;
          if (!rf) return;
          if (viewport) {
            rf.setViewport(viewport, { duration: 300 });
          } else {
            rf.fitView({ padding: 0.1, duration: 300 });
          }
        }, 50);
      },
      []
    );

    const handleEnterSubgraph = useCallback(
      (nodeId: string) => {
        const rf = reactFlowInstanceRef.current;
        const currentViewport = rf?.getViewport();
        const targetViewport = enterSubgraph(
          nodeId,
          nodesRef.current,
          edgesRef.current,
          setNodes,
          setEdges,
          enrichDepsRef.current,
          handleEdgeDelete,
          currentViewport
        );
        applyViewport(targetViewport);
      },
      [
        enterSubgraph,
        setNodes,
        setEdges,
        enrichDepsRef,
        handleEdgeDelete,
        applyViewport,
      ]
    );

    const handleBreadcrumbNavigate = useCallback(
      (targetDepth: number) => {
        const rf = reactFlowInstanceRef.current;
        const currentViewport = rf?.getViewport();
        const targetViewport = navNavigateTo(
          targetDepth,
          nodesRef.current,
          edgesRef.current,
          setNodes,
          setEdges,
          enrichDepsRef.current,
          handleEdgeDelete,
          currentViewport
        );
        applyViewport(targetViewport);
      },
      [
        navNavigateTo,
        setNodes,
        setEdges,
        enrichDepsRef,
        handleEdgeDelete,
        applyViewport,
      ]
    );

    useSubgraphCallbackSync({
      nodes,
      edges,
      setNodes,
      setEdges,
      handleEnterSubgraph,
      renameSubgraphPort,
      removeSubgraphPort,
      hasExternalConnection,
    });

    const prevHadSourceRef = useRef(false);
    const prevHadSinkRef = useRef(false);

    useEffect(() => {
      const hasSource = nodes.some(n => n.data.nodeType === "source");
      const hasSink = nodes.some(n => n.data.nodeType === "sink");

      if (
        isStreaming &&
        ((prevHadSourceRef.current && !hasSource) ||
          (prevHadSinkRef.current && !hasSink))
      ) {
        onStopStream?.();
      }

      prevHadSourceRef.current = hasSource;
      prevHadSinkRef.current = hasSink;
    }, [nodes, isStreaming, onStopStream]);

    useEffect(() => {
      if (fitViewTrigger === 0) return;
      const timer = setTimeout(() => {
        reactFlowInstanceRef.current?.fitView({ padding: 0.1, duration: 300 });
      }, 50);
      return () => clearTimeout(timer);
    }, [fitViewTrigger]);

    const suppressContextMenu = useCallback(
      (event: MouseEvent | React.MouseEvent<Element, MouseEvent>) => {
        event.preventDefault();
      },
      []
    );

    const suppressNodeContextMenu = useCallback(
      (event: React.MouseEvent, _node: Node<FlowNodeData>) => {
        event.preventDefault();
      },
      []
    );

    // Double-click on empty canvas opens pane context menu
    const handleWrapperDoubleClick = useCallback(
      (event: React.MouseEvent) => {
        if (isStreaming) return;
        const target = event.target as HTMLElement;
        if (target.closest(".react-flow__node")) return;
        const rf = reactFlowInstanceRef.current;
        if (!rf) return;
        const position = rf.screenToFlowPosition({
          x: event.clientX,
          y: event.clientY,
        });
        setPendingNodePosition(position);
        setContextMenu({
          x: event.clientX,
          y: event.clientY,
          type: "pane",
        });
      },
      [isStreaming, setContextMenu]
    );

    // Track connection drag for noodle-drop context menu
    const connectStartRef = useRef<{
      nodeId: string;
      handleId: string | null;
      handleType: string | null;
    } | null>(null);

    const handleConnectStart = useCallback(
      (
        _event: MouseEvent | TouchEvent,
        params: {
          nodeId: string | null;
          handleId: string | null;
          handleType: string | null;
        }
      ) => {
        connectStartRef.current = params.nodeId
          ? {
              nodeId: params.nodeId,
              handleId: params.handleId ?? null,
              handleType: params.handleType ?? null,
            }
          : null;
      },
      []
    );

    const handleConnectEnd = useCallback(
      (
        event: MouseEvent | TouchEvent,
        connectionState: FinalConnectionState
      ) => {
        if (isStreaming || isReconnectingRef.current) {
          connectStartRef.current = null;
          return;
        }
        if (
          connectStartRef.current &&
          !connectionState.isValid &&
          !connectionState.toNode
        ) {
          const rf = reactFlowInstanceRef.current;
          if (rf) {
            const clientX =
              "clientX" in event
                ? event.clientX
                : event.changedTouches[0].clientX;
            const clientY =
              "clientY" in event
                ? event.clientY
                : event.changedTouches[0].clientY;
            const position = rf.screenToFlowPosition({
              x: clientX,
              y: clientY,
            });
            setPendingNodePosition(position);
            setContextMenu({
              x: clientX,
              y: clientY,
              type: "pane",
            });
          }
        }
        connectStartRef.current = null;
      },
      [isStreaming, setContextMenu]
    );

    // Track reconnect state so dropping on canvas deletes the edge
    const isReconnectingRef = useRef(false);
    const reconnectingEdgeRef = useRef<string | null>(null);
    const reconnectSucceededRef = useRef(false);

    const handleReconnectStart = useCallback(
      (_event: React.MouseEvent, edge: Edge, _handleType: HandleType) => {
        isReconnectingRef.current = true;
        reconnectingEdgeRef.current = edge.id;
        reconnectSucceededRef.current = false;
      },
      []
    );

    const wrappedOnReconnect = useCallback(
      (...args: Parameters<typeof onReconnect>) => {
        reconnectSucceededRef.current = true;
        onReconnect(...args);
      },
      [onReconnect]
    );

    const handleReconnectEnd = useCallback(
      (
        _event: MouseEvent | TouchEvent,
        _edge: Edge,
        _handleType: HandleType,
        _connectionState: FinalConnectionState
      ) => {
        if (
          reconnectingEdgeRef.current &&
          !reconnectSucceededRef.current &&
          !isStreaming
        ) {
          handleEdgeDelete(reconnectingEdgeRef.current);
        }
        reconnectingEdgeRef.current = null;
        reconnectSucceededRef.current = false;
        isReconnectingRef.current = false;
      },
      [isStreaming, handleEdgeDelete]
    );

    return (
      <div className="flex h-full w-full">
        <div className="flex flex-col flex-1">
          <GraphToolbar
            isStreaming={isStreaming}
            isConnecting={isConnecting}
            isLoading={isLoading}
            loadingStage={loadingStage}
            status={status}
            onStartStream={onStartStream}
            onStopStream={onStopStream}
            onImport={handleImport}
            onExport={() => setShowExportDialog(true)}
            onClear={() => setShowClearConfirm(true)}
            onDefaultWorkflow={() => setShowDefaultConfirm(true)}
            onOpenSettings={onOpenSettings}
            onOpenPlugins={onOpenPlugins}
          />

          <BreadcrumbNav
            path={breadcrumbPath}
            onNavigate={handleBreadcrumbNavigate}
          />

          <div
            className={`flex-1 relative${isStreaming ? " streaming" : ""}`}
            onMouseDown={handleRightMouseDown}
            onDoubleClick={handleWrapperDoubleClick}
            onContextMenu={e => e.preventDefault()}
          >
            <ReactFlow
              nodes={nodes}
              edges={edges}
              onNodesChange={filteredOnNodesChange}
              onEdgesChange={filteredOnEdgesChange}
              onConnect={onConnect}
              onReconnect={wrappedOnReconnect}
              onReconnectStart={handleReconnectStart}
              onReconnectEnd={handleReconnectEnd}
              onConnectStart={handleConnectStart}
              onConnectEnd={handleConnectEnd}
              reconnectRadius={25}
              nodesConnectable={!isStreaming}
              isValidConnection={isValidConnection}
              minZoom={0.1}
              zoomOnDoubleClick={false}
              panActivationKeyCode="Space"
              multiSelectionKeyCode={["Meta", "Control"]}
              selectionMode={SelectionMode.Partial}
              onInit={instance => {
                reactFlowInstanceRef.current = instance;
              }}
              onSelectionChange={({ nodes: selected }) =>
                setSelectedNodeIds(prev => {
                  const next = selected.map(n => n.id);
                  if (
                    next.length === prev.length &&
                    next.every((id, i) => id === prev[i])
                  )
                    return prev;
                  return next;
                })
              }
              onPaneClick={() => setContextMenu(null)}
              onPaneContextMenu={suppressContextMenu}
              onNodeContextMenu={suppressNodeContextMenu}
              nodeTypes={nodeTypes}
              edgeTypes={edgeTypes}
              colorMode="dark"
              fitView
              deleteKeyCode={isStreaming ? [] : ["Backspace", "Delete"]}
            >
              <Controls />
              <Background variant={BackgroundVariant.Dots} gap={16} size={1} />
            </ReactFlow>

            {/* Add node button — upper right of canvas */}
            {!isStreaming && (
              <button
                onClick={e =>
                  handleOpenCreateMenu(e.clientX, e.clientY)
                }
                className="absolute top-4 right-4 z-30 w-12 h-12 rounded-xl border-2 border-dashed border-[rgba(119,119,119,0.4)] bg-[rgba(17,17,17,0.6)] hover:border-[rgba(119,119,119,0.7)] hover:bg-[rgba(17,17,17,0.8)] transition-colors cursor-pointer flex items-center justify-center"
                title="Add node"
              >
                <Plus className="h-5 w-5 text-[#8c8c8d]" />
              </button>
            )}

            {/* Empty state placeholder */}
            {nodes.length === 0 &&
              !isStreaming &&
              initialLoadDone.current && (
                <div className="absolute inset-0 flex items-center justify-center z-20 pointer-events-none">
                  <button
                    onClick={e =>
                      handleOpenCreateMenu(e.clientX, e.clientY)
                    }
                    className="pointer-events-auto flex flex-col items-center gap-3 cursor-pointer group"
                  >
                    <div className="w-28 h-28 rounded-xl border-2 border-dashed border-[rgba(119,119,119,0.3)] bg-[rgba(17,17,17,0.3)] flex items-center justify-center group-hover:border-[rgba(119,119,119,0.5)] transition-colors">
                      <Plus className="h-8 w-8 text-[#555]" />
                    </div>
                    <span className="text-sm text-[#555] group-hover:text-[#777] transition-colors">
                      Add first step…
                    </span>
                  </button>
                </div>
              )}

            {contextMenu && !isStreaming && (
              <ContextMenu
                x={contextMenu.x}
                y={contextMenu.y}
                onClose={() => setContextMenu(null)}
                header={contextMenu.type === "pane" ? "Create" : undefined}
                items={
                  contextMenu.type === "pane"
                    ? buildPaneMenuItems({
                        handleNodeTypeSelect,
                        selectedNodeIds,
                        nodes,
                        edges,
                        createSubgraphFromSelection,
                        onOpenBlueprints: () => setShowBlueprintModal(true),
                      })
                    : buildNodeMenuItems({
                        contextNodeId: contextMenu.nodeId!,
                        selectedNodeIds,
                        nodes,
                        edges,
                        setNodes,
                        handleDeleteNodes,
                        handleEnterSubgraph,
                        unpackSubgraph,
                        createSubgraphFromSelection,
                      })
                }
              />
            )}

            <AddNodeModal
              open={showAddNodeModal && !isStreaming}
              onClose={() => {
                setShowAddNodeModal(false);
                setPendingNodePosition(null);
              }}
              onSelectNodeType={handleNodeTypeSelect}
            />

            <BlueprintBrowserModal
              open={showBlueprintModal && !isStreaming}
              onClose={() => setShowBlueprintModal(false)}
              onInsert={blueprint => {
                insertBlueprint(blueprint, pendingNodePosition ?? undefined);
                setPendingNodePosition(null);
              }}
            />

            {selectionRect && (
              <div
                style={{
                  position: "fixed",
                  left: Math.min(selectionRect.x1, selectionRect.x2),
                  top: Math.min(selectionRect.y1, selectionRect.y2),
                  width: Math.abs(selectionRect.x2 - selectionRect.x1),
                  height: Math.abs(selectionRect.y2 - selectionRect.y1),
                  border: "1px solid rgba(59, 130, 246, 0.5)",
                  backgroundColor: "rgba(59, 130, 246, 0.08)",
                  pointerEvents: "none",
                  zIndex: 9999,
                }}
              />
            )}
          </div>

          <AlertDialog
            open={showClearConfirm}
            onOpenChange={(open: boolean) => {
              if (!open) setShowClearConfirm(false);
            }}
          >
            <AlertDialogContent className="sm:max-w-md">
              <AlertDialogHeader>
                <AlertDialogTitle>Clear graph?</AlertDialogTitle>
                <AlertDialogDescription>
                  This will remove all nodes and connections from the graph.
                  This action cannot be undone.
                </AlertDialogDescription>
              </AlertDialogHeader>
              <AlertDialogFooter>
                <AlertDialogCancel onClick={() => setShowClearConfirm(false)}>
                  Cancel
                </AlertDialogCancel>
                <AlertDialogAction
                  onClick={() => {
                    setShowClearConfirm(false);
                    if (isStreaming) onStopStream?.();
                    handleClear();
                  }}
                >
                  Clear
                </AlertDialogAction>
              </AlertDialogFooter>
            </AlertDialogContent>
          </AlertDialog>

          <AlertDialog
            open={showDefaultConfirm}
            onOpenChange={(open: boolean) => {
              if (!open) setShowDefaultConfirm(false);
            }}
          >
            <AlertDialogContent className="sm:max-w-md">
              <AlertDialogHeader>
                <AlertDialogTitle>Reset to default workflow?</AlertDialogTitle>
                <AlertDialogDescription>
                  This will replace the current graph with the default Source,
                  Passthrough, Sink workflow. This action cannot be undone.
                </AlertDialogDescription>
              </AlertDialogHeader>
              <AlertDialogFooter>
                <AlertDialogCancel onClick={() => setShowDefaultConfirm(false)}>
                  Cancel
                </AlertDialogCancel>
                <AlertDialogAction
                  onClick={() => {
                    setShowDefaultConfirm(false);
                    if (isStreaming) onStopStream?.();
                    loadGraphFromParsed(
                      {
                        nodes: [
                          { id: "input", type: "source", source_mode: "video" },
                          {
                            id: "passthrough",
                            type: "pipeline",
                            pipeline_id: "passthrough",
                          },
                          { id: "output", type: "sink" },
                        ],
                        edges: [
                          {
                            from: "input",
                            from_port: "video",
                            to_node: "passthrough",
                            to_port: "video",
                          },
                          {
                            from: "passthrough",
                            from_port: "video",
                            to_node: "output",
                            to_port: "video",
                          },
                        ],
                      },
                      "default"
                    );
                  }}
                >
                  Reset
                </AlertDialogAction>
              </AlertDialogFooter>
            </AlertDialogContent>
          </AlertDialog>

          <GraphWorkflowImportDialog
            workflow={pendingImportWorkflow}
            plan={pendingResolutionPlan}
            resolving={pendingImportResolving}
            onConfirm={confirmImport}
            onCancel={cancelImport}
            onReResolve={reResolveImport}
          />

          <ExportDialog
            open={showExportDialog}
            onClose={() => setShowExportDialog(false)}
            onSaveGeneration={() => {}}
            onSaveTimeline={() => {
              setShowExportDialog(false);
              setShowWorkflowExport(true);
            }}
            onExportToDaydream={handleExportToDaydream}
            isRecording={false}
            isAuthenticated={isDaydreamAuthenticated}
            isExportingToDaydream={isExportingToDaydream}
          />

          <GraphWorkflowExportDialog
            open={showWorkflowExport}
            onClose={() => setShowWorkflowExport(false)}
            buildWorkflow={buildCurrentWorkflow}
          />
        </div>
      </div>
    );
  }
);
