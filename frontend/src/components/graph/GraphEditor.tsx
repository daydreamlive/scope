import {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useMemo,
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
  Connection,
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
import { AudioNode } from "./nodes/AudioNode";
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
import { CustomNode } from "./nodes/CustomNode";
import { CustomEdge } from "./CustomEdge";
import { ContextMenu, type ContextMenuItem } from "./ContextMenu";
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
import { OscConfigDialog } from "./OscConfigDialog";
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
import { useOscInventory } from "./hooks/graph/useOscInventory";
import { useConnectionLogic } from "./hooks/connection/useConnectionLogic";
import { useNodeFactories } from "./hooks/node/useNodeFactories";
import { useValueForwarding } from "./hooks/value/useValueForwarding";
import {
  useKeyboardShortcuts,
  type KeyboardShortcutHandlers,
} from "./hooks/graph/useKeyboardShortcuts";
import { useGraphHistory } from "./hooks/graph/useGraphHistory";
import { KeyboardShortcutsDialog } from "./KeyboardShortcutsDialog";
import { useGraphNavigation } from "./hooks/subgraph/useGraphNavigation";
import { useParentValueBridge } from "./hooks/value/useParentValueBridge";
import { useSubgraphEval } from "./hooks/subgraph/useSubgraphEval";
import { useSubgraphCallbackSync } from "./hooks/subgraph/useSubgraphCallbackSync";
import { useSubgraphOperations } from "./hooks/subgraph/useSubgraphOperations";
import { useNodeDefinitions } from "../../hooks/useNodeDefinitions";
import { usePipelinesContext } from "../../contexts/PipelinesContext";

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
  audio: AudioNode,
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
  custom_node: CustomNode,
};

const edgeTypes = {
  default: CustomEdge,
};

type ConnectionDragStart = {
  nodeId: string;
  handleId: string | null;
  handleType: HandleType | null;
};

type ConnectionMenuState = {
  x: number;
  y: number;
  header: string;
  items: ContextMenuItem[];
};

type PendingConnectionCreate = {
  start: ConnectionDragStart;
};

function getClientPoint(event: MouseEvent | TouchEvent) {
  if ("clientX" in event) return { x: event.clientX, y: event.clientY };
  const touch = event.changedTouches[0] ?? event.touches[0];
  return touch ? { x: touch.clientX, y: touch.clientY } : null;
}

function readHandleElement(element: Element | null) {
  const handleEl = element?.closest<HTMLElement>(".react-flow__handle");
  if (!handleEl) return null;
  const nodeId =
    handleEl.dataset.nodeid ??
    handleEl.getAttribute("data-nodeid") ??
    handleEl.getAttribute("data-node-id");
  const handleId =
    handleEl.dataset.handleid ??
    handleEl.getAttribute("data-handleid") ??
    handleEl.getAttribute("data-handle-id");
  const handleType = handleEl.classList.contains("source")
    ? "source"
    : handleEl.classList.contains("target")
      ? "target"
      : null;

  if (!nodeId || !handleId || !handleType) return null;
  return { element: handleEl, nodeId, handleId, handleType };
}

function getHandleMenuLabel(handleId: string) {
  const rawName = handleId.includes(":")
    ? handleId.split(":").slice(1).join(":")
    : handleId;
  return rawName.replace(/^in:/, "").replace(/^out:/, "").replace(/_/g, " ");
}

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
  applyExternalParams: (
    params: Record<string, unknown>,
    targetNodeId?: string
  ) => void;
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
  localStreams?: Record<string, MediaStream>;
  remoteStream?: MediaStream | null;
  remoteStreams?: Record<string, MediaStream>;
  sinkStats?: Record<string, { fps: number; bitrate: number }>;
  onVideoFileUpload?: (file: File, nodeId?: string) => Promise<boolean>;
  onCycleSampleVideo?: (nodeId?: string) => void;
  onInitSampleVideo?: (nodeId?: string) => void;
  isPlaying?: boolean;
  onStartStream?: () => void;
  onStopStream?: () => void;
  onPlayPauseToggle?: () => void;
  onSourceModeChange?: (mode: string, nodeId?: string) => void;
  spoutAvailable?: boolean;
  ndiAvailable?: boolean;
  syphonAvailable?: boolean;
  availableInputSources?: import("../../lib/api").InputSourceType[];
  spoutReason?: string | null;
  ndiReason?: string | null;
  syphonReason?: string | null;
  onSpoutSourceChange?: (name: string) => void;
  onNdiSourceChange?: (identifier: string) => void;
  onSyphonSourceChange?: (identifier: string) => void;
  onOutputSinkChange?: (
    sinkType: string,
    config: { enabled: boolean; name: string }
  ) => void;

  spoutOutputAvailable?: boolean;
  ndiOutputAvailable?: boolean;
  syphonOutputAvailable?: boolean;
  onStartRecording?: (nodeId?: string) => void;
  onStopRecording?: (nodeId?: string) => void;
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
      localStreams,
      remoteStream,
      remoteStreams,
      sinkStats,
      onVideoFileUpload,
      onCycleSampleVideo,
      onInitSampleVideo,
      isPlaying = true,
      onStartStream,
      onStopStream,
      onPlayPauseToggle,
      onSourceModeChange,
      spoutAvailable = false,
      ndiAvailable = false,
      syphonAvailable = false,
      availableInputSources = [],
      spoutReason = null,
      ndiReason = null,
      syphonReason = null,
      onSpoutSourceChange,
      onNdiSourceChange,
      onSyphonSourceChange,
      onOutputSinkChange,
      spoutOutputAvailable = false,
      ndiOutputAvailable = false,
      syphonOutputAvailable = false,
      onStartRecording,
      onStopRecording,
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
      handlePromptChange,
      applyExternalNodeParams,
      enrichDepsRef,
      handleEdgeDelete,
      resolveBackendId,
      onNodeParamChangeRef,
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
        onInitSampleVideo,
        onSourceModeChange,
        onSpoutSourceChange,
        onNdiSourceChange,
        onSyphonSourceChange,
        onOutputSinkChange,
        onStartRecording,
        onStopRecording,
        onEnableTempo,
        onDisableTempo,
        onSetTempo,
        onRefreshTempoSources,
      },
      {
        localStream,
        localStreams,
        remoteStream,
        remoteStreams,
        sinkStats,
        isStreaming,
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
        availableInputSources,
        spoutReason,
        ndiReason,
        syphonReason,
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

    const { undo, redo } = useGraphHistory(
      nodes,
      edges,
      setNodes,
      setEdges,
      enrichDepsRef,
      handleEdgeDelete
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
        applyExternalParams: applyExternalNodeParams,
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
        applyExternalNodeParams,
        handleClear,
      ]
    );

    const [showAddNodeModal, setShowAddNodeModal] = useState(false);
    const [showBlueprintModal, setShowBlueprintModal] = useState(false);
    const [showClearConfirm, setShowClearConfirm] = useState(false);
    const [showDefaultConfirm, setShowDefaultConfirm] = useState(false);
    const [showExportDialog, setShowExportDialog] = useState(false);
    const [showWorkflowExport, setShowWorkflowExport] = useState(false);
    const [showShortcutsDialog, setShowShortcutsDialog] = useState(false);
    // Per-node Configure OSC modal: stores the id of the node being configured.
    const [oscConfigNodeId, setOscConfigNodeId] = useState<string | null>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);
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
    const graphWrapperRef = useRef<HTMLDivElement>(null);
    const [connectionMenu, setConnectionMenu] =
      useState<ConnectionMenuState | null>(null);
    const pendingConnectionCreateRef = useRef<PendingConnectionCreate | null>(
      null
    );
    const suppressNextPaneClickRef = useRef(false);

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
        pendingConnectionCreateRef.current = null;
        setConnectionMenu(null);
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

    const isValidConnectionRef = useRef(isValidConnection);
    const onConnectRef = useRef(onConnect);
    useEffect(() => {
      isValidConnectionRef.current = isValidConnection;
      onConnectRef.current = onConnect;
    }, [isValidConnection, onConnect]);

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

    const connectPendingCreatedNode = useCallback(
      (newNodeId: string | null) => {
        const pending = pendingConnectionCreateRef.current;
        pendingConnectionCreateRef.current = null;
        if (!pending || !newNodeId) return;

        const { start } = pending;
        const oppositeHandleType =
          start.handleType === "source"
            ? "target"
            : start.handleType === "target"
              ? "source"
              : null;
        if (!start.handleId || !oppositeHandleType) return;

        const tryConnect = () => {
          const wrapper = graphWrapperRef.current;
          if (!wrapper) return;

          const handles = Array.from(
            wrapper.querySelectorAll<HTMLElement>(
              `.react-flow__handle.${oppositeHandleType}`
            )
          )
            .map(readHandleElement)
            .filter(
              (
                handle
              ): handle is NonNullable<ReturnType<typeof readHandleElement>> =>
                !!handle && handle.nodeId === newNodeId
            );

          for (const handle of handles) {
            const connection: Connection =
              start.handleType === "source"
                ? {
                    source: start.nodeId,
                    sourceHandle: start.handleId,
                    target: newNodeId,
                    targetHandle: handle.handleId,
                  }
                : {
                    source: newNodeId,
                    sourceHandle: handle.handleId,
                    target: start.nodeId,
                    targetHandle: start.handleId,
                  };

            if (!isValidConnectionRef.current(connection)) continue;
            onConnectRef.current(connection);
            return;
          }

          toast.info("Created node, but no compatible handle was found");
        };

        requestAnimationFrame(() => requestAnimationFrame(tryConnect));
      },
      []
    );

    const handleCreateNodeTypeSelect = useCallback(
      (...args: Parameters<typeof handleNodeTypeSelect>) => {
        const newNodeId = handleNodeTypeSelect(...args);
        connectPendingCreatedNode(newNodeId);
      },
      [handleNodeTypeSelect, connectPendingCreatedNode]
    );

    const { createSubgraphFromSelection, unpackSubgraph } =
      useSubgraphOperations({
        nodes,
        setNodes,
        setEdges,
        setSelectedNodeIds,
      });

    const { customNodes: availableCustomNodes } = useNodeDefinitions();
    const { pipelines } = usePipelinesContext();

    const handleDebugNodes = useCallback(() => {
      const DEBUG_NODES: Array<{
        id: string;
        type: string;
        nodeType: string;
        position: { x: number; y: number };
        extra?: Partial<FlowNodeData>;
      }> = [
        {
          id: "source",
          type: "source",
          nodeType: "source",
          position: { x: 50, y: 50 },
        },
        {
          id: "pipeline",
          type: "pipeline",
          nodeType: "pipeline",
          position: { x: 321.74, y: 49.33 },
        },
        {
          id: "sink",
          type: "sink",
          nodeType: "sink",
          position: { x: 577.54, y: 42.91 },
        },
        {
          id: "record",
          type: "record",
          nodeType: "record",
          position: { x: 584.35, y: 274.9 },
        },
        {
          id: "primitive",
          type: "primitive",
          nodeType: "primitive",
          position: { x: 586.99, y: 393.92 },
        },
        {
          id: "bool",
          type: "bool",
          nodeType: "bool",
          position: { x: 50, y: 350 },
        },
        {
          id: "slider",
          type: "slider",
          nodeType: "slider",
          position: { x: 43.29, y: 773.73 },
        },
        {
          id: "knobs",
          type: "knobs",
          nodeType: "knobs",
          position: { x: 39.43, y: 966.33 },
        },
        {
          id: "xypad",
          type: "xypad",
          nodeType: "xypad",
          position: { x: 850.46, y: 46.12 },
        },
        {
          id: "control",
          type: "control",
          nodeType: "control",
          position: { x: 856.59, y: 334.32 },
          extra: { controlType: "float" },
        },
        {
          id: "control_1",
          type: "control",
          nodeType: "control",
          position: { x: 47.15, y: 558.69 },
          extra: { controlType: "int" },
        },
        {
          id: "control_2",
          type: "control",
          nodeType: "control",
          position: { x: 318.9, y: 923.97 },
          extra: { controlType: "string" },
        },
        {
          id: "math",
          type: "math",
          nodeType: "math",
          position: { x: 586.16, y: 588.74 },
        },
        {
          id: "tuple",
          type: "tuple",
          nodeType: "tuple",
          position: { x: 857.59, y: 543.01 },
        },
        {
          id: "output",
          type: "output",
          nodeType: "output",
          position: { x: 860.87, y: 697.09 },
        },
        {
          id: "vace",
          type: "vace",
          nodeType: "vace",
          position: { x: 587.22, y: 914.44 },
        },
        {
          id: "lora",
          type: "lora",
          nodeType: "lora",
          position: { x: 586.18, y: 794.57 },
        },
        {
          id: "midi",
          type: "midi",
          nodeType: "midi",
          position: { x: 860.45, y: 860.13 },
        },
        {
          id: "trigger",
          type: "trigger",
          nodeType: "trigger",
          position: { x: 318.72, y: 817.36 },
        },
        {
          id: "tempo",
          type: "tempo",
          nodeType: "tempo",
          position: { x: 1121.76, y: 278.44 },
        },
        {
          id: "prompt_list",
          type: "prompt_list",
          nodeType: "prompt_list",
          position: { x: 1123.61, y: 479.6 },
        },
        {
          id: "prompt_blend",
          type: "prompt_blend",
          nodeType: "prompt_blend",
          position: { x: 1129.74, y: 709.3 },
        },
        {
          id: "scheduler",
          type: "scheduler",
          nodeType: "scheduler",
          position: { x: 1128.73, y: 883.37 },
        },
        {
          id: "note",
          type: "note",
          nodeType: "note",
          position: { x: 1122.5, y: 49.42 },
        },
        {
          id: "reroute",
          type: "reroute",
          nodeType: "reroute",
          position: { x: 971.33, y: 1063.07 },
        },
      ];

      const debugNodes: Node<FlowNodeData>[] = DEBUG_NODES.map(def => ({
        id: def.id,
        type: def.type,
        position: def.position,
        data: {
          label: def.id,
          nodeType: def.nodeType,
          ...def.extra,
        } as FlowNodeData,
      }));

      setNodes(debugNodes);
      setEdges([]);
    }, [setNodes, setEdges]);

    const onPromptForwardRef = useRef(handlePromptChange);
    onPromptForwardRef.current = handlePromptChange;

    useValueForwarding(
      nodes,
      edges,
      findConnectedPipelineParams,
      resolveBackendId,
      isStreaming,
      onNodeParamChangeRef,
      setNodes,
      onPromptForwardRef
    );

    const shortcutHandlers: KeyboardShortcutHandlers = useMemo(
      () => ({
        "zoom-in": () => reactFlowInstanceRef.current?.zoomIn(),
        "zoom-out": () => reactFlowInstanceRef.current?.zoomOut(),
        "zoom-reset": () =>
          reactFlowInstanceRef.current?.setViewport(
            { x: 0, y: 0, zoom: 1 },
            { duration: 300 }
          ),
        "fit-view": () =>
          reactFlowInstanceRef.current?.fitView({
            padding: 0.1,
            duration: 300,
          }),
        "fit-view-home": () =>
          reactFlowInstanceRef.current?.fitView({
            padding: 0.1,
            duration: 300,
          }),
        "open-add-node": () => {
          const rf = reactFlowInstanceRef.current;
          if (rf) {
            const vp = rf.getViewport();
            const wrapper = document.querySelector(".react-flow");
            const rect = wrapper?.getBoundingClientRect();
            if (rect) {
              setPendingNodePosition(
                rf.screenToFlowPosition({
                  x: rect.left + rect.width / 2,
                  y: rect.top + rect.height / 2,
                })
              );
            } else {
              setPendingNodePosition({
                x: -vp.x / vp.zoom,
                y: -vp.y / vp.zoom,
              });
            }
          }
          pendingConnectionCreateRef.current = null;
          setShowAddNodeModal(true);
        },
        undo,
        redo,
        save: handleSave,
        export: () => setShowExportDialog(true),
        "toggle-stream": () =>
          isStreaming ? onStopStream?.() : onStartStream?.(),
        "show-shortcuts": () => setShowShortcutsDialog(true),
        "select-all": () =>
          setNodes(nds => nds.map(n => ({ ...n, selected: true }))),
        deselect: () =>
          setNodes(nds =>
            nds.map(n => (n.selected ? { ...n, selected: false } : n))
          ),
        "lock-node": () =>
          setNodes(nds =>
            nds.map(n =>
              n.selected
                ? {
                    ...n,
                    draggable: n.draggable === false ? true : false,
                    data: {
                      ...n.data,
                      locked: !n.data.locked,
                    },
                  }
                : n
            )
          ),
        "pin-node": () =>
          setNodes(nds =>
            nds.map(n =>
              n.selected
                ? {
                    ...n,
                    data: {
                      ...n.data,
                      pinned: !n.data.pinned,
                    },
                  }
                : n
            )
          ),
        "group-nodes": () => {
          if (selectedNodeIds.length >= 2) {
            createSubgraphFromSelection(nodes, edges, selectedNodeIds);
          }
        },
      }),
      [
        undo,
        redo,
        handleSave,
        isStreaming,
        onStartStream,
        onStopStream,
        setNodes,
        selectedNodeIds,
        nodes,
        edges,
        createSubgraphFromSelection,
      ]
    );

    useKeyboardShortcuts({
      nodes,
      edges,
      setNodes,
      setEdges,
      isStreaming,
      handlers: shortcutHandlers,
    });

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
    // Push the user-curated OSC inventory to the backend whenever the
    // graph or any node's oscConfig overlay changes.
    useOscInventory(nodes);

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
        pendingConnectionCreateRef.current = null;
        setConnectionMenu(null);
        setContextMenu({
          x: event.clientX,
          y: event.clientY,
          type: "pane",
        });
      },
      [isStreaming, setContextMenu]
    );

    const getCompatibleConnectionMenuItems = useCallback(
      (start: ConnectionDragStart): ContextMenuItem[] => {
        const oppositeHandleType =
          start.handleType === "source"
            ? "target"
            : start.handleType === "target"
              ? "source"
              : null;
        if (
          !start.handleId ||
          !oppositeHandleType ||
          !graphWrapperRef.current
        ) {
          return [];
        }

        const nodeById = new Map(nodes.map(node => [node.id, node]));
        const items: ContextMenuItem[] = [];

        for (const handleEl of Array.from(
          graphWrapperRef.current.querySelectorAll<HTMLElement>(
            `.react-flow__handle.${oppositeHandleType}`
          )
        )) {
          const handle = readHandleElement(handleEl);
          if (!handle || handle.nodeId === start.nodeId) continue;

          const connection: Connection =
            start.handleType === "source"
              ? {
                  source: start.nodeId,
                  sourceHandle: start.handleId,
                  target: handle.nodeId,
                  targetHandle: handle.handleId,
                }
              : {
                  source: handle.nodeId,
                  sourceHandle: handle.handleId,
                  target: start.nodeId,
                  targetHandle: start.handleId,
                };

          if (!isValidConnection(connection)) continue;

          const node = nodeById.get(handle.nodeId);
          if (!node) continue;
          const nodeLabel = node.data.customTitle || node.data.label || node.id;
          const handleLabel = getHandleMenuLabel(handle.handleId);
          items.push({
            label: `${nodeLabel} (${handleLabel})`,
            keywords: [nodeLabel, node.id, handleLabel],
            onClick: () => onConnect(connection),
          });
        }

        return items;
      },
      [isValidConnection, nodes, onConnect]
    );

    const openConnectionMenu = useCallback(
      (start: ConnectionDragStart, point: { x: number; y: number }) => {
        const rf = reactFlowInstanceRef.current;
        const createItem: ContextMenuItem = {
          label: "Create Node...",
          icon: <Plus />,
          keywords: ["create", "new", "node"],
          onClick: () => {
            if (!rf) return;
            setPendingNodePosition(
              rf.screenToFlowPosition({ x: point.x, y: point.y })
            );
            pendingConnectionCreateRef.current = { start };
            setConnectionMenu(null);
            setContextMenu({ x: point.x, y: point.y, type: "pane" });
          },
        };
        const items = [createItem, ...getCompatibleConnectionMenuItems(start)];

        suppressNextPaneClickRef.current = true;
        window.setTimeout(() => {
          suppressNextPaneClickRef.current = false;
        }, 50);
        setContextMenu(null);
        setConnectionMenu({
          x: point.x,
          y: point.y,
          header:
            start.handleType === "source" ? "Connect Target" : "Connect Source",
          items,
        });
      },
      [getCompatibleConnectionMenuItems, setContextMenu]
    );

    // Track connection drag for noodle-drop context menu
    const connectStartRef = useRef<ConnectionDragStart | null>(null);

    const handleConnectStart = useCallback(
      (
        _event: MouseEvent | TouchEvent,
        params: {
          nodeId: string | null;
          handleId: string | null;
          handleType: HandleType | null;
        }
      ) => {
        setConnectionMenu(null);
        setContextMenu(null);
        connectStartRef.current = params.nodeId
          ? {
              nodeId: params.nodeId,
              handleId: params.handleId ?? null,
              handleType: params.handleType ?? null,
            }
          : null;
      },
      [setContextMenu]
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
        const start = connectStartRef.current;
        if (start && !connectionState.isValid) {
          const point = getClientPoint(event);
          if (point) openConnectionMenu(start, point);
        }
        connectStartRef.current = null;
      },
      [isStreaming, openConnectionMenu]
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

    // ComfyUI/Blueprints-style "grab existing edge from connected handle".
    //
    // React Flow's built-in reconnect (edgesReconnectable + onReconnect) only
    // triggers when the user grabs the edge endpoint hit-area near the handle
    // — clicking the Handle DOM itself starts a *new* connection because the
    // Handle's pointerdown wins event ordering. We intercept pointerdown in
    // the capture phase on the wrapper, and when a user click-drags a target
    // handle that already has an incoming edge we (a) swallow the event so
    // React Flow doesn't start a new connection, (b) wait until the cursor
    // moves past a small threshold so a stray click doesn't tear the edge,
    // (c) detach the existing edge, (d) synthesize a mousedown on the
    // opposite *source* handle so React Flow starts a connection drag from
    // there, and (e) immediately kickstart that drag with a synthesized
    // mousemove so the live connection line shows while the button is still
    // held — without this final step React Flow stays under the
    // connectionDragThreshold and falls into click-to-connect mode (line
    // only appears after release).
    useEffect(() => {
      const root = graphWrapperRef.current;
      if (!root) return;

      const PICKUP_THRESHOLD_SQ = 9; // 3 px
      const KICKSTART_DELTA = 4; // px past XYHandle's connectionDragThreshold

      const onPointerDownCapture = (event: PointerEvent) => {
        if (event.button !== 0) return;
        // Synthesized events (the ones we dispatch below) must pass through.
        if (!event.isTrusted) return;
        if (isStreaming) return;
        if (isReconnectingRef.current) return;

        const handle = readHandleElement(event.target as Element | null);
        if (!handle) return;
        // Only pick up from target handles. Source handles can fan out to
        // many edges, so the natural "start a new connection" behavior on
        // mousedown is preserved there.
        if (handle.handleType !== "target") return;

        const edge = edges.find(
          e => e.target === handle.nodeId && e.targetHandle === handle.handleId
        );
        if (!edge) return;

        const sourceHandleEl = root.querySelector<HTMLElement>(
          `.react-flow__handle.source[data-nodeid="${edge.source}"][data-handleid="${edge.sourceHandle ?? ""}"]`
        );
        if (!sourceHandleEl) return;

        event.stopPropagation();
        event.preventDefault();

        const startX = event.clientX;
        const startY = event.clientY;

        const cleanup = () => {
          document.removeEventListener("mousemove", onMove, true);
          document.removeEventListener("mouseup", onUp, true);
        };

        const triggerPickup = (clientX: number, clientY: number) => {
          cleanup();
          handleEdgeDelete(edge.id);

          const baseInit: MouseEventInit = {
            bubbles: true,
            cancelable: true,
            composed: true,
            button: 0,
            buttons: 1,
            view: window,
          };

          // Start a connection drag from the source handle anchored at the
          // user's original click position. React's delegated onMouseDown
          // picks this up, calling XYHandle.onPointerDown which attaches
          // mousemove/mouseup listeners on the document for the drag.
          sourceHandleEl.dispatchEvent(
            new MouseEvent("mousedown", {
              ...baseInit,
              clientX: startX,
              clientY: startY,
            })
          );

          // Push past XYHandle's connectionDragThreshold immediately so the
          // live connection line appears while the button is still held.
          // Without this, the user has to release before the line shows
          // (click-to-connect mode kicks in instead of drag mode).
          document.dispatchEvent(
            new MouseEvent("mousemove", {
              ...baseInit,
              clientX: clientX + KICKSTART_DELTA,
              clientY: clientY + KICKSTART_DELTA,
            })
          );
          // Settle the connection line at the user's actual cursor.
          document.dispatchEvent(
            new MouseEvent("mousemove", {
              ...baseInit,
              clientX,
              clientY,
            })
          );
        };

        const onMove = (e: MouseEvent) => {
          const dx = e.clientX - startX;
          const dy = e.clientY - startY;
          if (dx * dx + dy * dy < PICKUP_THRESHOLD_SQ) return;
          triggerPickup(e.clientX, e.clientY);
        };

        const onUp = () => {
          // Released without crossing the pickup threshold: leave the edge
          // intact and let React Flow's normal click handling proceed (a
          // tiny click on a connected handle is a no-op for us).
          cleanup();
        };

        document.addEventListener("mousemove", onMove, true);
        document.addEventListener("mouseup", onUp, true);
      };

      root.addEventListener("pointerdown", onPointerDownCapture, true);
      return () => {
        root.removeEventListener("pointerdown", onPointerDownCapture, true);
      };
    }, [edges, isStreaming, handleEdgeDelete]);

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
            onDebugNodes={handleDebugNodes}
            fileInputRef={fileInputRef}
          />

          <BreadcrumbNav
            path={breadcrumbPath}
            onNavigate={handleBreadcrumbNavigate}
          />

          <div
            ref={graphWrapperRef}
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
              edgesReconnectable={!isStreaming}
              reconnectRadius={35}
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
              onPaneClick={() => {
                if (suppressNextPaneClickRef.current) {
                  suppressNextPaneClickRef.current = false;
                  return;
                }
                setContextMenu(null);
                setConnectionMenu(null);
              }}
              onPaneContextMenu={suppressContextMenu}
              onNodeContextMenu={suppressNodeContextMenu}
              nodeTypes={nodeTypes}
              edgeTypes={edgeTypes}
              colorMode="dark"
              fitView
              deleteKeyCode={isStreaming ? [] : ["Backspace", "Delete"]}
            >
              <Controls />
              <Background
                variant={BackgroundVariant.Dots}
                gap={20}
                size={1.2}
                color="rgba(255,255,255,0.22)"
              />
            </ReactFlow>

            {/* Add node button — upper right of canvas */}
            {!isStreaming && (
              <button
                onClick={e => handleOpenCreateMenu(e.clientX, e.clientY)}
                className="absolute top-4 right-4 z-30 w-12 h-12 rounded-lg border-2 border-dashed border-[rgba(119,119,119,0.3)] bg-[rgba(17,17,17,0.6)] hover:border-[rgba(119,119,119,0.6)] hover:bg-[rgba(17,17,17,0.8)] transition-colors cursor-pointer flex items-center justify-center"
                title="Add node"
              >
                <Plus className="h-5 w-5 text-[#8c8c8d]" />
              </button>
            )}

            {/* Empty state placeholder */}
            {nodes.length === 0 && !isStreaming && initialLoadDone.current && (
              <div className="absolute inset-0 flex items-center justify-center z-20 pointer-events-none">
                <button
                  onClick={e => handleOpenCreateMenu(e.clientX, e.clientY)}
                  className="pointer-events-auto flex flex-col items-center gap-3 cursor-pointer group"
                >
                  <div className="w-28 h-28 rounded-lg border-2 border-dashed border-[rgba(119,119,119,0.2)] bg-[rgba(17,17,17,0.3)] flex items-center justify-center group-hover:border-[rgba(119,119,119,0.4)] transition-colors">
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
                onClose={() => {
                  setContextMenu(null);
                  pendingConnectionCreateRef.current = null;
                }}
                header={contextMenu.type === "pane" ? "Create" : undefined}
                items={
                  contextMenu.type === "pane"
                    ? buildPaneMenuItems({
                        handleNodeTypeSelect: handleCreateNodeTypeSelect,
                        selectedNodeIds,
                        nodes,
                        edges,
                        createSubgraphFromSelection,
                        onOpenBlueprints: () => setShowBlueprintModal(true),
                        availablePipelineIds,
                        pipelines,
                        availableInputSources,
                        customNodes: availableCustomNodes,
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
                        openOscConfig: id => setOscConfigNodeId(id),
                      })
                }
              />
            )}

            {connectionMenu && !isStreaming && (
              <ContextMenu
                x={connectionMenu.x}
                y={connectionMenu.y}
                onClose={() => setConnectionMenu(null)}
                header={connectionMenu.header}
                items={connectionMenu.items}
              />
            )}

            <AddNodeModal
              open={showAddNodeModal && !isStreaming}
              onClose={() => {
                setShowAddNodeModal(false);
                setPendingNodePosition(null);
              }}
              onSelectNodeType={handleNodeTypeSelect}
              availableInputSources={availableInputSources}
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

          <KeyboardShortcutsDialog
            open={showShortcutsDialog}
            onOpenChange={setShowShortcutsDialog}
          />

          <OscConfigDialog
            open={oscConfigNodeId !== null}
            onOpenChange={open => {
              if (!open) setOscConfigNodeId(null);
            }}
            node={
              oscConfigNodeId
                ? (nodes.find(n => n.id === oscConfigNodeId) ?? null)
                : null
            }
            onSave={oscConfig => {
              if (!oscConfigNodeId) return;
              const targetId = oscConfigNodeId;
              setNodes(nds =>
                nds.map(n =>
                  n.id === targetId
                    ? { ...n, data: { ...n.data, oscConfig } }
                    : n
                )
              );
            }}
          />
        </div>
      </div>
    );
  }
);
