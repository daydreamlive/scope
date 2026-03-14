import {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
} from "react";
import {
  ReactFlow,
  Controls,
  Background,
  BackgroundVariant,
} from "@xyflow/react";
import type { Edge, Node, ReactFlowInstance } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import {
  Camera,
  Workflow,
  Monitor,
  SlidersHorizontal,
  Trash2,
  Type,
  Hash,
  ToggleLeft,
  Sigma,
  StickyNote,
  Send,
  Gauge,
  CircleDot,
  Grid2x2,
  ListOrdered,
  GitBranch,
  Image,
  Sparkles,
  Lock,
  LockOpen,
  Pin,
  PinOff,
  Music,
  FolderOpen,
  PackageOpen,
} from "lucide-react";

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
import { MidiNode } from "./nodes/MidiNode";
import { BoolNode } from "./nodes/BoolNode";
import { SubgraphNode } from "./nodes/SubgraphNode";
import { SubgraphInputNode } from "./nodes/SubgraphInputNode";
import { SubgraphOutputNode } from "./nodes/SubgraphOutputNode";
import { CustomEdge } from "./CustomEdge";
import { ContextMenu } from "./ContextMenu";
import { AddNodeModal } from "./AddNodeModal";
import { BreadcrumbNav } from "./BreadcrumbNav";
import { NODE_TOKENS } from "./ui";
import type { FlowNodeData } from "../../lib/graphUtils";
import { parseHandleId } from "../../lib/graphUtils";
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

import { useGraphState } from "./hooks/useGraphState";
import { useConnectionLogic } from "./hooks/useConnectionLogic";
import { useNodeFactories } from "./hooks/useNodeFactories";
import { useValueForwarding } from "./hooks/useValueForwarding";
import { useOutputSinkSync } from "./hooks/useOutputSinkSync";
import { useKeyboardShortcuts } from "./hooks/useKeyboardShortcuts";
import {
  useGraphNavigation,
  BOUNDARY_INPUT_ID,
  BOUNDARY_OUTPUT_ID,
} from "./hooks/useGraphNavigation";
import { useParentValueBridge } from "./hooks/useParentValueBridge";
import { useSubgraphEval } from "./hooks/useSubgraphEval";

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
  midi: MidiNode,
  bool: BoolNode,
  subgraph: SubgraphNode,
  subgraph_input: SubgraphInputNode,
  subgraph_output: SubgraphOutputNode,
};

const edgeTypes = {
  default: CustomEdge,
};

export interface GraphEditorHandle {
  refreshGraph: () => void;
  getCurrentGraphConfig: () => import("../../lib/api").GraphConfig;
  getGraphNodePrompts: () => Array<{ nodeId: string; text: string }>;
}

interface GraphEditorProps {
  isStreaming?: boolean;
  isConnecting?: boolean;
  isLoading?: boolean;
  onNodeParameterChange?: (nodeId: string, key: string, value: unknown) => void;
  onGraphChange?: () => void;
  onGraphClear?: () => void;
  localStream?: MediaStream | null;
  remoteStream?: MediaStream | null;
  onVideoFileUpload?: (file: File) => Promise<boolean>;
  onStartStream?: () => void;
  onStopStream?: () => void;
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
  spoutOutputAvailable?: boolean;
  ndiOutputAvailable?: boolean;
  syphonOutputAvailable?: boolean;
}

export const GraphEditor = forwardRef<GraphEditorHandle, GraphEditorProps>(
  function GraphEditor(
    {
      isStreaming = false,
      isConnecting = false,
      isLoading = false,
      onNodeParameterChange,
      onGraphChange,
      onGraphClear,
      localStream,
      remoteStream,
      onVideoFileUpload,
      onStartStream,
      onStopStream,
      onSourceModeChange,
      spoutAvailable = false,
      ndiAvailable = false,
      syphonAvailable = false,
      onSpoutSourceChange,
      onNdiSourceChange,
      onSyphonSourceChange,
      onOutputSinkChange,
      spoutOutputAvailable = false,
      ndiOutputAvailable = false,
      syphonOutputAvailable = false,
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
      enrichDepsRef,
      handleEdgeDelete,
      resolveBackendId,
      isStreamingRef,
      onNodeParamChangeRef,
      onOutputSinkChangeRef,
      handleClear,
      handleSave,
      handleImport,
      handleExport,
      refreshGraph,
      getCurrentGraphConfig,
      getGraphNodePrompts,
      fitViewTrigger,
    } = useGraphState(
      {
        onNodeParameterChange,
        onGraphChange,
        onGraphClear,
        onVideoFileUpload,
        onSourceModeChange,
        onSpoutSourceChange,
        onNdiSourceChange,
        onSyphonSourceChange,
        onOutputSinkChange,
      },
      { localStream, remoteStream, isStreaming },
      {
        spoutAvailable,
        ndiAvailable,
        syphonAvailable,
        spoutOutputAvailable,
        ndiOutputAvailable,
        syphonOutputAvailable,
      },
      resolveRootGraphRef,
      resetNavigationRef
    );

    useImperativeHandle(
      ref,
      () => ({ refreshGraph, getCurrentGraphConfig, getGraphNodePrompts }),
      [refreshGraph, getCurrentGraphConfig, getGraphNodePrompts]
    );

    const [contextMenu, setContextMenu] = useState<{
      x: number;
      y: number;
      type: "pane" | "node";
      nodeId?: string;
    } | null>(null);

    const [showAddNodeModal, setShowAddNodeModal] = useState(false);
    const [showClearConfirm, setShowClearConfirm] = useState(false);
    const [pendingNodePosition, setPendingNodePosition] = useState<{
      x: number;
      y: number;
    } | null>(null);

    const reactFlowInstanceRef = useRef<ReactFlowInstance<
      Node<FlowNodeData>,
      Edge
    > | null>(null);

    const [selectionRect, setSelectionRect] = useState<{
      x1: number;
      y1: number;
      x2: number;
      y2: number;
    } | null>(null);

    const handleRightMouseDown = useCallback(
      (e: React.MouseEvent) => {
        if (e.button !== 2) return; // only right-click

        const startX = e.clientX;
        const startY = e.clientY;
        const startTarget = e.target as HTMLElement;
        let isDrag = false;

        setContextMenu(null);

        const handleMove = (me: MouseEvent) => {
          const dx = me.clientX - startX;
          const dy = me.clientY - startY;
          if (!isDrag && Math.sqrt(dx * dx + dy * dy) > 5) {
            isDrag = true;
          }
          if (isDrag) {
            setSelectionRect({
              x1: startX,
              y1: startY,
              x2: me.clientX,
              y2: me.clientY,
            });
          }
        };

        const handleUp = (me: MouseEvent) => {
          window.removeEventListener("mousemove", handleMove);
          window.removeEventListener("mouseup", handleUp);

          if (isDrag) {
            const rf = reactFlowInstanceRef.current;
            if (rf) {
              const start = rf.screenToFlowPosition({ x: startX, y: startY });
              const end = rf.screenToFlowPosition({
                x: me.clientX,
                y: me.clientY,
              });

              const minX = Math.min(start.x, end.x);
              const maxX = Math.max(start.x, end.x);
              const minY = Math.min(start.y, end.y);
              const maxY = Math.max(start.y, end.y);

              setNodes(nds =>
                nds.map(n => {
                  const w = n.measured?.width ?? n.width ?? 200;
                  const h = n.measured?.height ?? n.height ?? 100;
                  const overlaps =
                    n.position.x < maxX &&
                    n.position.x + w > minX &&
                    n.position.y < maxY &&
                    n.position.y + h > minY;
                  return n.selected === overlaps
                    ? n
                    : { ...n, selected: overlaps };
                })
              );
            }
          } else {
            const nodeEl = startTarget.closest(".react-flow__node");
            if (nodeEl) {
              const nodeId = nodeEl.getAttribute("data-id");
              if (nodeId) {
                setContextMenu({
                  x: startX,
                  y: startY,
                  type: "node",
                  nodeId,
                });
              }
            } else {
              const rf = reactFlowInstanceRef.current;
              if (rf) {
                const position = rf.screenToFlowPosition({
                  x: startX,
                  y: startY,
                });
                setPendingNodePosition(position);
                setContextMenu({
                  x: startX,
                  y: startY,
                  type: "pane",
                });
              }
            }
          }

          setSelectionRect(null);
        };

        window.addEventListener("mousemove", handleMove);
        window.addEventListener("mouseup", handleUp);
      },
      [setNodes, setPendingNodePosition]
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
      onConnect,
      onReconnect,
      findConnectedPipelineParams,
    } = useConnectionLogic(
      nodes,
      setNodes,
      setEdges,
      handleEdgeDelete,
      addSubgraphPortRef
    );

    const {
      handleNodeTypeSelect,
      handleDeleteNodes,
      createSubgraphFromSelection,
      unpackSubgraph,
    } = useNodeFactories({
      nodes,
      setNodes,
      setEdges,
      availablePipelineIds,
      portsMap,
      handlePipelineSelect,
      selectedNodeIds,
      setSelectedNodeIds,
      spoutOutputAvailable,
      ndiOutputAvailable,
      syphonOutputAvailable,
      pendingNodePosition,
      setPendingNodePosition,
    });

    useValueForwarding(
      nodes,
      edges,
      findConnectedPipelineParams,
      resolveBackendId,
      isStreamingRef,
      onNodeParamChangeRef,
      setNodes
    );

    useOutputSinkSync(nodes, onOutputSinkChangeRef);
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
    useSubgraphEval(nodes, edges, setNodes);
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

    const enterSubgraphRef = useRef(handleEnterSubgraph);
    enterSubgraphRef.current = handleEnterSubgraph;

    const stableEnterSubgraph = useCallback(
      (nodeId: string) => enterSubgraphRef.current(nodeId),
      []
    );
    const hasSubgraphNeedingCallback = nodes.some(
      n =>
        n.data.nodeType === "subgraph" &&
        n.data.onEnterSubgraph !== stableEnterSubgraph
    );
    useEffect(() => {
      if (!hasSubgraphNeedingCallback) return;
      setNodes(nds =>
        nds.map(n => {
          if (n.data.nodeType !== "subgraph") return n;
          if (n.data.onEnterSubgraph === stableEnterSubgraph) return n;
          return {
            ...n,
            data: { ...n.data, onEnterSubgraph: stableEnterSubgraph },
          };
        })
      );
    }, [hasSubgraphNeedingCallback, stableEnterSubgraph, setNodes]);

    const renameInputRef = useRef(
      (oldName: string, newName: string, portType: string) =>
        renameSubgraphPort(
          "input",
          oldName,
          newName,
          portType,
          setNodes,
          setEdges
        )
    );
    renameInputRef.current = (oldName, newName, portType) =>
      renameSubgraphPort(
        "input",
        oldName,
        newName,
        portType,
        setNodes,
        setEdges
      );

    const renameOutputRef = useRef(
      (oldName: string, newName: string, portType: string) =>
        renameSubgraphPort(
          "output",
          oldName,
          newName,
          portType,
          setNodes,
          setEdges
        )
    );
    renameOutputRef.current = (oldName, newName, portType) =>
      renameSubgraphPort(
        "output",
        oldName,
        newName,
        portType,
        setNodes,
        setEdges
      );

    const stableRenameInput = useCallback(
      (oldName: string, newName: string, portType: string) =>
        renameInputRef.current(oldName, newName, portType),
      []
    );
    const stableRenameOutput = useCallback(
      (oldName: string, newName: string, portType: string) =>
        renameOutputRef.current(oldName, newName, portType),
      []
    );

    const hasBoundaryNeedingRename = nodes.some(
      n =>
        (n.id === BOUNDARY_INPUT_ID &&
          n.data.onPortRename !== stableRenameInput) ||
        (n.id === BOUNDARY_OUTPUT_ID &&
          n.data.onPortRename !== stableRenameOutput)
    );
    useEffect(() => {
      if (!hasBoundaryNeedingRename) return;
      setNodes(nds =>
        nds.map(n => {
          if (
            n.id === BOUNDARY_INPUT_ID &&
            n.data.onPortRename !== stableRenameInput
          ) {
            return {
              ...n,
              data: { ...n.data, onPortRename: stableRenameInput },
            };
          }
          if (
            n.id === BOUNDARY_OUTPUT_ID &&
            n.data.onPortRename !== stableRenameOutput
          ) {
            return {
              ...n,
              data: { ...n.data, onPortRename: stableRenameOutput },
            };
          }
          return n;
        })
      );
    }, [
      hasBoundaryNeedingRename,
      stableRenameInput,
      stableRenameOutput,
      setNodes,
    ]);

    useEffect(() => {
      const currentInputHandles = new Set<string>();
      const currentOutputHandles = new Set<string>();
      for (const e of edges) {
        if (e.source === BOUNDARY_INPUT_ID) {
          const parsed = parseHandleId(e.sourceHandle);
          if (parsed && parsed.name !== "__add__")
            currentInputHandles.add(parsed.name);
        }
        if (e.target === BOUNDARY_OUTPUT_ID) {
          const parsed = parseHandleId(e.targetHandle);
          if (parsed && parsed.name !== "__add__")
            currentOutputHandles.add(parsed.name);
        }
      }

      const inputBoundary = nodes.find(n => n.id === BOUNDARY_INPUT_ID);
      const outputBoundary = nodes.find(n => n.id === BOUNDARY_OUTPUT_ID);

      if (inputBoundary) {
        const ports = inputBoundary.data.subgraphInputs ?? [];
        for (const port of ports) {
          if (
            !currentInputHandles.has(port.name) &&
            !hasExternalConnection("input", port.name, port.portType)
          ) {
            removeSubgraphPort("input", port.name, setNodes);
          }
        }
      }
      if (outputBoundary) {
        const ports = outputBoundary.data.subgraphOutputs ?? [];
        for (const port of ports) {
          if (
            !currentOutputHandles.has(port.name) &&
            !hasExternalConnection("output", port.name, port.portType)
          ) {
            removeSubgraphPort("output", port.name, setNodes);
          }
        }
      }
    }, [edges, nodes, removeSubgraphPort, hasExternalConnection, setNodes]);

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

    const fileInputRef = useRef<HTMLInputElement>(null);

    return (
      <div className="flex h-full w-full">
        <div className="flex flex-col flex-1">
          <div className={NODE_TOKENS.toolbar}>
            <button
              onClick={isStreaming ? onStopStream : onStartStream}
              disabled={isConnecting || isLoading}
              className={`${NODE_TOKENS.toolbarButton} ${isConnecting || isLoading ? "opacity-50 cursor-not-allowed" : ""}`}
              title={isStreaming ? "Stop stream" : "Start stream"}
            >
              {isConnecting || isLoading ? (
                <span className="inline-flex items-center gap-1">
                  <svg
                    className="animate-spin h-3 w-3"
                    viewBox="0 0 24 24"
                    fill="none"
                  >
                    <circle
                      className="opacity-25"
                      cx="12"
                      cy="12"
                      r="10"
                      stroke="currentColor"
                      strokeWidth="4"
                    />
                    <path
                      className="opacity-75"
                      fill="currentColor"
                      d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
                    />
                  </svg>
                </span>
              ) : isStreaming ? (
                <svg
                  className="h-3.5 w-3.5"
                  viewBox="0 0 24 24"
                  fill="currentColor"
                >
                  <rect x="4" y="4" width="16" height="16" rx="2" />
                </svg>
              ) : (
                <svg
                  className="h-3.5 w-3.5"
                  viewBox="0 0 24 24"
                  fill="currentColor"
                >
                  <polygon points="5,3 19,12 5,21" />
                </svg>
              )}
            </button>
            {isStreaming && (
              <button
                onClick={onStopStream}
                className={NODE_TOKENS.toolbarButton}
                title="Stop and clear"
              >
                <svg
                  className="h-3.5 w-3.5"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <path d="M1 4v6h6" />
                  <path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10" />
                </svg>
              </button>
            )}

            <div className="flex-1" />

            {status && (
              <span className={NODE_TOKENS.toolbarStatus}>{status}</span>
            )}

            <button
              onClick={handleSave}
              className={NODE_TOKENS.toolbarButton}
              title="Save graph (Ctrl+S)"
            >
              Save
            </button>

            <input
              ref={fileInputRef}
              type="file"
              accept=".json"
              onChange={handleImport}
              className="hidden"
            />
            <button
              onClick={() => fileInputRef.current?.click()}
              className={NODE_TOKENS.toolbarButton}
            >
              Import
            </button>
            <button
              onClick={handleExport}
              className={NODE_TOKENS.toolbarButton}
            >
              Export
            </button>
            <button
              onClick={() => setShowClearConfirm(true)}
              className={NODE_TOKENS.toolbarButton}
              title="Clear graph"
            >
              Clear
            </button>
          </div>

          <BreadcrumbNav
            path={breadcrumbPath}
            onNavigate={handleBreadcrumbNavigate}
          />

          <div className="flex-1 relative" onMouseDown={handleRightMouseDown}>
            <ReactFlow
              nodes={nodes}
              edges={edges}
              onNodesChange={onNodesChange}
              onEdgesChange={onEdgesChange}
              onConnect={onConnect}
              onReconnect={onReconnect}
              reconnectRadius={25}
              isValidConnection={isValidConnection}
              minZoom={0.1}
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
              deleteKeyCode={["Backspace", "Delete"]}
            >
              <Controls />
              <Background variant={BackgroundVariant.Dots} gap={16} size={1} />
            </ReactFlow>

            {contextMenu && (
              <ContextMenu
                x={contextMenu.x}
                y={contextMenu.y}
                onClose={() => setContextMenu(null)}
                header={contextMenu.type === "pane" ? "Create" : undefined}
                items={
                  contextMenu.type === "pane"
                    ? [
                        {
                          label: "Source",
                          icon: <Camera />,
                          onClick: () => handleNodeTypeSelect("source"),
                          keywords: ["input", "camera", "video"],
                        },
                        {
                          label: "Pipeline",
                          icon: <Workflow />,
                          onClick: () => handleNodeTypeSelect("pipeline"),
                          keywords: ["process", "effect", "filter"],
                        },
                        {
                          label: "Sink",
                          icon: <Monitor />,
                          onClick: () => handleNodeTypeSelect("sink"),
                          keywords: ["output", "display", "preview"],
                        },
                        {
                          label: "Output",
                          icon: <Send />,
                          onClick: () => handleNodeTypeSelect("output"),
                          keywords: ["spout", "ndi", "syphon", "send"],
                        },
                        {
                          label: "Controls",
                          icon: <SlidersHorizontal />,
                          children: [
                            {
                              label: "FloatControl",
                              icon: <Gauge />,
                              onClick: () =>
                                handleNodeTypeSelect("control", "float"),
                              keywords: ["float", "animated", "sine"],
                            },
                            {
                              label: "IntControl",
                              icon: <Hash />,
                              onClick: () =>
                                handleNodeTypeSelect("control", "int"),
                              keywords: ["integer", "animated"],
                            },
                            {
                              label: "StringControl",
                              icon: <Type />,
                              onClick: () =>
                                handleNodeTypeSelect("control", "string"),
                              keywords: ["text", "cycle", "animated"],
                            },
                            {
                              label: "MIDI",
                              icon: <Music />,
                              onClick: () => handleNodeTypeSelect("midi"),
                              keywords: [
                                "midi",
                                "controller",
                                "cc",
                                "knob",
                                "fader",
                              ],
                            },
                          ],
                        },
                        {
                          label: "UI",
                          icon: <CircleDot />,
                          children: [
                            {
                              label: "Slider",
                              icon: <SlidersHorizontal />,
                              onClick: () => handleNodeTypeSelect("slider"),
                              keywords: ["range", "value"],
                            },
                            {
                              label: "Knobs",
                              icon: <CircleDot />,
                              onClick: () => handleNodeTypeSelect("knobs"),
                              keywords: ["dial", "rotary"],
                            },
                            {
                              label: "XY Pad",
                              icon: <Grid2x2 />,
                              onClick: () => handleNodeTypeSelect("xypad"),
                              keywords: ["pad", "2d", "touch"],
                            },
                            {
                              label: "Tuple",
                              icon: <ListOrdered />,
                              onClick: () => handleNodeTypeSelect("tuple"),
                              keywords: ["list", "numbers", "array"],
                            },
                          ],
                        },
                        {
                          label: "Utility",
                          icon: <Sigma />,
                          children: [
                            {
                              label: "Math",
                              icon: <Sigma />,
                              onClick: () => handleNodeTypeSelect("math"),
                              keywords: ["add", "multiply", "arithmetic"],
                            },
                            {
                              label: "Note",
                              icon: <StickyNote />,
                              onClick: () => handleNodeTypeSelect("note"),
                              keywords: ["comment", "annotation", "text"],
                            },
                            {
                              label: "Bool",
                              icon: <ToggleLeft />,
                              onClick: () => handleNodeTypeSelect("bool"),
                              keywords: [
                                "boolean",
                                "gate",
                                "toggle",
                                "switch",
                                "on",
                                "off",
                              ],
                            },
                            {
                              label: "Reroute",
                              icon: <GitBranch />,
                              onClick: () => handleNodeTypeSelect("reroute"),
                              keywords: ["passthrough", "wire", "dot"],
                            },
                          ],
                        },
                        {
                          label: "Media",
                          icon: <Image />,
                          onClick: () => handleNodeTypeSelect("image"),
                          keywords: [
                            "media",
                            "image",
                            "video",
                            "picture",
                            "photo",
                            "reference",
                            "film",
                          ],
                        },
                        {
                          label: "VACE",
                          icon: <Sparkles />,
                          onClick: () => handleNodeTypeSelect("vace"),
                          keywords: [
                            "vace",
                            "conditioning",
                            "reference",
                            "frame",
                          ],
                        },
                        {
                          label: "Primitive",
                          icon: <ToggleLeft />,
                          onClick: () => handleNodeTypeSelect("primitive"),
                          keywords: ["value", "string", "number", "boolean"],
                        },
                        {
                          label: "Subgraph",
                          icon: <FolderOpen />,
                          onClick: () => handleNodeTypeSelect("subgraph"),
                          keywords: ["group", "container", "nest", "bundle"],
                        },
                        ...(selectedNodeIds.length > 0
                          ? [
                              {
                                label: `Group ${selectedNodeIds.length} node${selectedNodeIds.length !== 1 ? "s" : ""} into Subgraph`,
                                icon: <PackageOpen />,
                                onClick: () => {
                                  createSubgraphFromSelection(
                                    nodes,
                                    edges,
                                    selectedNodeIds
                                  );
                                },
                                keywords: [
                                  "create",
                                  "subgraph",
                                  "group",
                                  "selection",
                                ],
                              },
                            ]
                          : []),
                      ]
                    : (() => {
                        const clickedId = contextMenu.nodeId;
                        const isInSelection =
                          !!clickedId && selectedNodeIds.includes(clickedId);
                        const targetIds =
                          isInSelection && selectedNodeIds.length > 1
                            ? selectedNodeIds
                            : clickedId
                              ? [clickedId]
                              : [];
                        const count = targetIds.length;
                        const targetNodes = nodes.filter(n =>
                          targetIds.includes(n.id)
                        );
                        const allLocked = targetNodes.every(
                          n => !!n.data.locked
                        );
                        const allPinned = targetNodes.every(
                          n => !!n.data.pinned
                        );
                        const isSingleSubgraph =
                          count === 1 &&
                          targetNodes[0]?.data.nodeType === "subgraph";
                        const canCreateSubgraph =
                          count >= 1 &&
                          !targetNodes.every(
                            n => n.data.nodeType === "subgraph"
                          );

                        return [
                          ...(isSingleSubgraph
                            ? [
                                {
                                  label: "Enter Subgraph",
                                  icon: <FolderOpen />,
                                  onClick: () =>
                                    handleEnterSubgraph(targetIds[0]),
                                },
                                {
                                  label: "Unpack Subgraph",
                                  icon: <PackageOpen />,
                                  onClick: () =>
                                    unpackSubgraph(targetIds[0], nodes, edges),
                                },
                              ]
                            : []),
                          ...(canCreateSubgraph
                            ? [
                                {
                                  label: `Group into Subgraph`,
                                  icon: <PackageOpen />,
                                  onClick: () =>
                                    createSubgraphFromSelection(
                                      nodes,
                                      edges,
                                      targetIds
                                    ),
                                },
                              ]
                            : []),
                          {
                            label: allLocked
                              ? count > 1
                                ? `Unlock ${count} nodes`
                                : "Unlock"
                              : count > 1
                                ? `Lock ${count} nodes`
                                : "Lock",
                            icon: allLocked ? <LockOpen /> : <Lock />,
                            onClick: () => {
                              const newLocked = !allLocked;
                              setNodes(nds =>
                                nds.map(n =>
                                  targetIds.includes(n.id)
                                    ? {
                                        ...n,
                                        data: {
                                          ...n.data,
                                          locked: newLocked,
                                        },
                                      }
                                    : n
                                )
                              );
                            },
                          },
                          {
                            label: allPinned
                              ? count > 1
                                ? `Unpin ${count} nodes`
                                : "Unpin"
                              : count > 1
                                ? `Pin ${count} nodes`
                                : "Pin",
                            icon: allPinned ? <PinOff /> : <Pin />,
                            onClick: () => {
                              const newPinned = !allPinned;
                              setNodes(nds =>
                                nds.map(n =>
                                  targetIds.includes(n.id)
                                    ? {
                                        ...n,
                                        draggable: !newPinned,
                                        data: {
                                          ...n.data,
                                          pinned: newPinned,
                                        },
                                      }
                                    : n
                                )
                              );
                            },
                          },
                          {
                            label:
                              count > 1 ? `Delete ${count} nodes` : "Delete",
                            icon: <Trash2 />,
                            onClick: () => handleDeleteNodes(targetIds),
                            danger: true,
                          },
                        ];
                      })()
                }
              />
            )}

            <AddNodeModal
              open={showAddNodeModal}
              onClose={() => {
                setShowAddNodeModal(false);
                setPendingNodePosition(null);
              }}
              onSelectNodeType={handleNodeTypeSelect}
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
        </div>
      </div>
    );
  }
);
