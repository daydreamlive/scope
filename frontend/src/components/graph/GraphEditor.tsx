import {
  forwardRef,
  useCallback,
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
import { CustomEdge } from "./CustomEdge";
import { ContextMenu } from "./ContextMenu";
import { AddNodeModal } from "./AddNodeModal";
import { NODE_TOKENS } from "./ui";
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

import { useGraphState } from "./hooks/useGraphState";
import { useConnectionLogic } from "./hooks/useConnectionLogic";
import { useNodeFactories } from "./hooks/useNodeFactories";
import { useValueForwarding } from "./hooks/useValueForwarding";
import { useOutputSinkSync } from "./hooks/useOutputSinkSync";
import { useKeyboardShortcuts } from "./hooks/useKeyboardShortcuts";

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
};

const edgeTypes = {
  default: CustomEdge,
};

export interface GraphEditorHandle {
  refreshGraph: () => void;
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
    // Graph state
    const {
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
      selectedNodeId,
      setSelectedNodeId,
      handlePipelineSelect,
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
      }
    );

    // Expose refreshGraph
    useImperativeHandle(ref, () => ({ refreshGraph }), [refreshGraph]);

    // Context menu & add-node modal
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

    // Right-click drag = box-select, click = context menu
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

        // Close existing context menu
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
            // Box selection
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
            // Show context menu
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

    // Connection logic
    const {
      isValidConnection,
      onConnect,
      onReconnect,
      findConnectedPipelineParams,
    } = useConnectionLogic(nodes, setNodes, setEdges, handleEdgeDelete);

    // Node factories
    const { handleNodeTypeSelect, handleDeleteNode } = useNodeFactories({
      nodes,
      setNodes,
      setEdges,
      availablePipelineIds,
      portsMap,
      handlePipelineSelect,
      selectedNodeId,
      setSelectedNodeId,
      spoutOutputAvailable,
      ndiOutputAvailable,
      syphonOutputAvailable,
      pendingNodePosition,
      setPendingNodePosition,
    });

    // Value forwarding
    useValueForwarding(
      nodes,
      edges,
      findConnectedPipelineParams,
      resolveBackendId,
      isStreamingRef,
      onNodeParamChangeRef,
      setNodes
    );

    // Output sink sync
    useOutputSinkSync(nodes, onOutputSinkChangeRef);

    // Keyboard shortcuts
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

    // Context menu suppression
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

    // File input ref
    const fileInputRef = useRef<HTMLInputElement>(null);

    // Render
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
              <span className={NODE_TOKENS.toolbarStatus}>
                {status}
                {graphSource && (
                  <span className="text-[#8c8c8d]/70 ml-1">
                    ({graphSource})
                  </span>
                )}
              </span>
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
              onInit={instance => {
                reactFlowInstanceRef.current = instance;
              }}
              onNodeClick={(_event, node) => setSelectedNodeId(node.id)}
              onPaneClick={() => {
                setSelectedNodeId(null);
                setContextMenu(null);
              }}
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
                items={
                  contextMenu.type === "pane"
                    ? [
                        {
                          label: "+ Add node",
                          onClick: () => {
                            setShowAddNodeModal(true);
                          },
                        },
                      ]
                    : [
                        {
                          label: "Delete node",
                          onClick: () => {
                            if (contextMenu.nodeId) {
                              handleDeleteNode(contextMenu.nodeId);
                            }
                          },
                          danger: true,
                        },
                      ]
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

            {/* Right-click drag selection rectangle */}
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

          {/* Clear confirmation dialog */}
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
