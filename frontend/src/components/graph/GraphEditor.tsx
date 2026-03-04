import { useCallback, useEffect, useRef, useState, useMemo } from "react";
import {
  ReactFlow,
  Controls,
  Background,
  BackgroundVariant,
  addEdge,
  reconnectEdge,
  useNodesState,
  useEdgesState,
} from "@xyflow/react";
import type { Connection, Edge, Node, ReactFlowInstance } from "@xyflow/react";
import "@xyflow/react/dist/style.css";

import { SourceNode } from "./SourceNode";
import { PipelineNode } from "./PipelineNode";
import { SinkNode } from "./SinkNode";
import { ValueNode } from "./ValueNode";
import { NodeParametersPanel } from "./NodeParametersPanel";
import { CustomEdge } from "./CustomEdge";
import { ContextMenu } from "./ContextMenu";
import { AddNodeModal } from "./AddNodeModal";
import { NODE_TOKENS } from "./node-ui";
import {
  graphConfigToFlow,
  flowToGraphConfig,
  generateNodeId,
  buildPipelinePortsMap,
  parseHandleId,
  extractParameterPorts,
} from "../../lib/graphUtils";
import type { FlowNodeData } from "../../lib/graphUtils";
import type { PipelineSchemaInfo } from "../../lib/api";
import {
  getGraph,
  setGraph,
  clearGraph,
  getPipelineSchemas,
} from "../../lib/api";

const nodeTypes = {
  source: SourceNode,
  pipeline: PipelineNode,
  sink: SinkNode,
  value: ValueNode,
};

const edgeTypes = {
  default: CustomEdge,
};

const HANDLE_COLORS: Record<string, string> = {
  video: "#60a5fa",
  video2: "#22d3ee",
  vace_input_frames: "#a78bfa",
  vace_input_masks: "#f472b6",
  source: "#4ade80",
  sink: "#fb923c",
};

const PARAM_TYPE_COLORS: Record<string, string> = {
  string: "#fbbf24",
  number: "#38bdf8",
  boolean: "#34d399",
};

function getEdgeColor(
  sourceNode: Node<FlowNodeData> | undefined,
  handleId: string | null | undefined
): string {
  if (!sourceNode || !handleId) return "#9ca3af";

  const parsed = parseHandleId(handleId);
  if (!parsed) return "#9ca3af";

  if (parsed.kind === "param") {
    if (sourceNode.data.nodeType === "value") {
      const valueType = sourceNode.data.valueType;
      return PARAM_TYPE_COLORS[valueType || "string"] || "#9ca3af";
    }
    return "#9ca3af";
  }

  if (sourceNode.data.nodeType === "pipeline") {
    return HANDLE_COLORS[parsed.name] || HANDLE_COLORS.video;
  }

  if (sourceNode.data.nodeType === "source") {
    return HANDLE_COLORS.source;
  }
  if (sourceNode.data.nodeType === "sink") {
    return HANDLE_COLORS.sink;
  }

  return HANDLE_COLORS.video;
}

interface GraphEditorProps {
  /** Whether the stream is currently active */
  isStreaming?: boolean;
  /** Callback when a per-node parameter changes during streaming */
  onNodeParameterChange?: (nodeId: string, key: string, value: unknown) => void;
  /** Called whenever the user edits the graph (not on initial load) */
  onGraphChange?: () => void;
  /** Called when the user clears the graph */
  onGraphClear?: () => void;
}

export function GraphEditor({
  isStreaming = false,
  onNodeParameterChange,
  onGraphChange,
  onGraphClear,
}: GraphEditorProps) {
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

  const [contextMenu, setContextMenu] = useState<{
    x: number;
    y: number;
    type: "pane" | "node";
    nodeId?: string;
  } | null>(null);

  const [showAddNodeModal, setShowAddNodeModal] = useState(false);
  const [pendingNodePosition, setPendingNodePosition] = useState<{
    x: number;
    y: number;
  } | null>(null);

  const reactFlowInstanceRef = useRef<ReactFlowInstance<Node<FlowNodeData>, Edge> | null>(null);
  const handlePipelineSelect = useCallback(
    (nodeId: string, newPipelineId: string | null) => {
      setNodes(nds =>
        nds.map(n => {
          if (n.id !== nodeId) return n;
          const ports =
            newPipelineId && portsMap ? portsMap[newPipelineId] : null;
          const schema = newPipelineId ? pipelineSchemas[newPipelineId] : null;
          const parameterInputs = schema ? extractParameterPorts(schema) : [];
          return {
            ...n,
            data: {
              ...n.data,
              pipelineId: newPipelineId,
              label: newPipelineId || n.id,
              streamInputs: ports?.inputs ?? ["video"],
              streamOutputs: ports?.outputs ?? ["video"],
              parameterInputs,
            },
          };
        })
      );
    },
    [setNodes, portsMap, pipelineSchemas]
  );

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

  const handleNodeParameterChange = useCallback(
    (nodeId: string, key: string, value: unknown) => {
      setNodeParams(prev => ({
        ...prev,
        [nodeId]: { ...(prev[nodeId] || {}), [key]: value },
      }));
      onNodeParameterChange?.(nodeId, key, value);
    },
    [onNodeParameterChange]
  );

  useEffect(() => {
    if (availablePipelineIds.length === 0) return;
    setNodes(nds =>
      nds.map(n => {
        if (n.data.nodeType !== "pipeline") return n;
        const pipelineId = n.data.pipelineId;
        const schema = pipelineId ? pipelineSchemas[pipelineId] : null;
        const parameterInputs = schema ? extractParameterPorts(schema) : [];
        return {
          ...n,
          data: {
            ...n.data,
            availablePipelineIds,
            pipelinePortsMap: portsMap,
            onPipelineSelect: handlePipelineSelect,
            parameterInputs,
            parameterValues: nodeParams[n.id] || {},
            onParameterChange: handleNodeParameterChange,
          },
        };
      })
    );
  }, [availablePipelineIds, portsMap, handlePipelineSelect, setNodes, pipelineSchemas, nodeParams, handleNodeParameterChange]);

  useEffect(() => {
    if (Object.keys(portsMap).length === 0) return;

    getGraph()
      .then(response => {
        if (response.graph) {
          const { nodes: flowNodes, edges: flowEdges } = graphConfigToFlow(
            response.graph,
            portsMap
          );
        const enrichedNodes = flowNodes.map(n => {
          if (n.data.nodeType !== "pipeline") return n;
          const pipelineId = n.data.pipelineId;
          const schema = pipelineId ? pipelineSchemas[pipelineId] : null;
          const parameterInputs = schema ? extractParameterPorts(schema) : [];
          return {
            ...n,
            data: {
              ...n.data,
              availablePipelineIds,
              pipelinePortsMap: portsMap,
              onPipelineSelect: handlePipelineSelect,
              parameterInputs,
              parameterValues: nodeParams[n.id] || {},
              onParameterChange: handleNodeParameterChange,
            },
          };
        });
          setNodes(enrichedNodes);
          const coloredEdges = flowEdges.map(edge => {
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
          setEdges(coloredEdges);
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

  const handleEdgeDelete = useCallback(
    (edgeId: string) => {
      setEdges(eds => eds.filter(e => e.id !== edgeId));
    },
    [setEdges]
  );

  const isValidConnection = useCallback(
    (edgeOrConnection: Edge | Connection): boolean => {
      let connection: Connection;
      if ("source" in edgeOrConnection && "target" in edgeOrConnection) {
        connection = edgeOrConnection as Connection;
      } else {
        const edge = edgeOrConnection as Edge;
        connection = {
          source: edge.source,
          target: edge.target,
          sourceHandle: edge.sourceHandle ?? null,
          targetHandle: edge.targetHandle ?? null,
        };
      }

      if (!connection.source || !connection.target || !connection.sourceHandle || !connection.targetHandle) {
        return true;
      }

      const sourceParsed = parseHandleId(connection.sourceHandle);
      const targetParsed = parseHandleId(connection.targetHandle);

      if (!sourceParsed || !targetParsed) {
        return true;
      }

      if (sourceParsed.kind === "stream" && targetParsed.kind === "stream") {
        return true;
      }

      if (sourceParsed.kind === "param" && targetParsed.kind === "param") {
        const sourceNode = nodes.find(n => n.id === connection.source);
        const targetNode = nodes.find(n => n.id === connection.target);

        if (!sourceNode || !targetNode) return false;

        if (sourceNode.data.nodeType === "value") {
          const sourceType = sourceNode.data.valueType;
          const targetParam = targetNode.data.parameterInputs?.find(
            p => p.name === targetParsed.name
          );
          return sourceType === targetParam?.type;
        }

        return false;
      }

      return false;
    },
    [nodes]
  );

  const onConnect = useCallback(
    (connection: Connection) => {
      if (!isValidConnection(connection)) {
        return;
      }

      if (connection.target && connection.targetHandle) {
        setEdges(eds =>
          eds.filter(
            e =>
              !(
                e.target === connection.target &&
                e.targetHandle === connection.targetHandle
              )
          )
        );
      }

      const sourceNode = nodes.find(n => n.id === connection.source);
      const edgeColor = getEdgeColor(sourceNode, connection.sourceHandle);

      setEdges(eds =>
        addEdge(
          {
            ...connection,
            type: "default",
            reconnectable: "target" as const,
            style: { stroke: edgeColor, strokeWidth: 2 },
            animated: false,
            data: { onDelete: handleEdgeDelete },
          },
          eds
        )
      );
    },
    [setEdges, nodes, handleEdgeDelete, isValidConnection]
  );

  const onReconnect = useCallback(
    (oldEdge: Edge, newConnection: Connection) => {
      setEdges(eds => {
        const updated = reconnectEdge(oldEdge, newConnection, eds);
        return updated.map(e => {
          if (
            e.source === newConnection.source &&
            e.target === newConnection.target &&
            e.sourceHandle === newConnection.sourceHandle &&
            e.targetHandle === newConnection.targetHandle
          ) {
            const sourceNode = nodes.find(n => n.id === e.source);
            const edgeColor = getEdgeColor(sourceNode, e.sourceHandle);
            return {
              ...e,
              type: "default",
              reconnectable: "target" as const,
              style: { stroke: edgeColor, strokeWidth: 2 },
              animated: false,
              data: { onDelete: handleEdgeDelete },
            };
          }
          return e;
        });
      });
    },
    [setEdges, nodes, handleEdgeDelete]
  );

  const existingIds = useMemo(() => new Set(nodes.map(n => n.id)), [nodes]);

  const addSourceNode = useCallback(
    (position?: { x: number; y: number }) => {
      const id = generateNodeId("input", existingIds);
      const newNode: Node<FlowNodeData> = {
        id,
        type: "source",
        position: position ?? { x: 50, y: 50 + nodes.length * 100 },
        data: { label: id, nodeType: "source" },
      };
      setNodes(nds => [...nds, newNode]);
    },
    [existingIds, nodes.length, setNodes]
  );

  const addPipelineNode = useCallback(
    (position?: { x: number; y: number }) => {
      const id = generateNodeId("pipeline", existingIds);
      const newNode: Node<FlowNodeData> = {
        id,
        type: "pipeline",
        position: position ?? { x: 350, y: 50 + nodes.length * 100 },
        data: {
          label: id,
          pipelineId: null,
          nodeType: "pipeline",
          availablePipelineIds,
          pipelinePortsMap: portsMap,
          onPipelineSelect: handlePipelineSelect,
          streamInputs: ["video"],
          streamOutputs: ["video"],
        },
      };
      setNodes(nds => [...nds, newNode]);
    },
    [existingIds, nodes.length, setNodes, availablePipelineIds, portsMap, handlePipelineSelect]
  );

  const addSinkNode = useCallback(
    (position?: { x: number; y: number }) => {
      const id = generateNodeId("output", existingIds);
      const newNode: Node<FlowNodeData> = {
        id,
        type: "sink",
        position: position ?? { x: 650, y: 50 + nodes.length * 100 },
        data: { label: id, nodeType: "sink" },
      };
      setNodes(nds => [...nds, newNode]);
    },
    [existingIds, nodes.length, setNodes]
  );

  const onPaneContextMenu = useCallback(
    (event: MouseEvent | React.MouseEvent<Element, MouseEvent>) => {
      event.preventDefault();
      if (!reactFlowInstanceRef.current) return;

      const clientX = event instanceof MouseEvent ? event.clientX : event.clientX;
      const clientY = event instanceof MouseEvent ? event.clientY : event.clientY;

      const position = reactFlowInstanceRef.current.screenToFlowPosition({
        x: clientX,
        y: clientY,
      });

      setPendingNodePosition(position);
      setContextMenu({
        x: clientX,
        y: clientY,
        type: "pane",
      });
    },
    []
  );

  const onNodeContextMenu = useCallback(
    (event: React.MouseEvent, node: Node<FlowNodeData>) => {
      event.preventDefault();
      setContextMenu({
        x: event.clientX,
        y: event.clientY,
        type: "node",
        nodeId: node.id,
      });
    },
    []
  );

  const addValueNode = useCallback(
    (valueType: "string" | "number" | "boolean", position?: { x: number; y: number }) => {
      const id = generateNodeId(valueType, existingIds);
      const defaultValue = valueType === "boolean" ? false : valueType === "number" ? 0 : "";
      const newNode: Node<FlowNodeData> = {
        id,
        type: "value",
        position: position ?? { x: 50, y: 50 + nodes.length * 100 },
        data: {
          label: valueType,
          nodeType: "value",
          valueType,
          value: defaultValue,
          parameterOutputs: [{
            name: "value",
            type: valueType,
            defaultValue,
          }],
        },
      };
      setNodes(nds => [...nds, newNode]);
    },
    [existingIds, nodes.length, setNodes]
  );

  const handleNodeTypeSelect = useCallback(
    (type: "source" | "pipeline" | "sink" | "value", valueType?: "string" | "number" | "boolean") => {
      if (!pendingNodePosition) return;

      switch (type) {
        case "source":
          addSourceNode(pendingNodePosition);
          break;
        case "pipeline":
          addPipelineNode(pendingNodePosition);
          break;
        case "sink":
          addSinkNode(pendingNodePosition);
          break;
        case "value":
          if (valueType) {
            addValueNode(valueType, pendingNodePosition);
          }
          break;
      }

      setPendingNodePosition(null);
    },
    [pendingNodePosition, addSourceNode, addPipelineNode, addSinkNode, addValueNode]
  );

  const handleDeleteNode = useCallback(
    (nodeId: string) => {
      setNodes(nds => nds.filter(n => n.id !== nodeId));
      setEdges(eds =>
        eds.filter(e => e.source !== nodeId && e.target !== nodeId)
      );
      if (selectedNodeId === nodeId) {
        setSelectedNodeId(null);
      }
    },
    [setNodes, setEdges, selectedNodeId]
  );

  const initialLoadDone = useRef(false);

  useEffect(() => {
    if (!initialLoadDone.current) return;
    if (nodes.length === 0 && edges.length === 0) return;

    onGraphChange?.();
  }, [nodes, edges, onGraphChange]);

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

  const handleClear = useCallback(async () => {
    try {
      await clearGraph();
      setNodes([]);
      setEdges([]);
      setGraphSource(null);
      setSelectedNodeId(null);
      setStatus("Graph cleared");
      onGraphClear?.();
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unknown error";
      setStatus(`Clear failed: ${message}`);
    }
  }, [setNodes, setEdges, onGraphClear]);

  const fileInputRef = useRef<HTMLInputElement>(null);

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
          const enrichedNodes = flowNodes.map((n: Node<FlowNodeData>) => {
            if (n.data.nodeType !== "pipeline") return n;
            const pipelineId = n.data.pipelineId;
            const schema = pipelineId ? pipelineSchemas[pipelineId] : null;
            const parameterInputs = schema ? extractParameterPorts(schema) : [];
            return {
              ...n,
              data: {
                ...n.data,
                availablePipelineIds,
                pipelinePortsMap: portsMap,
                onPipelineSelect: handlePipelineSelect,
                parameterInputs,
                parameterValues: nodeParams[n.id] || {},
                onParameterChange: handleNodeParameterChange,
              },
            };
          });
          setNodes(enrichedNodes);
          const coloredEdges = flowEdges.map((edge: Edge) => {
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
          setEdges(coloredEdges);
          setGraphSource(null);
          setStatus(`Imported from ${file.name}`);
        } catch {
          setStatus("Import failed: invalid JSON");
        }
      };
      reader.readAsText(file);
      event.target.value = "";
    },
    [portsMap, availablePipelineIds, handlePipelineSelect, setNodes, setEdges]
  );

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

  const selectedNode = nodes.find(n => n.id === selectedNodeId);
  const selectedPipelineId =
    selectedNode?.data.nodeType === "pipeline"
      ? (selectedNode.data.pipelineId as string | null)
      : null;

  return (
    <div className="flex h-full w-full">
      <div className="flex flex-col flex-1">
        <div className={NODE_TOKENS.toolbar}>
          <button
            onClick={() => addSourceNode()}
            className={NODE_TOKENS.toolbarButton}
          >
            + Source
          </button>
          <button
            onClick={() => addPipelineNode()}
            className={NODE_TOKENS.toolbarButton}
          >
            + Pipeline
          </button>
          <button
            onClick={() => addSinkNode()}
            className={NODE_TOKENS.toolbarButton}
          >
            + Sink
          </button>
          <div className="flex-1" />
          <input
            ref={fileInputRef}
            type="file"
            accept=".json"
            onChange={handleImport}
            className="hidden"
          />
          <button onClick={() => fileInputRef.current?.click()} className={NODE_TOKENS.toolbarButton}>
            Import JSON
          </button>
          <button onClick={handleExport} className={NODE_TOKENS.toolbarButton}>
            Export JSON
          </button>
          <button onClick={handleClear} className={NODE_TOKENS.toolbarButton}>
            Clear
          </button>
          {status && (
            <span className={NODE_TOKENS.toolbarStatus}>
              {status}
              {graphSource && (
                <span className="text-[#8c8c8d]/70 ml-1">({graphSource})</span>
              )}
            </span>
          )}
        </div>

        <div className="flex-1 relative">
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onReconnect={onReconnect}
            reconnectRadius={25}
            isValidConnection={isValidConnection}
            onInit={(instance) => {
              reactFlowInstanceRef.current = instance;
            }}
            onNodeClick={(_event, node) => setSelectedNodeId(node.id)}
            onPaneClick={() => {
              setSelectedNodeId(null);
              setContextMenu(null);
            }}
            onPaneContextMenu={onPaneContextMenu}
            onNodeContextMenu={onNodeContextMenu}
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
        </div>
      </div>

      {selectedNodeId && selectedPipelineId && (
        <div className={`w-72 border-l ${NODE_TOKENS.panelBorder} ${NODE_TOKENS.panelBackground} overflow-y-auto`}>
          <NodeParametersPanel
            pipelineId={selectedPipelineId}
            nodeId={selectedNodeId}
            pipelineSchemas={pipelineSchemas}
            parameterValues={nodeParams[selectedNodeId] || {}}
            onParameterChange={handleNodeParameterChange}
            isStreaming={isStreaming}
          />
        </div>
      )}
    </div>
  );
}
