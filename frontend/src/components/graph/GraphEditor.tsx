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
import { ControlNode } from "./ControlNode";
import { MathNode } from "./MathNode";
import { NoteNode } from "./NoteNode";
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
  control: ControlNode,
  math: MathNode,
  note: NoteNode,
};

const edgeTypes = {
  default: CustomEdge,
};

const HANDLE_COLORS: Record<string, string> = {
  video: "#ffffff",
  video2: "#ffffff",
  vace_input_frames: "#a78bfa",
  vace_input_masks: "#f472b6",
  source: "#4ade80",
  sink: "#fb923c",
};

const PARAM_TYPE_COLORS: Record<string, string> = {
  string: "#fbbf24",
  number: "#38bdf8",
  boolean: "#34d399",
  float: "#a78bfa",
  int: "#a78bfa",
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
    if (sourceNode.data.nodeType === "control") {
      const controlType = sourceNode.data.controlType;
      const outputType = controlType === "string" ? "string" : "number";
      return PARAM_TYPE_COLORS[outputType] || "#9ca3af";
    }
    if (sourceNode.data.nodeType === "math") {
      return PARAM_TYPE_COLORS["number"] || "#9ca3af";
    }
    return "#9ca3af";
  }

  if (sourceNode.data.nodeType === "pipeline") {
    return HANDLE_COLORS[parsed.name] || HANDLE_COLORS.video;
  }

  if (sourceNode.data.nodeType === "source") {
    return HANDLE_COLORS[parsed.name] || HANDLE_COLORS.video;
  }
  if (sourceNode.data.nodeType === "sink") {
    return HANDLE_COLORS.sink;
  }

  return HANDLE_COLORS.video;
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
}

export function GraphEditor({
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
          const supportsPrompts = schema?.supports_prompts ?? false;
          // Clear all height constraints so node auto-sizes to new content
          const newStyle = { ...n.style };
          delete newStyle.height;
          return {
            ...n,
            style: newStyle,
            height: undefined,
            measured: undefined,
            data: {
              ...n.data,
              pipelineId: newPipelineId,
              label: newPipelineId || n.id,
              streamInputs: ports?.inputs ?? ["video"],
              streamOutputs: ports?.outputs ?? ["video"],
              parameterInputs,
              supportsPrompts,
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

  // Refs to keep callbacks stable
  const nodesRef = useRef(nodes);
  nodesRef.current = nodes;

  const onNodeParamChangeRef = useRef(onNodeParameterChange);
  onNodeParamChangeRef.current = onNodeParameterChange;

  const isStreamingRef = useRef(isStreaming);
  isStreamingRef.current = isStreaming;

  const onVideoFileUploadRef = useRef(onVideoFileUpload);
  onVideoFileUploadRef.current = onVideoFileUpload;

  const onSourceModeChangeRef = useRef(onSourceModeChange);
  onSourceModeChangeRef.current = onSourceModeChange;

  const onSpoutSourceChangeRef = useRef(onSpoutSourceChange);
  onSpoutSourceChangeRef.current = onSpoutSourceChange;

  const onNdiSourceChangeRef = useRef(onNdiSourceChange);
  onNdiSourceChangeRef.current = onNdiSourceChange;

  const onSyphonSourceChangeRef = useRef(onSyphonSourceChange);
  onSyphonSourceChangeRef.current = onSyphonSourceChange;

  const onGraphChangeRef = useRef(onGraphChange);
  onGraphChangeRef.current = onGraphChange;

  const nodeParamsRef = useRef(nodeParams);
  nodeParamsRef.current = nodeParams;

  const resolveBackendId = useCallback(
    (nodeId: string): string => {
      const node = nodesRef.current.find(n => n.id === nodeId);
      if (node?.data.nodeType === "pipeline" && node.data.pipelineId) {
        return node.data.pipelineId as string;
      }
      return nodeId;
    },
    []
  );

  const handleNodeParameterChange = useCallback(
    (nodeId: string, key: string, value: unknown) => {
      setNodeParams(prev => ({
        ...prev,
        [nodeId]: { ...(prev[nodeId] || {}), [key]: value },
      }));
      onNodeParamChangeRef.current?.(resolveBackendId(nodeId), key, value);
    },
    [resolveBackendId]
  );

  const handlePromptChange = useCallback(
    (nodeId: string, text: string) => {
      setNodeParams(prev => ({
        ...prev,
        [nodeId]: { ...(prev[nodeId] || {}), __prompt: text },
      }));
      if (isStreamingRef.current) {
        onNodeParamChangeRef.current?.(resolveBackendId(nodeId), "prompts", [{ text, weight: 100 }]);
      }
    },
    [resolveBackendId]
  );

  useEffect(() => {
    if (availablePipelineIds.length === 0) return;
    setNodes(nds =>
      nds.map(n => {
        if (n.data.nodeType === "pipeline") {
          const pipelineId = n.data.pipelineId;
          const schema = pipelineId ? pipelineSchemas[pipelineId] : null;
          const parameterInputs = schema ? extractParameterPorts(schema) : [];
          const supportsPrompts = schema?.supports_prompts ?? false;
          const nodeParamValues = nodeParamsRef.current[n.id] || {};
          return {
            ...n,
            data: {
              ...n.data,
              availablePipelineIds,
              pipelinePortsMap: portsMap,
              onPipelineSelect: handlePipelineSelect,
              parameterInputs,
              parameterValues: nodeParamValues,
              onParameterChange: handleNodeParameterChange,
              supportsPrompts,
              promptText: (nodeParamValues.__prompt as string) || "",
              onPromptChange: handlePromptChange,
            },
          };
        }
        if (n.data.nodeType === "source") {
          return {
            ...n,
            data: {
              ...n.data,
              localStream,
              onVideoFileUpload: onVideoFileUploadRef.current,
              onSourceModeChange: onSourceModeChangeRef.current,
              spoutAvailable,
              ndiAvailable,
              syphonAvailable,
              onSpoutSourceChange: onSpoutSourceChangeRef.current,
              onNdiSourceChange: onNdiSourceChangeRef.current,
              onSyphonSourceChange: onSyphonSourceChangeRef.current,
            },
          };
        }
        if (n.data.nodeType === "sink") {
          return {
            ...n,
            data: { ...n.data, remoteStream },
          };
        }
        return n;
      })
    );
  }, [availablePipelineIds, portsMap, handlePipelineSelect, setNodes, pipelineSchemas, handleNodeParameterChange, localStream, remoteStream, spoutAvailable, ndiAvailable, syphonAvailable]);

  // Sync nodeParams to pipeline node parameterValues
  useEffect(() => {
    setNodes(nds =>
      nds.map(n => {
        if (n.data.nodeType === "pipeline") {
          const vals = nodeParams[n.id] || {};
          if (n.data.parameterValues === vals) return n;
          return {
            ...n,
            data: {
              ...n.data,
              parameterValues: vals,
              promptText: (vals.__prompt as string) || "",
            },
          };
        }
        return n;
      })
    );
  }, [nodeParams, setNodes]);

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
          if (n.data.nodeType === "pipeline") {
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
                parameterValues: nodeParamsRef.current[n.id] || {},
                onParameterChange: handleNodeParameterChange,
              },
            };
          }
          if (n.data.nodeType === "source") {
            return {
              ...n,
              data: {
                ...n.data,
                localStream,
                onVideoFileUpload: onVideoFileUploadRef.current,
                onSourceModeChange: onSourceModeChangeRef.current,
                spoutAvailable,
                ndiAvailable,
                syphonAvailable,
                onSpoutSourceChange: onSpoutSourceChangeRef.current,
                onNdiSourceChange: onNdiSourceChangeRef.current,
                onSyphonSourceChange: onSyphonSourceChangeRef.current,
              },
            };
          }
          if (n.data.nodeType === "sink") {
            return { ...n, data: { ...n.data, remoteStream } };
          }
          return n;
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

  // Find pipeline params connected to a value/control node output
  const findConnectedPipelineParams = useCallback(
    (sourceNodeId: string, edges: Edge[], nodes: Node<FlowNodeData>[]): Array<{ nodeId: string; paramName: string }> => {
      const connected: Array<{ nodeId: string; paramName: string }> = [];

      for (const edge of edges) {
        if (edge.source !== sourceNodeId) continue;

        const targetParsed = parseHandleId(edge.targetHandle);
        if (targetParsed?.kind !== "param") continue;

        const targetNode = nodes.find(n => n.id === edge.target);
        if (targetNode?.data.nodeType !== "pipeline") continue;

        connected.push({
          nodeId: edge.target,
          paramName: targetParsed.name,
        });
      }

      return connected;
    },
    []
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

        let sourceType: "string" | "number" | "boolean" | undefined;

        if (sourceNode.data.nodeType === "value") {
          sourceType = sourceNode.data.valueType;
        } else if (sourceNode.data.nodeType === "control") {
          const controlType = sourceNode.data.controlType;
          sourceType = controlType === "string" ? "string" : "number";
        } else if (sourceNode.data.nodeType === "math") {
          sourceType = "number";
        }

        if (!sourceType) return false;

        if (targetParsed.name === "__prompt") {
          return sourceType === "string";
        }

        // Math nodes accept param:a and param:b inputs (both must be number)
        if (targetNode.data.nodeType === "math") {
          return sourceType === "number" && (targetParsed.name === "a" || targetParsed.name === "b");
        }

        const targetParam = targetNode.data.parameterInputs?.find(
          p => p.name === targetParsed.name
        );
        if (!targetParam) return false;

        if (targetParam.type === "list_number" && sourceType === "number") {
          return true;
        }

        return sourceType === targetParam.type;
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
        style: { width: 240, height: 200 },
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
        style: { width: 240, height: 200 },
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

  const addControlNode = useCallback(
    (controlType: "float" | "int" | "string", position?: { x: number; y: number }) => {
      const id = generateNodeId(controlType === "float" ? "floatControl" : controlType === "int" ? "intControl" : "stringControl", existingIds);
      const newNode: Node<FlowNodeData> = {
        id,
        type: "control",
        position: position ?? { x: 50, y: 50 + nodes.length * 100 },
        data: {
          label: controlType === "float" ? "FloatControl" : controlType === "int" ? "IntControl" : "StringControl",
          nodeType: "control",
          controlType,
          controlPattern: "sine",
          controlSpeed: 1.0,
          controlMin: 0,
          controlMax: 1.0,
          controlItems: controlType === "string" ? ["item1", "item2", "item3"] : undefined,
          isPlaying: false,
          parameterOutputs: [{
            name: "value",
            type: controlType === "string" ? "string" : "number",
            defaultValue: controlType === "string" ? "" : 0,
          }],
        },
      };
      setNodes(nds => [...nds, newNode]);
    },
    [existingIds, nodes.length, setNodes]
  );

  const addMathNode = useCallback(
    (position?: { x: number; y: number }) => {
      const id = generateNodeId("math", existingIds);
      const newNode: Node<FlowNodeData> = {
        id,
        type: "math",
        position: position ?? { x: 50, y: 50 + nodes.length * 100 },
        data: {
          label: "Math",
          nodeType: "math",
          mathOp: "add",
          currentValue: undefined,
          parameterOutputs: [{
            name: "value",
            type: "number",
            defaultValue: 0,
          }],
        },
      };
      setNodes(nds => [...nds, newNode]);
    },
    [existingIds, nodes.length, setNodes]
  );

  const addNoteNode = useCallback(
    (position?: { x: number; y: number }) => {
      const id = generateNodeId("note", existingIds);
      const newNode: Node<FlowNodeData> = {
        id,
        type: "note",
        position: position ?? { x: 50, y: 50 + nodes.length * 100 },
        data: {
          label: "Note",
          nodeType: "note",
          noteText: "",
        },
      };
      setNodes(nds => [...nds, newNode]);
    },
    [existingIds, nodes.length, setNodes]
  );

  const handleNodeTypeSelect = useCallback(
    (type: "source" | "pipeline" | "sink" | "value" | "control" | "math" | "note", subType?: string) => {
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
          if (subType === "string" || subType === "number" || subType === "boolean") {
            addValueNode(subType, pendingNodePosition);
          }
          break;
        case "control":
          if (subType === "float" || subType === "int" || subType === "string") {
            addControlNode(subType, pendingNodePosition);
          }
          break;
        case "math":
          addMathNode(pendingNodePosition);
          break;
        case "note":
          addNoteNode(pendingNodePosition);
          break;
      }

      setPendingNodePosition(null);
    },
    [pendingNodePosition, addSourceNode, addPipelineNode, addSinkNode, addValueNode, addControlNode, addMathNode, addNoteNode]
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

    onGraphChangeRef.current?.();
  }, [nodes, edges]);

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

  // Forward value/control node changes to connected params
  const lastForwardTimeRef = useRef<Record<string, number>>({});
  useEffect(() => {
    if (!isStreamingRef.current || !onNodeParamChangeRef.current) return;

    const throttleMs = 100;

    for (const node of nodes) {
      if (node.data.nodeType !== "value" && node.data.nodeType !== "control" && node.data.nodeType !== "math") continue;

      const connected = findConnectedPipelineParams(node.id, edges, nodes);
      if (connected.length === 0) continue;

      let value: unknown;
      if (node.data.nodeType === "value") {
        value = node.data.value;
      } else if (node.data.nodeType === "control" || node.data.nodeType === "math") {
        value = node.data.currentValue;
      }

      if (value === undefined) continue;

      if (node.data.nodeType === "control" || node.data.nodeType === "math") {
        const now = Date.now();
        const lastTime = lastForwardTimeRef.current[node.id] || 0;
        if (now - lastTime < throttleMs) continue;
        lastForwardTimeRef.current[node.id] = now;
      }

      for (const { nodeId, paramName } of connected) {
        const backendId = resolveBackendId(nodeId);
        if (paramName === "__prompt") {
          onNodeParamChangeRef.current(backendId, "prompts", [{ text: String(value), weight: 100 }]);
        } else {
          onNodeParamChangeRef.current(backendId, paramName, value);
        }
      }
    }
  }, [nodes, edges, findConnectedPipelineParams, resolveBackendId]);

  // Keyboard shortcut: Tab to open Add Node modal at viewport center
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key !== "Tab") return;

      const activeElement = document.activeElement;
      const isInputElement = activeElement?.tagName === "INPUT" ||
                             activeElement?.tagName === "TEXTAREA" ||
                             activeElement?.tagName === "SELECT";

      if (isInputElement) return;

      e.preventDefault();

      if (!reactFlowInstanceRef.current) return;

      // Get viewport center
      const centerX = window.innerWidth / 2;
      const centerY = window.innerHeight / 2;

      const flowPosition = reactFlowInstanceRef.current.screenToFlowPosition({
        x: centerX,
        y: centerY,
      });

      setPendingNodePosition(flowPosition);
      setShowAddNodeModal(true);
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

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
            if (n.data.nodeType === "pipeline") {
              const pipelineId = n.data.pipelineId;
              const schema = pipelineId ? pipelineSchemas[pipelineId] : null;
              const parameterInputs = schema ? extractParameterPorts(schema) : [];
              const supportsPrompts = schema?.supports_prompts ?? false;
              const nodeParamValues = nodeParamsRef.current[n.id] || {};
              return {
                ...n,
                data: {
                  ...n.data,
                  availablePipelineIds,
                  pipelinePortsMap: portsMap,
                  onPipelineSelect: handlePipelineSelect,
                  parameterInputs,
                  parameterValues: nodeParamValues,
                  onParameterChange: handleNodeParameterChange,
                  supportsPrompts,
                  promptText: (nodeParamValues.__prompt as string) || "",
                  onPromptChange: handlePromptChange,
                },
              };
            }
            if (n.data.nodeType === "source") {
              return {
                ...n,
                data: {
                  ...n.data,
                  localStream,
                  onVideoFileUpload,
                  onSourceModeChange,
                  spoutAvailable,
                  ndiAvailable,
                  syphonAvailable,
                  onSpoutSourceChange,
                  onNdiSourceChange,
                  onSyphonSourceChange,
                },
              };
            }
            if (n.data.nodeType === "sink") {
              return { ...n, data: { ...n.data, remoteStream } };
            }
            return n;
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
                <svg className="animate-spin h-3 w-3" viewBox="0 0 24 24" fill="none">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
              </span>
            ) : isStreaming ? (
              <svg className="h-3.5 w-3.5" viewBox="0 0 24 24" fill="currentColor"><rect x="4" y="4" width="16" height="16" rx="2" /></svg>
            ) : (
              <svg className="h-3.5 w-3.5" viewBox="0 0 24 24" fill="currentColor"><polygon points="5,3 19,12 5,21" /></svg>
            )}
          </button>
          {isStreaming && (
            <button
              onClick={onStopStream}
              className={NODE_TOKENS.toolbarButton}
              title="Stop and clear"
            >
              <svg className="h-3.5 w-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M1 4v6h6" /><path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10" />
              </svg>
            </button>
          )}

          <div className="flex-1" />

          {status && (
            <span className={NODE_TOKENS.toolbarStatus}>
              {status}
              {graphSource && (
                <span className="text-[#8c8c8d]/70 ml-1">({graphSource})</span>
              )}
            </span>
          )}

          <input
            ref={fileInputRef}
            type="file"
            accept=".json"
            onChange={handleImport}
            className="hidden"
          />
          <button onClick={() => fileInputRef.current?.click()} className={NODE_TOKENS.toolbarButton}>
            Import
          </button>
          <button onClick={handleExport} className={NODE_TOKENS.toolbarButton}>
            Export
          </button>
          <button onClick={handleClear} className={NODE_TOKENS.toolbarButton} title="Clear graph">
            Clear
          </button>
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

    </div>
  );
}
