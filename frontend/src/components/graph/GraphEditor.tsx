import { useCallback, useEffect, useRef, useState, useMemo } from "react";
import {
  ReactFlow,
  Controls,
  Background,
  BackgroundVariant,
  addEdge,
  useNodesState,
  useEdgesState,
} from "@xyflow/react";
import type { Connection, Edge, Node } from "@xyflow/react";
import "@xyflow/react/dist/style.css";

import { SourceNode } from "./SourceNode";
import { PipelineNode } from "./PipelineNode";
import { SinkNode } from "./SinkNode";
import { NodeParametersPanel } from "./NodeParametersPanel";
import {
  graphConfigToFlow,
  flowToGraphConfig,
  generateNodeId,
  buildPipelinePortsMap,
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
};

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

  // Callback for PipelineNode to update its selected pipeline via controlled state
  const handlePipelineSelect = useCallback(
    (nodeId: string, newPipelineId: string | null) => {
      setNodes(nds =>
        nds.map(n => {
          if (n.id !== nodeId) return n;
          const ports =
            newPipelineId && portsMap ? portsMap[newPipelineId] : null;
          return {
            ...n,
            data: {
              ...n.data,
              pipelineId: newPipelineId,
              label: newPipelineId || n.id,
              streamInputs: ports?.inputs ?? ["video"],
              streamOutputs: ports?.outputs ?? ["video"],
            },
          };
        })
      );
    },
    [setNodes, portsMap]
  );

  // Fetch available pipeline IDs and port schemas on mount
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

  // Inject availablePipelineIds, portsMap, and callback into all pipeline nodes when they change
  useEffect(() => {
    if (availablePipelineIds.length === 0) return;
    setNodes(nds =>
      nds.map(n =>
        n.data.nodeType === "pipeline"
          ? {
              ...n,
              data: {
                ...n.data,
                availablePipelineIds,
                pipelinePortsMap: portsMap,
                onPipelineSelect: handlePipelineSelect,
              },
            }
          : n
      )
    );
  }, [availablePipelineIds, portsMap, handlePipelineSelect, setNodes]);

  // Load existing graph on mount (after portsMap is ready)
  useEffect(() => {
    // Wait until portsMap is populated
    if (Object.keys(portsMap).length === 0) return;

    getGraph()
      .then(response => {
        if (response.graph) {
          const { nodes: flowNodes, edges: flowEdges } = graphConfigToFlow(
            response.graph,
            portsMap
          );
          // Inject available pipeline IDs, portsMap, and callback into pipeline nodes
          const enrichedNodes = flowNodes.map(n =>
            n.data.nodeType === "pipeline"
              ? {
                  ...n,
                  data: {
                    ...n.data,
                    availablePipelineIds,
                    pipelinePortsMap: portsMap,
                    onPipelineSelect: handlePipelineSelect,
                  },
                }
              : n
          );
          setNodes(enrichedNodes);
          setEdges(flowEdges);
          setGraphSource(response.source);
          setStatus(`Loaded from ${response.source}`);
        } else {
          setStatus("No graph configured");
        }
        // Mark initial load complete so auto-save can start
        initialLoadDone.current = true;
      })
      .catch(err => {
        console.error("Failed to load graph:", err);
        setStatus("Failed to load graph");
        initialLoadDone.current = true;
      });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [portsMap]);

  const onConnect = useCallback(
    (connection: Connection) => {
      setEdges(eds => addEdge({ ...connection, animated: true }, eds));
    },
    [setEdges]
  );

  const existingIds = useMemo(() => new Set(nodes.map(n => n.id)), [nodes]);

  const addSourceNode = useCallback(() => {
    const id = generateNodeId("input", existingIds);
    const newNode: Node<FlowNodeData> = {
      id,
      type: "source",
      position: { x: 50, y: 50 + nodes.length * 100 },
      data: { label: id, nodeType: "source" },
    };
    setNodes(nds => [...nds, newNode]);
  }, [existingIds, nodes.length, setNodes]);

  const addPipelineNode = useCallback(() => {
    const id = generateNodeId("pipeline", existingIds);
    const newNode: Node<FlowNodeData> = {
      id,
      type: "pipeline",
      position: { x: 350, y: 50 + nodes.length * 100 },
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
  }, [existingIds, nodes.length, setNodes, availablePipelineIds, portsMap, handlePipelineSelect]);

  const addSinkNode = useCallback(() => {
    const id = generateNodeId("output", existingIds);
    const newNode: Node<FlowNodeData> = {
      id,
      type: "sink",
      position: { x: 650, y: 50 + nodes.length * 100 },
      data: { label: id, nodeType: "sink" },
    };
    setNodes(nds => [...nds, newNode]);
  }, [existingIds, nodes.length, setNodes]);

  // Track whether initial load is complete to avoid auto-saving on mount
  const initialLoadDone = useRef(false);

  // Notify parent that user edited the graph (skip initial load)
  useEffect(() => {
    if (!initialLoadDone.current) return;
    if (nodes.length === 0 && edges.length === 0) return;

    onGraphChange?.();
  }, [nodes, edges, onGraphChange]);

  // Auto-save graph to backend on every change (debounced 500ms)
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
          const enrichedNodes = flowNodes.map((n: Node<FlowNodeData>) =>
            n.data.nodeType === "pipeline"
              ? {
                  ...n,
                  data: {
                    ...n.data,
                    availablePipelineIds,
                    pipelinePortsMap: portsMap,
                    onPipelineSelect: handlePipelineSelect,
                  },
                }
              : n
          );
          setNodes(enrichedNodes);
          setEdges(flowEdges);
          setGraphSource(null);
          setStatus(`Imported from ${file.name}`);
        } catch {
          setStatus("Import failed: invalid JSON");
        }
      };
      reader.readAsText(file);

      // Reset so the same file can be re-imported
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

  // Find selected node's pipeline_id
  const selectedNode = nodes.find(n => n.id === selectedNodeId);
  const selectedPipelineId =
    selectedNode?.data.nodeType === "pipeline"
      ? (selectedNode.data.pipelineId as string | null)
      : null;

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

  return (
    <div className="flex h-full w-full">
      {/* Main editor area */}
      <div className="flex flex-col flex-1">
        {/* Toolbar */}
        <div className="flex items-center gap-2 px-4 py-2 bg-zinc-900 border-b border-zinc-700">
          <button
            onClick={addSourceNode}
            className="px-3 py-1.5 text-xs font-medium rounded bg-green-800 hover:bg-green-700 text-green-100 transition-colors"
          >
            + Source
          </button>
          <button
            onClick={addPipelineNode}
            className="px-3 py-1.5 text-xs font-medium rounded bg-blue-800 hover:bg-blue-700 text-blue-100 transition-colors"
          >
            + Pipeline
          </button>
          <button
            onClick={addSinkNode}
            className="px-3 py-1.5 text-xs font-medium rounded bg-orange-800 hover:bg-orange-700 text-orange-100 transition-colors"
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
          <button
            onClick={() => fileInputRef.current?.click()}
            className="px-3 py-1.5 text-xs font-medium rounded bg-zinc-700 hover:bg-zinc-600 text-zinc-200 transition-colors"
          >
            Import JSON
          </button>
          <button
            onClick={handleExport}
            className="px-3 py-1.5 text-xs font-medium rounded bg-zinc-700 hover:bg-zinc-600 text-zinc-200 transition-colors"
          >
            Export JSON
          </button>
          <button
            onClick={handleClear}
            className="px-3 py-1.5 text-xs font-medium rounded bg-zinc-700 hover:bg-zinc-600 text-zinc-200 transition-colors"
          >
            Clear
          </button>
          {status && (
            <span className="text-xs text-zinc-400 ml-2">
              {status}
              {graphSource && (
                <span className="text-zinc-500 ml-1">({graphSource})</span>
              )}
            </span>
          )}
        </div>

        {/* Flow Canvas */}
        <div className="flex-1">
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onNodeClick={(_event, node) => setSelectedNodeId(node.id)}
            onPaneClick={() => setSelectedNodeId(null)}
            nodeTypes={nodeTypes}
            colorMode="dark"
            fitView
            deleteKeyCode={["Backspace", "Delete"]}
          >
            <Controls />
            <Background variant={BackgroundVariant.Dots} gap={16} size={1} />
          </ReactFlow>
        </div>
      </div>

      {/* Right panel: Node parameters */}
      {selectedNodeId && selectedPipelineId && (
        <div className="w-72 border-l border-zinc-700 bg-zinc-900 overflow-y-auto">
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
