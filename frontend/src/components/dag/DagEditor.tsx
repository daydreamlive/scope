import { useCallback, useEffect, useState, useMemo } from "react";
import {
  ReactFlow,
  Controls,
  MiniMap,
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
import {
  dagConfigToFlow,
  flowToDagConfig,
  generateNodeId,
} from "../../lib/dagUtils";
import type { FlowNodeData } from "../../lib/dagUtils";
import { getDag, setDag, clearDag, getPipelineSchemas } from "../../lib/api";

const nodeTypes = {
  source: SourceNode,
  pipeline: PipelineNode,
  sink: SinkNode,
};

export function DagEditor() {
  const [nodes, setNodes, onNodesChange] = useNodesState<Node<FlowNodeData>>(
    []
  );
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
  const [status, setStatus] = useState<string>("");
  const [dagSource, setDagSource] = useState<string | null>(null);
  const [availablePipelineIds, setAvailablePipelineIds] = useState<string[]>(
    []
  );

  // Fetch available pipeline IDs on mount
  useEffect(() => {
    getPipelineSchemas()
      .then(schemas => {
        setAvailablePipelineIds(Object.keys(schemas.pipelines));
      })
      .catch(err => {
        console.error("Failed to fetch pipeline schemas:", err);
      });
  }, []);

  // Inject availablePipelineIds into all pipeline nodes when the list changes
  useEffect(() => {
    if (availablePipelineIds.length === 0) return;
    setNodes(nds =>
      nds.map(n =>
        n.data.nodeType === "pipeline"
          ? { ...n, data: { ...n.data, availablePipelineIds } }
          : n
      )
    );
  }, [availablePipelineIds, setNodes]);

  // Load existing DAG on mount
  useEffect(() => {
    getDag()
      .then(response => {
        if (response.dag) {
          const { nodes: flowNodes, edges: flowEdges } = dagConfigToFlow(
            response.dag
          );
          // Inject available pipeline IDs into pipeline nodes
          const enrichedNodes = flowNodes.map(n =>
            n.data.nodeType === "pipeline"
              ? { ...n, data: { ...n.data, availablePipelineIds } }
              : n
          );
          setNodes(enrichedNodes);
          setEdges(flowEdges);
          setDagSource(response.source);
          setStatus(`Loaded from ${response.source}`);
        } else {
          setStatus("No DAG configured");
        }
      })
      .catch(err => {
        console.error("Failed to load DAG:", err);
        setStatus("Failed to load DAG");
      });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const onConnect = useCallback(
    (connection: Connection) => {
      setEdges(eds => addEdge({ ...connection, animated: true }, eds));
    },
    [setEdges]
  );

  const existingIds = useMemo(
    () => new Set(nodes.map(n => n.id)),
    [nodes]
  );

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
      },
    };
    setNodes(nds => [...nds, newNode]);
  }, [existingIds, nodes.length, setNodes, availablePipelineIds]);

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

  const handleSave = useCallback(async () => {
    try {
      const dagConfig = flowToDagConfig(nodes, edges);
      const result = await setDag(dagConfig);
      setDagSource("api");
      setStatus(`Saved: ${result.nodes} nodes, ${result.edges} edges`);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unknown error";
      setStatus(`Save failed: ${message}`);
    }
  }, [nodes, edges]);

  const handleClear = useCallback(async () => {
    try {
      await clearDag();
      setNodes([]);
      setEdges([]);
      setDagSource(null);
      setStatus("DAG cleared");
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unknown error";
      setStatus(`Clear failed: ${message}`);
    }
  }, [setNodes, setEdges]);

  return (
    <div className="flex flex-col h-full w-full">
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
        <button
          onClick={handleSave}
          className="px-3 py-1.5 text-xs font-medium rounded bg-indigo-700 hover:bg-indigo-600 text-white transition-colors"
        >
          Save
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
            {dagSource && (
              <span className="text-zinc-500 ml-1">({dagSource})</span>
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
          nodeTypes={nodeTypes}
          colorMode="dark"
          fitView
          deleteKeyCode={["Backspace", "Delete"]}
        >
          <Controls />
          <MiniMap
            nodeStrokeWidth={3}
            className="!bg-zinc-900"
          />
          <Background variant={BackgroundVariant.Dots} gap={16} size={1} />
        </ReactFlow>
      </div>
    </div>
  );
}
