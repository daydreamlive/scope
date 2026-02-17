import { useCallback, useEffect, useRef, useState, useMemo } from "react";
import { DagPreviewContext } from "./DagPreviewContext";
import { useDagPreviews } from "./useDagPreviews";
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
  buildPipelinePortsMap,
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
  const [portsMap, setPortsMap] = useState<
    Record<string, { inputs: string[]; outputs: string[] }>
  >({});
  const [previewsEnabled, setPreviewsEnabled] = useState(false);
  const previews = useDagPreviews(previewsEnabled);

  // Fetch available pipeline IDs and port schemas on mount
  useEffect(() => {
    getPipelineSchemas()
      .then(schemas => {
        setAvailablePipelineIds(Object.keys(schemas.pipelines));
        setPortsMap(buildPipelinePortsMap(schemas.pipelines));
      })
      .catch(err => {
        console.error("Failed to fetch pipeline schemas:", err);
      });
  }, []);

  // Inject availablePipelineIds and portsMap into all pipeline nodes when they change
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
              },
            }
          : n
      )
    );
  }, [availablePipelineIds, portsMap, setNodes]);

  // Load existing DAG on mount (after portsMap is ready)
  useEffect(() => {
    // Wait until portsMap is populated
    if (Object.keys(portsMap).length === 0) return;

    getDag()
      .then(response => {
        if (response.dag) {
          const { nodes: flowNodes, edges: flowEdges } = dagConfigToFlow(
            response.dag,
            portsMap
          );
          // Inject available pipeline IDs and portsMap into pipeline nodes
          const enrichedNodes = flowNodes.map(n =>
            n.data.nodeType === "pipeline"
              ? {
                  ...n,
                  data: {
                    ...n.data,
                    availablePipelineIds,
                    pipelinePortsMap: portsMap,
                  },
                }
              : n
          );
          setNodes(enrichedNodes);
          setEdges(flowEdges);
          setDagSource(response.source);
          setStatus(`Loaded from ${response.source}`);
        } else {
          setStatus("No DAG configured");
        }
        // Mark initial load complete so auto-save can start
        initialLoadDone.current = true;
      })
      .catch(err => {
        console.error("Failed to load DAG:", err);
        setStatus("Failed to load DAG");
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
        streamInputs: ["video"],
        streamOutputs: ["video"],
      },
    };
    setNodes(nds => [...nds, newNode]);
  }, [existingIds, nodes.length, setNodes, availablePipelineIds, portsMap]);

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

  // Auto-save DAG on every change (debounced 500ms)
  useEffect(() => {
    if (!initialLoadDone.current) return;
    if (nodes.length === 0 && edges.length === 0) return;

    const timer = setTimeout(async () => {
      try {
        const dagConfig = flowToDagConfig(nodes, edges);
        const result = await setDag(dagConfig);
        setDagSource("api");
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

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleImport = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      if (!file) return;

      const reader = new FileReader();
      reader.onload = e => {
        try {
          const dagConfig = JSON.parse(e.target?.result as string);
          if (!dagConfig.nodes || !dagConfig.edges) {
            setStatus("Import failed: invalid DAG format");
            return;
          }
          const { nodes: flowNodes, edges: flowEdges } = dagConfigToFlow(
            dagConfig,
            portsMap
          );
          const enrichedNodes = flowNodes.map(n =>
            n.data.nodeType === "pipeline"
              ? {
                  ...n,
                  data: {
                    ...n.data,
                    availablePipelineIds,
                    pipelinePortsMap: portsMap,
                  },
                }
              : n
          );
          setNodes(enrichedNodes);
          setEdges(flowEdges);
          setDagSource(null);
          setStatus(`Imported from ${file.name}`);
        } catch {
          setStatus("Import failed: invalid JSON");
        }
      };
      reader.readAsText(file);

      // Reset so the same file can be re-imported
      event.target.value = "";
    },
    [portsMap, availablePipelineIds, setNodes, setEdges]
  );

  const handleExport = useCallback(() => {
    const dagConfig = flowToDagConfig(nodes, edges);
    const dataStr = JSON.stringify(dagConfig, null, 2);
    const blob = new Blob([dataStr], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `dag-${new Date().toISOString().split("T")[0]}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
    setStatus("DAG exported");
  }, [nodes, edges]);

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
        <button
          onClick={() => setPreviewsEnabled(p => !p)}
          className={`px-3 py-1.5 text-xs font-medium rounded transition-colors ${
            previewsEnabled
              ? "bg-violet-700 hover:bg-violet-600 text-white"
              : "bg-zinc-700 hover:bg-zinc-600 text-zinc-200"
          }`}
        >
          {previewsEnabled ? "Previews ON" : "Previews OFF"}
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
        <DagPreviewContext.Provider value={previews}>
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
            <MiniMap nodeStrokeWidth={3} className="!bg-zinc-900" />
            <Background variant={BackgroundVariant.Dots} gap={16} size={1} />
          </ReactFlow>
        </DagPreviewContext.Provider>
      </div>
    </div>
  );
}
