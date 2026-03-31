import { useCallback, useEffect, useRef, useState } from "react";
import type { Edge, Node } from "@xyflow/react";
import {
  graphConfigToFlow,
  flowToGraphConfig,
  workflowToGraphConfig,
  parseHandleId,
} from "../../../../lib/graphUtils";
import { trackEvent } from "../../../../lib/analytics";
import type { FlowNodeData } from "../../../../lib/graphUtils";
import type { PluginInfo } from "../../../../lib/api";
import { resolveWorkflow } from "../../../../lib/api";
import type {
  ScopeWorkflow,
  WorkflowResolutionPlan,
} from "../../../../lib/workflowApi";
import { buildGraphWorkflow } from "../../../../lib/workflowSettings";
import { usePipelinesContext } from "../../../../contexts/PipelinesContext";
import { usePluginsContext } from "../../../../contexts/PluginsContext";
import { useLoRAsContext } from "../../../../contexts/LoRAsContext";
import { useServerInfoContext } from "../../../../contexts/ServerInfoContext";
import {
  enrichNodes,
  colorEdges,
  resetAutoHeightNodes,
  attachNodeParams,
  extractNodeParams,
} from "../../utils/nodeEnrichment";
import type { EnrichNodesDeps } from "../../utils/nodeEnrichment";

// Re-export for backwards compatibility with existing consumers
export {
  enrichNodes,
  colorEdges,
  resetAutoHeightNodes,
  attachNodeParams,
  extractNodeParams,
};
export type { EnrichNodesDeps };

const LS_GRAPH_KEY = "scope:graph:backup";

function saveGraphToLocalStorage(graphJson: string): void {
  try {
    localStorage.setItem(LS_GRAPH_KEY, graphJson);
  } catch {
    // ignore
  }
}

function loadGraphFromLocalStorage(): unknown | null {
  try {
    const raw = localStorage.getItem(LS_GRAPH_KEY);
    if (!raw) return null;
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

function clearGraphFromLocalStorage(): void {
  try {
    localStorage.removeItem(LS_GRAPH_KEY);
  } catch {
    // ignore
  }
}

interface UseGraphPersistenceArgs {
  nodes: Node<FlowNodeData>[];
  edges: Edge[];
  setNodes: React.Dispatch<React.SetStateAction<Node<FlowNodeData>[]>>;
  setEdges: React.Dispatch<React.SetStateAction<Edge[]>>;
  portsMap: Record<string, { inputs: string[]; outputs: string[] }>;
  nodeParamsRef: React.RefObject<Record<string, Record<string, unknown>>>;
  setNodeParams: React.Dispatch<
    React.SetStateAction<Record<string, Record<string, unknown>>>
  >;
  enrichDepsRef: React.RefObject<EnrichNodesDeps>;
  handleEdgeDelete: (edgeId: string) => void;
  onGraphChange?: () => void;
  onGraphClear?: () => void;
  resolveRootGraphRef: React.RefObject<
    (
      nodes: Node<FlowNodeData>[],
      edges: Edge[]
    ) => { nodes: Node<FlowNodeData>[]; edges: Edge[] }
  >;
  resetNavigationRef: React.RefObject<() => void>;
}

export function useGraphPersistence({
  nodes,
  edges,
  setNodes,
  setEdges,
  portsMap,
  nodeParamsRef,
  setNodeParams,
  enrichDepsRef,
  handleEdgeDelete,
  onGraphChange,
  onGraphClear,
  resolveRootGraphRef,
  resetNavigationRef,
}: UseGraphPersistenceArgs) {
  const [status, setStatus] = useState<string>("");
  const [fitViewTrigger, setFitViewTrigger] = useState(0);

  const { pipelines: pipelineInfoMap, refreshPipelines } =
    usePipelinesContext();
  const { plugins, refresh: refreshPlugins } = usePluginsContext();
  const { loraFiles, refresh: refreshLoRAs } = useLoRAsContext();
  const { version: scopeVersion } = useServerInfoContext();

  const nodesRef = useRef(nodes);
  nodesRef.current = nodes;
  const edgesRef = useRef(edges);
  edgesRef.current = edges;

  const onGraphChangeRef = useRef(onGraphChange);
  onGraphChangeRef.current = onGraphChange;

  const initialLoadDone = useRef(false);
  // Suppress onGraphChange during initial load so that restoring from
  // localStorage does not mark the graph as user-edited.
  const suppressChanges = useRef(false);

  // Fingerprint of the last auto-saved graph JSON.  Compared before writing
  // to localStorage so we skip the expensive save when nothing changed.
  const lastSavedJsonRef = useRef<string>("");

  const loadGraph = useCallback(() => {
    if (Object.keys(portsMap).length === 0) return;
    resetNavigationRef.current?.();

    const backup = loadGraphFromLocalStorage();
    if (
      backup &&
      typeof backup === "object" &&
      backup !== null &&
      "nodes" in backup &&
      "edges" in backup
    ) {
      try {
        const graphConfig = backup as Parameters<typeof graphConfigToFlow>[0];
        const { nodes: flowNodes, edges: flowEdges } = graphConfigToFlow(
          graphConfig,
          portsMap
        );
        const restoredParams = extractNodeParams(
          (backup as Record<string, unknown>).ui_state as
            | Record<string, unknown>
            | null
            | undefined
        );
        setNodeParams(restoredParams);
        const sized = resetAutoHeightNodes(flowNodes);
        const enriched = enrichNodes(sized, enrichDepsRef.current);
        suppressChanges.current = true;
        setNodes(enriched);
        setEdges(colorEdges(flowEdges, enriched, handleEdgeDelete));
        setStatus("Restored from local storage");

        const sourceNodes = flowNodes.filter(n => n.data.nodeType === "source");
        const modesToRestore = sourceNodes
          .map(n => ({
            mode: n.data.sourceMode as string | undefined,
            nodeId: n.id,
          }))
          .filter(
            (entry): entry is { mode: string; nodeId: string } =>
              !!entry.mode && entry.mode !== "video"
          );
        if (modesToRestore.length > 0) {
          setTimeout(() => {
            for (const { mode, nodeId } of modesToRestore) {
              enrichDepsRef.current.onSourceModeChangeRef.current?.(
                mode,
                nodeId
              );
            }
          }, 0);
        }

        // Allow async side-effects (e.g. source mode restore) to settle
        // before re-enabling change notifications.
        setTimeout(() => {
          suppressChanges.current = false;
        }, 0);
      } catch {
        setStatus("No graph configured");
      }
    } else {
      setStatus("No graph configured");
    }
    initialLoadDone.current = true;
  }, [
    portsMap,
    enrichDepsRef,
    handleEdgeDelete,
    resetNavigationRef,
    setEdges,
    setNodeParams,
    setNodes,
  ]);

  useEffect(() => {
    loadGraph();
  }, [loadGraph]);

  useEffect(() => {
    if (!initialLoadDone.current) return;
    if (nodes.length === 0 && edges.length === 0) return;
    if (suppressChanges.current) return;
    onGraphChangeRef.current?.();
  }, [nodes, edges]);

  useEffect(() => {
    if (!initialLoadDone.current) return;
    if (nodes.length === 0 && edges.length === 0) return;

    const timer = setTimeout(() => {
      try {
        const root = resolveRootGraphRef.current(nodes, edges);
        const graphConfig = attachNodeParams(
          flowToGraphConfig(root.nodes, root.edges),
          nodeParamsRef.current
        );
        const graphJson = JSON.stringify(graphConfig);
        if (graphJson === lastSavedJsonRef.current) return;
        lastSavedJsonRef.current = graphJson;
        saveGraphToLocalStorage(graphJson);
        setStatus(
          `Saved: ${graphConfig.nodes.length} nodes, ${graphConfig.edges.length} edges`
        );
      } catch (err) {
        const message = err instanceof Error ? err.message : "Unknown error";
        setStatus(`Save failed: ${message}`);
      }
    }, 500);

    return () => clearTimeout(timer);
  }, [nodes, edges, nodeParamsRef, resolveRootGraphRef]);

  const handleSave = useCallback(() => {
    if (nodes.length === 0 && edges.length === 0) {
      setStatus("Nothing to save");
      return;
    }
    try {
      const root = resolveRootGraphRef.current(nodes, edges);
      const graphConfig = attachNodeParams(
        flowToGraphConfig(root.nodes, root.edges),
        nodeParamsRef.current
      );
      const graphJson = JSON.stringify(graphConfig);
      lastSavedJsonRef.current = graphJson;
      saveGraphToLocalStorage(graphJson);
      trackEvent("workflow_saved", {
        node_count: graphConfig.nodes.length,
        edge_count: graphConfig.edges.length,
        surface: "graph_mode",
      });
      setStatus(
        `Saved: ${graphConfig.nodes.length} nodes, ${graphConfig.edges.length} edges`
      );
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unknown error";
      setStatus(`Save failed: ${message}`);
    }
  }, [nodes, edges, nodeParamsRef, resolveRootGraphRef]);

  useEffect(() => {
    const handler = () => {
      try {
        const currentNodes = nodesRef.current;
        const currentEdges = edgesRef.current;
        if (currentNodes.length > 0 || currentEdges.length > 0) {
          const root = resolveRootGraphRef.current(currentNodes, currentEdges);
          const graphConfig = attachNodeParams(
            flowToGraphConfig(root.nodes, root.edges),
            nodeParamsRef.current
          );
          saveGraphToLocalStorage(JSON.stringify(graphConfig));
        }
      } catch {
        // ignore
      }
    };

    window.addEventListener("beforeunload", handler);
    return () => window.removeEventListener("beforeunload", handler);
  }, [nodeParamsRef, resolveRootGraphRef]);

  const handleClear = useCallback(() => {
    resetNavigationRef.current?.();
    clearGraphFromLocalStorage();
    setNodes([]);
    setEdges([]);
    setStatus("Graph cleared");
    onGraphClear?.();
  }, [setNodes, setEdges, onGraphClear, resetNavigationRef]);

  // -- Pending import state for workflow review dialog ----------------------
  const [pendingImportWorkflow, setPendingImportWorkflow] =
    useState<ScopeWorkflow | null>(null);
  const [pendingResolutionPlan, setPendingResolutionPlan] =
    useState<WorkflowResolutionPlan | null>(null);
  const [pendingImportResolving, setPendingImportResolving] = useState(false);

  const loadGraphFromParsed = useCallback(
    (parsed: Record<string, unknown>, fileName: string) => {
      let graphConfig: Parameters<typeof graphConfigToFlow>[0];
      let importedParams: Record<string, Record<string, unknown>> | null = null;

      if (
        parsed.format === "scope-workflow" &&
        Array.isArray(parsed.pipelines)
      ) {
        const workflow = parsed as unknown as ScopeWorkflow;
        if (workflow.graph?.nodes && workflow.graph?.edges) {
          graphConfig = workflow.graph as Parameters<
            typeof graphConfigToFlow
          >[0];
        } else {
          const result = workflowToGraphConfig(workflow, {
            availableLoRAs: loraFiles,
          });
          graphConfig = result.graphConfig as Parameters<
            typeof graphConfigToFlow
          >[0];
          importedParams = result.nodeParams;
        }
      } else if (parsed.nodes && parsed.edges) {
        graphConfig = parsed as unknown as Parameters<
          typeof graphConfigToFlow
        >[0];
      } else {
        setStatus("Import failed: unrecognized format");
        return;
      }

      resetNavigationRef.current?.();
      const { nodes: flowNodes, edges: flowEdges } = graphConfigToFlow(
        graphConfig,
        portsMap
      );
      const restoredParams =
        importedParams ?? extractNodeParams(graphConfig.ui_state);
      setNodeParams(restoredParams);
      const sized = resetAutoHeightNodes(flowNodes);
      const enriched = enrichNodes(sized, enrichDepsRef.current);
      setNodes(enriched);
      setEdges(colorEdges(flowEdges, enriched, handleEdgeDelete));
      setStatus(`Imported from ${fileName}`);
      setFitViewTrigger(c => c + 1);

      const sourceNodes = flowNodes.filter(n => n.data.nodeType === "source");
      const modesToRestore = sourceNodes
        .map(n => ({
          mode: n.data.sourceMode as string | undefined,
          nodeId: n.id,
        }))
        .filter(
          (entry): entry is { mode: string; nodeId: string } => !!entry.mode
        );
      if (modesToRestore.length > 0) {
        setTimeout(() => {
          for (const { mode, nodeId } of modesToRestore) {
            enrichDepsRef.current.onSourceModeChangeRef.current?.(mode, nodeId);
          }
        }, 0);
      }
    },
    [
      portsMap,
      loraFiles,
      setNodes,
      setEdges,
      handleEdgeDelete,
      setNodeParams,
      enrichDepsRef,
      resetNavigationRef,
    ]
  );

  const handleImport = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      if (!file) return;

      const reader = new FileReader();
      reader.onload = async e => {
        try {
          const parsed = JSON.parse(e.target?.result as string);

          // If it's a scope-workflow, resolve dependencies and show review dialog
          if (
            parsed.format === "scope-workflow" &&
            Array.isArray(parsed.pipelines)
          ) {
            const workflow = parsed as ScopeWorkflow;
            setPendingImportWorkflow(workflow);
            setPendingImportResolving(true);
            try {
              const plan = await resolveWorkflow(workflow);

              // If all dependencies are already resolved, skip the review
              // dialog and load the workflow directly.
              if (
                plan.items.every(i => i.status === "ok") &&
                plan.warnings.length === 0
              ) {
                loadGraphFromParsed(
                  workflow as unknown as Record<string, unknown>,
                  workflow.metadata?.name ?? file.name
                );
                setPendingImportWorkflow(null);
                return;
              }

              setPendingResolutionPlan(plan);
            } catch (err) {
              console.error("Workflow resolution failed:", err);
              setStatus("Import failed: could not resolve dependencies");
              setPendingImportWorkflow(null);
            } finally {
              setPendingImportResolving(false);
            }
            return;
          }

          // Plain graph JSON — load directly
          if (parsed.nodes && parsed.edges) {
            loadGraphFromParsed(parsed, file.name);
          } else {
            setStatus("Import failed: unrecognized format");
          }
        } catch {
          setStatus("Import failed: invalid JSON");
        }
      };
      reader.readAsText(file);
      event.target.value = "";
    },
    [loadGraphFromParsed]
  );

  const confirmImport = useCallback(() => {
    if (!pendingImportWorkflow) return;
    loadGraphFromParsed(
      pendingImportWorkflow as unknown as Record<string, unknown>,
      pendingImportWorkflow.metadata?.name ?? "workflow"
    );
    setPendingImportWorkflow(null);
    setPendingResolutionPlan(null);
  }, [pendingImportWorkflow, loadGraphFromParsed]);

  const cancelImport = useCallback(() => {
    setPendingImportWorkflow(null);
    setPendingResolutionPlan(null);
  }, []);

  const reResolveImport = useCallback(async () => {
    if (!pendingImportWorkflow) return;
    try {
      await Promise.all([refreshPipelines(), refreshLoRAs(), refreshPlugins()]);
      const plan = await resolveWorkflow(pendingImportWorkflow);
      setPendingResolutionPlan(plan);
    } catch (err) {
      console.error("Failed to re-resolve workflow:", err);
    }
  }, [pendingImportWorkflow, refreshPipelines, refreshLoRAs, refreshPlugins]);

  const buildCurrentWorkflow = useCallback(
    (name?: string) => {
      const root = resolveRootGraphRef.current(nodes, edges);
      const graphConfig = attachNodeParams(
        flowToGraphConfig(root.nodes, root.edges),
        nodeParamsRef.current
      );

      const pluginInfoMap = new Map<string, PluginInfo>(
        plugins.map(p => [p.name, p])
      );

      return buildGraphWorkflow({
        name: name ?? `Graph Export ${new Date().toISOString().split("T")[0]}`,
        graphConfig,
        pipelineInfoMap: pipelineInfoMap ?? {},
        pluginInfoMap,
        scopeVersion: scopeVersion ?? "unknown",
        loraFiles,
      });
    },
    [
      nodes,
      edges,
      nodeParamsRef,
      resolveRootGraphRef,
      pipelineInfoMap,
      plugins,
      scopeVersion,
      loraFiles,
    ]
  );

  const handleExport = useCallback(() => {
    const workflow = buildCurrentWorkflow();

    const dataStr = JSON.stringify(workflow, null, 2);
    const blob = new Blob([dataStr], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    const safeName = workflow.metadata.name
      .replace(/[^a-zA-Z0-9_-]/g, "_")
      .toLowerCase();
    link.download = `${safeName}.scope-workflow.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
    setStatus("Graph exported");
  }, [buildCurrentWorkflow]);

  const getCurrentGraphConfig = useCallback(() => {
    const root = resolveRootGraphRef.current(
      nodesRef.current,
      edgesRef.current
    );
    return attachNodeParams(
      flowToGraphConfig(root.nodes, root.edges),
      nodeParamsRef.current
    );
  }, [nodeParamsRef, resolveRootGraphRef]);

  const getGraphNodePrompts = useCallback((): Array<{
    nodeId: string;
    text: string;
  }> => {
    const results: Array<{ nodeId: string; text: string }> = [];
    for (const node of nodesRef.current) {
      if (node.data.nodeType !== "pipeline") continue;
      const text = (nodeParamsRef.current[node.id]?.__prompt as string) || "";
      if (text) results.push({ nodeId: node.id, text });
    }
    return results;
  }, [nodeParamsRef]);

  /**
   * Extract VACE settings from VaceNode -> PipelineNode connections.
   * Returns per-pipeline-node VACE params that should be included in
   * initialParameters at stream start.
   */
  const getGraphVaceSettings = useCallback((): Array<{
    pipelineNodeId: string;
    vace_context_scale: number;
    vace_use_input_video: boolean;
    vace_ref_images?: string[];
    first_frame_image?: string;
    last_frame_image?: string;
  }> => {
    const results: Array<{
      pipelineNodeId: string;
      vace_context_scale: number;
      vace_use_input_video: boolean;
      vace_ref_images?: string[];
      first_frame_image?: string;
      last_frame_image?: string;
    }> = [];
    const currentNodes = nodesRef.current;
    const currentEdges = edgesRef.current;

    for (const edge of currentEdges) {
      const sourceParsed = parseHandleId(edge.sourceHandle);
      const targetParsed = parseHandleId(edge.targetHandle);
      if (sourceParsed?.name !== "__vace" || targetParsed?.name !== "__vace")
        continue;

      const vaceNode = currentNodes.find(n => n.id === edge.source);
      const pipelineNode = currentNodes.find(n => n.id === edge.target);
      if (!vaceNode || !pipelineNode) continue;
      if (vaceNode.data.nodeType !== "vace") continue;
      if (pipelineNode.data.nodeType !== "pipeline") continue;

      const entry: (typeof results)[number] = {
        pipelineNodeId: pipelineNode.id,
        vace_context_scale:
          typeof vaceNode.data.vaceContextScale === "number"
            ? vaceNode.data.vaceContextScale
            : 1.0,
        vace_use_input_video: false,
      };

      const refImg = (vaceNode.data.vaceRefImage as string) || "";
      if (refImg) entry.vace_ref_images = [refImg];

      const firstFrame = (vaceNode.data.vaceFirstFrame as string) || "";
      if (firstFrame) entry.first_frame_image = firstFrame;

      const lastFrame = (vaceNode.data.vaceLastFrame as string) || "";
      if (lastFrame) entry.last_frame_image = lastFrame;

      results.push(entry);
    }
    return results;
  }, []);

  /**
   * Extract LoRA settings from LoraNode -> PipelineNode connections.
   * Returns per-pipeline-node LoRA config for load_params at stream start.
   */
  const getGraphLoRASettings = useCallback((): Array<{
    pipelineNodeId: string;
    loras: Array<{ path: string; scale: number; merge_mode?: string }>;
    lora_merge_mode: string;
  }> => {
    const results: Array<{
      pipelineNodeId: string;
      loras: Array<{ path: string; scale: number; merge_mode?: string }>;
      lora_merge_mode: string;
    }> = [];
    const currentNodes = nodesRef.current;
    const currentEdges = edgesRef.current;

    for (const edge of currentEdges) {
      const sourceParsed = parseHandleId(edge.sourceHandle);
      const targetParsed = parseHandleId(edge.targetHandle);
      if (sourceParsed?.name !== "__loras" || targetParsed?.name !== "__loras")
        continue;

      const loraNode = currentNodes.find(n => n.id === edge.source);
      const pipelineNode = currentNodes.find(n => n.id === edge.target);
      if (!loraNode || !pipelineNode) continue;
      if (loraNode.data.nodeType !== "lora") continue;
      if (pipelineNode.data.nodeType !== "pipeline") continue;

      const entries =
        (loraNode.data.loras as Array<{
          path: string;
          scale: number;
          mergeMode?: string;
        }>) || [];

      const validLoras = entries
        .filter(l => l.path)
        .map(l => ({
          path: l.path,
          scale: l.scale,
          ...(l.mergeMode ? { merge_mode: l.mergeMode } : {}),
        }));

      if (validLoras.length > 0) {
        results.push({
          pipelineNodeId: pipelineNode.id,
          loras: validLoras,
          lora_merge_mode:
            (loraNode.data.loraMergeMode as string) || "permanent_merge",
        });
      }
    }
    return results;
  }, []);

  return {
    status,
    fitViewTrigger,
    handleSave,
    handleClear,
    handleImport,
    handleExport,
    buildCurrentWorkflow,
    refreshGraph: loadGraph,
    getCurrentGraphConfig,
    getGraphNodePrompts,
    getGraphVaceSettings,
    getGraphLoRASettings,
    initialLoadDone,
    pendingImportWorkflow,
    pendingResolutionPlan,
    pendingImportResolving,
    confirmImport,
    cancelImport,
    reResolveImport,
    loadGraphFromParsed,
  };
}
