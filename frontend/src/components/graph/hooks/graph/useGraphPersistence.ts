import { useCallback, useEffect, useRef, useState } from "react";
import type { Edge, Node } from "@xyflow/react";
import {
  graphConfigToFlow,
  flowToGraphConfig,
  workflowToGraphConfig,
  parseHandleId,
} from "../../../../lib/graphUtils";
import type { FlowNodeData } from "../../../../lib/graphUtils";
import type { PluginInfo, NodeDefinitionDto } from "../../../../lib/api";
import { resolveWorkflow, fetchNodeDefinitions } from "../../../../lib/api";
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

/**
 * After loading or importing a workflow, fetch `/api/v1/nodes/definitions`
 * and hydrate each custom_node flow node with its inputs/outputs/params/
 * display metadata. Saved workflows only persist `node_type_id` and
 * `params`, so the port definitions have to be re-attached before render.
 * User-supplied param values override the defaults from the definition.
 *
 * Accepts an AbortSignal so rapid reloads / unmounts can cancel an
 * in-flight fetch before its setNodes callback stomps on newer state.
 */
function hydrateCustomNodeDefinitions(
  nodes: Node<FlowNodeData>[],
  setNodes: React.Dispatch<React.SetStateAction<Node<FlowNodeData>[]>>,
  signal: AbortSignal,
  onDefinitionsLoaded?: (defMap: Map<string, NodeDefinitionDto>) => void
): void {
  const customFlowNodes = nodes.filter(
    n => n.data.nodeType === "custom_node" && !n.data.customNodeInputs
  );
  if (customFlowNodes.length === 0) return;
  fetchNodeDefinitions({ signal })
    .then(data => {
      if (signal.aborted) return;
      const defMap = new Map<string, NodeDefinitionDto>(
        data.nodes.map(d => [d.node_type_id, d])
      );
      onDefinitionsLoaded?.(defMap);
      setNodes(prev => {
        if (signal.aborted) return prev;
        return prev.map(n => {
          if (
            n.data.nodeType !== "custom_node" ||
            !n.data.customNodeTypeId ||
            n.data.customNodeInputs
          ) {
            return n;
          }
          const def = defMap.get(n.data.customNodeTypeId);
          if (!def) return n;
          return {
            ...n,
            data: {
              ...n.data,
              customNodeDisplayName: def.display_name,
              customNodeCategory: def.category,
              customNodeInputs: def.inputs ?? [],
              customNodeOutputs: def.outputs ?? [],
              customNodeParamDefs: def.params ?? [],
              customNodeParams: {
                ...Object.fromEntries(
                  (def.params ?? [])
                    .filter(p => p.default != null)
                    .map(p => [p.name, p.default] as const)
                ),
                // User-edited values take precedence over definition defaults
                ...(n.data.customNodeParams || {}),
              },
            },
          };
        });
      });
    })
    .catch((err: unknown) => {
      if (err instanceof DOMException && err.name === "AbortError") return;
      // custom nodes just won't be hydrated; render falls back to placeholders
    });
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

  // AbortController for in-flight custom-node hydration fetches. Aborted
  // before each new hydrate and on unmount so a stale /api/v1/nodes/definitions
  // response can't overwrite newer nodes state.
  const hydrateAbortRef = useRef<AbortController | null>(null);
  // Cache of node definitions from the most recent hydrate. Used by the
  // export path to attribute custom nodes to their providing plugin.
  const nodeDefinitionsRef = useRef<Map<string, NodeDefinitionDto>>(new Map());
  const startHydrate = useCallback(
    (initialNodes: Node<FlowNodeData>[]) => {
      hydrateAbortRef.current?.abort();
      const controller = new AbortController();
      hydrateAbortRef.current = controller;
      hydrateCustomNodeDefinitions(
        initialNodes,
        setNodes,
        controller.signal,
        defMap => {
          nodeDefinitionsRef.current = defMap;
        }
      );
    },
    [setNodes]
  );
  useEffect(() => {
    return () => {
      hydrateAbortRef.current?.abort();
    };
  }, []);

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

        startHydrate(enriched);

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
    startHydrate,
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
            portsMap,
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

      // Restore non-default source modes that need external setup
      // (camera permission, Spout/NDI/Syphon hardware connection). File mode
      // ("video") is the default and is handled by SourceNode's auto-init
      // effect — re-dispatching it here would race with and cancel the init
      // since handlePerNodeSourceModeChange unconditionally stops any existing
      // stream for the node.
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
            enrichDepsRef.current.onSourceModeChangeRef.current?.(mode, nodeId);
          }
        }, 0);
      }

      startHydrate(enriched);
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
      startHydrate,
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
        nodeDefinitionMap: nodeDefinitionsRef.current,
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
    // Merge parameterValues from node data (graph connections written by the
    // RAF loop in useValueForwarding) with nodeParams (manual edits).  Manual
    // edits take precedence so explicit user changes aren't overwritten.
    const mergedParams: Record<string, Record<string, unknown>> = {};
    for (const node of root.nodes) {
      if (node.data.nodeType !== "pipeline") continue;
      const fromData =
        (node.data.parameterValues as Record<string, unknown>) ?? {};
      const fromState = nodeParamsRef.current[node.id] ?? {};
      const merged = { ...fromData, ...fromState };
      if (Object.keys(merged).length > 0) {
        mergedParams[node.id] = merged;
      }
    }
    // Preserve non-pipeline node params from nodeParamsRef
    for (const [nodeId, bag] of Object.entries(nodeParamsRef.current)) {
      if (!mergedParams[nodeId]) {
        mergedParams[nodeId] = bag;
      }
    }
    return attachNodeParams(
      flowToGraphConfig(root.nodes, root.edges),
      mergedParams
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
