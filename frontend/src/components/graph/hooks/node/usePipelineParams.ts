import { useCallback, useEffect, useRef, useState } from "react";
import type { Edge, Node } from "@xyflow/react";
import type { FlowNodeData } from "../../../../lib/graphUtils";
import {
  extractParameterPorts,
  generateNodeId,
} from "../../../../lib/graphUtils";
import type {
  PipelineSchemaInfo,
  HardwareInfoResponse,
} from "../../../../lib/api";
import { getDefaultPromptForMode } from "../../../../data/pipelines";
import type { InputMode } from "../../../../types";

interface UsePipelineParamsArgs {
  setNodes: React.Dispatch<React.SetStateAction<Node<FlowNodeData>[]>>;
  setEdges: React.Dispatch<React.SetStateAction<Edge[]>>;
  portsMap: Record<string, { inputs: string[]; outputs: string[] }>;
  pipelineSchemas: Record<string, PipelineSchemaInfo>;
  isStreamingRef: React.RefObject<boolean>;
  nodesRef: React.RefObject<Node<FlowNodeData>[]>;
  onNodeParameterChange?: (nodeId: string, key: string, value: unknown) => void;
  hardwareInfo: HardwareInfoResponse | null;
}

export function usePipelineParams({
  setNodes,
  setEdges,
  portsMap,
  pipelineSchemas,
  isStreamingRef,
  nodesRef,
  onNodeParameterChange,
  hardwareInfo,
}: UsePipelineParamsArgs) {
  const [nodeParams, setNodeParams] = useState<
    Record<string, Record<string, unknown>>
  >({});

  const nodeParamsRef = useRef(nodeParams);
  nodeParamsRef.current = nodeParams;

  const onNodeParamChangeRef = useRef(onNodeParameterChange);
  onNodeParamChangeRef.current = onNodeParameterChange;

  // Resolve backend node ID (identity for now)
  const resolveBackendId = useCallback((nodeId: string): string => {
    return nodeId;
  }, []);

  const handlePipelineSelect = useCallback(
    (nodeId: string, newPipelineId: string | null) => {
      const schema = newPipelineId ? pipelineSchemas[newPipelineId] : null;
      const supportsPrompts = schema?.supports_prompts ?? false;

      // Compute new node ID to match the pipeline, keeping it unique
      const existingIds = new Set(
        nodesRef.current.filter(n => n.id !== nodeId).map(n => n.id)
      );
      const newNodeId = newPipelineId
        ? generateNodeId(newPipelineId, existingIds)
        : nodeId;
      const needsRename = newNodeId !== nodeId;

      // Pre-fill prompt default and recommended quantization
      const paramOverrides: Record<string, unknown> = {};

      if (supportsPrompts && schema) {
        const existing = nodeParamsRef.current[nodeId]?.__prompt;
        if (!existing) {
          const defaultMode = (schema.default_mode ?? "text") as InputMode;
          paramOverrides.__prompt = getDefaultPromptForMode(defaultMode);
        }
      }

      // Set recommended quantization based on VRAM (same logic as perform mode)
      const vramThreshold =
        schema?.recommended_quantization_vram_threshold ?? null;
      if (
        vramThreshold !== null &&
        vramThreshold !== undefined &&
        hardwareInfo?.vram_gb !== null &&
        hardwareInfo?.vram_gb !== undefined
      ) {
        paramOverrides.quantization =
          hardwareInfo.vram_gb > vramThreshold ? null : "fp8_e4m3fn";
      } else {
        // No recommendation from pipeline: reset quantization to null
        paramOverrides.quantization = null;
      }

      // Migrate nodeParams from old key to new key, merging overrides
      setNodeParams(prev => {
        const { [nodeId]: oldParams, ...rest } = prev;
        return {
          ...rest,
          [newNodeId]: { ...(oldParams || {}), ...paramOverrides },
        };
      });

      setNodes(nds =>
        nds.map(n => {
          if (n.id !== nodeId) return n;
          const ports =
            newPipelineId && portsMap ? portsMap[newPipelineId] : null;
          const parameterInputs = schema ? extractParameterPorts(schema) : [];
          const supportsCacheManagement =
            schema?.supports_cache_management ?? false;
          const supportsVace = schema?.supports_vace ?? false;
          const supportsLoRA = schema?.supports_lora ?? false;
          const newStyle = { ...n.style };
          delete newStyle.height;
          return {
            ...n,
            id: newNodeId,
            style: newStyle,
            height: undefined,
            measured: undefined,
            data: {
              ...n.data,
              pipelineId: newPipelineId,
              label: newPipelineId || newNodeId,
              streamInputs: ports?.inputs ?? ["video"],
              streamOutputs: ports?.outputs ?? ["video"],
              parameterInputs,
              supportsPrompts,
              supportsCacheManagement,
              supportsVace,
              supportsLoRA,
            },
          };
        })
      );

      // Update edges that reference the old node ID
      if (needsRename) {
        setEdges(eds =>
          eds.map(e => {
            const srcMatch = e.source === nodeId;
            const tgtMatch = e.target === nodeId;
            if (!srcMatch && !tgtMatch) return e;
            const newSource = srcMatch ? newNodeId : e.source;
            const newTarget = tgtMatch ? newNodeId : e.target;
            return {
              ...e,
              id: `reactflow__edge-${newSource}${e.sourceHandle ?? ""}-${newTarget}${e.targetHandle ?? ""}`,
              source: newSource,
              target: newTarget,
            };
          })
        );
      }
    },
    [setNodes, setEdges, portsMap, pipelineSchemas, hardwareInfo, nodesRef]
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

  // Apply parameter updates from external sources (REST API, OSC, MCP)
  // without sending them back to the backend (they already have the values).
  const applyExternalNodeParams = useCallback(
    (params: Record<string, unknown>, targetNodeId?: string) => {
      const patch: Record<string, unknown> = {};
      for (const [k, v] of Object.entries(params)) {
        if (k === "node_id") continue;
        if (k === "prompts") {
          const arr = v as Array<{ text: string; weight: number }>;
          if (Array.isArray(arr) && arr.length > 0) {
            patch.__prompt = arr[0].text;
          }
          continue;
        }
        patch[k] = v;
      }
      if (Object.keys(patch).length === 0) return;

      setNodeParams(prev => {
        const next = { ...prev };
        if (targetNodeId) {
          next[targetNodeId] = { ...(next[targetNodeId] || {}), ...patch };
        } else {
          for (const node of nodesRef.current) {
            if (node.data.nodeType !== "pipeline") continue;
            next[node.id] = { ...(next[node.id] || {}), ...patch };
          }
        }
        return next;
      });
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    []
  );

  // Prompt handling

  const sendPromptToBackend = useCallback(
    (nodeId: string) => {
      if (!isStreamingRef.current) return;
      const text = (nodeParamsRef.current[nodeId]?.__prompt as string) || "";
      onNodeParamChangeRef.current?.(resolveBackendId(nodeId), "prompts", [
        { text, weight: 100 },
      ]);
    },
    [resolveBackendId, isStreamingRef]
  );

  const handlePromptChange = useCallback((nodeId: string, text: string) => {
    setNodeParams(prev => ({
      ...prev,
      [nodeId]: { ...(prev[nodeId] || {}), __prompt: text },
    }));
  }, []);

  const handlePromptSubmit = useCallback(
    (nodeId: string) => {
      sendPromptToBackend(nodeId);
    },
    [sendPromptToBackend]
  );

  // Flush prompts to backend when streaming starts
  const wasStreamingRef = useRef(false);
  useEffect(() => {
    const nowStreaming = isStreamingRef.current;
    if (nowStreaming && !wasStreamingRef.current) {
      const timerId = setTimeout(() => {
        const currentNodes = nodesRef.current;
        const currentParams = nodeParamsRef.current;
        for (const node of currentNodes) {
          if (node.data.nodeType !== "pipeline") continue;
          const prompt = (currentParams[node.id]?.__prompt as string) || "";
          if (!prompt) continue;
          onNodeParamChangeRef.current?.(resolveBackendId(node.id), "prompts", [
            { text: prompt, weight: 100 },
          ]);
        }
      }, 500);
      wasStreamingRef.current = nowStreaming;
      return () => clearTimeout(timerId);
    }
    wasStreamingRef.current = nowStreaming;
    // Trigger on streaming state change
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isStreamingRef.current, resolveBackendId]);

  // Sync nodeParams → node data
  useEffect(() => {
    setNodes(nds => {
      if (nds.length === 0) return nds; // nothing to sync
      let changed = false;
      const result = nds.map(n => {
        if (n.data.nodeType === "pipeline") {
          const vals = nodeParams[n.id] || {};
          if (n.data.parameterValues === vals) return n;
          changed = true;
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
      });
      return changed ? result : nds;
    });
  }, [nodeParams, setNodes]);

  return {
    nodeParams,
    setNodeParams,
    nodeParamsRef,
    handlePipelineSelect,
    handleNodeParameterChange,
    handlePromptChange,
    handlePromptSubmit,
    resolveBackendId,
    onNodeParamChangeRef,
    applyExternalNodeParams,
  };
}
