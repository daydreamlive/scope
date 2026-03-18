import { useCallback, useEffect, useRef, useState } from "react";
import type { Node } from "@xyflow/react";
import type { FlowNodeData } from "../../../../lib/graphUtils";
import { extractParameterPorts } from "../../../../lib/graphUtils";
import type { PipelineSchemaInfo } from "../../../../lib/api";
import { getDefaultPromptForMode } from "../../../../data/pipelines";
import type { InputMode } from "../../../../types";

interface UsePipelineParamsArgs {
  setNodes: React.Dispatch<React.SetStateAction<Node<FlowNodeData>[]>>;
  portsMap: Record<string, { inputs: string[]; outputs: string[] }>;
  pipelineSchemas: Record<string, PipelineSchemaInfo>;
  isStreamingRef: React.RefObject<boolean>;
  nodesRef: React.RefObject<Node<FlowNodeData>[]>;
  onNodeParameterChange?: (nodeId: string, key: string, value: unknown) => void;
}

export function usePipelineParams({
  setNodes,
  portsMap,
  pipelineSchemas,
  isStreamingRef,
  nodesRef,
  onNodeParameterChange,
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

      // Pre-fill prompt default
      if (supportsPrompts && schema) {
        const defaultMode = (schema.default_mode ?? "text") as InputMode;
        const defaultPrompt = getDefaultPromptForMode(defaultMode);
        setNodeParams(prev => ({
          ...prev,
          [nodeId]: { ...(prev[nodeId] || {}), __prompt: defaultPrompt },
        }));
      }

      setNodes(nds =>
        nds.map(n => {
          if (n.id !== nodeId) return n;
          const ports =
            newPipelineId && portsMap ? portsMap[newPipelineId] : null;
          const parameterInputs = schema ? extractParameterPorts(schema) : [];
          const supportsCacheManagement =
            schema?.supports_cache_management ?? false;
          const supportsVace = schema?.supports_vace ?? false;
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
              supportsCacheManagement,
              supportsVace,
            },
          };
        })
      );
    },
    [setNodes, portsMap, pipelineSchemas]
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
  };
}
