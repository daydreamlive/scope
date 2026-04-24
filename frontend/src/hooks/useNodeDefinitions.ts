import { useState, useEffect, useCallback, useMemo, useRef } from "react";
import { fetchNodeDefinitions } from "@/lib/api";
import type { NodeDefinitionDto } from "@/lib/api";
import { usePipelinesContext } from "@/contexts/PipelinesContext";

export interface UseNodeDefinitionsReturn {
  nodes: NodeDefinitionDto[];
  customNodes: NodeDefinitionDto[];
  refresh: () => Promise<void>;
}

export function useNodeDefinitions(): UseNodeDefinitionsReturn {
  const [nodes, setNodes] = useState<NodeDefinitionDto[]>([]);
  const { pipelinesVersion } = usePipelinesContext();
  const lastPayloadRef = useRef<string>("");

  const refresh = useCallback(async (signal?: AbortSignal) => {
    try {
      const data = await fetchNodeDefinitions({ signal });
      if (signal?.aborted) return;
      const next = data.nodes ?? [];
      // Skip the state update when the payload is identical so downstream
      // useMemos that depend on `nodes` identity don't churn on every refresh.
      const serialized = JSON.stringify(next);
      if (serialized === lastPayloadRef.current) return;
      lastPayloadRef.current = serialized;
      setNodes(next);
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") return;
      console.warn("Failed to fetch node definitions:", err);
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    refresh(controller.signal);
    return () => controller.abort();
  }, [refresh, pipelinesVersion]);

  const customNodes = useMemo(
    () =>
      nodes.filter(
        n => n.pipeline_meta == null && n.node_type_id !== "scheduler"
      ),
    [nodes]
  );

  const refreshPublic = useCallback(() => refresh(), [refresh]);

  return { nodes, customNodes, refresh: refreshPublic };
}
