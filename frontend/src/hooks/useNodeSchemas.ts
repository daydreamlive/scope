/**
 * Fetch and cache backend node type schemas.
 *
 * Calls `GET /api/v1/nodes/types` once on mount and provides a lookup
 * by `node_type_id` for the generic BackendNode renderer.
 */

import { useEffect, useState, useCallback, useRef } from "react";

export interface ConnectorSchema {
  name: string;
  type: "float" | "int" | "string" | "bool" | "trigger";
  direction: "input" | "output";
  default?: unknown;
  ui?: Record<string, unknown> | null;
}

export interface NodeTypeSchema {
  node_type_id: string;
  node_name: string;
  node_description: string;
  node_version: string;
  node_category: string;
  inputs: ConnectorSchema[];
  outputs: ConnectorSchema[];
  dynamic_ports: boolean;
}

export function useNodeSchemas() {
  const [schemas, setSchemas] = useState<NodeTypeSchema[]>([]);
  const [loading, setLoading] = useState(true);
  const fetchedRef = useRef(false);

  const fetchSchemas = useCallback(async () => {
    try {
      const res = await fetch("/api/v1/nodes/types");
      if (res.ok) {
        const data: NodeTypeSchema[] = await res.json();
        setSchemas(data);
      }
    } catch {
      // Server may not be available yet; schemas will be empty
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (fetchedRef.current) return;
    fetchedRef.current = true;
    fetchSchemas();
  }, [fetchSchemas]);

  const getSchema = useCallback(
    (nodeTypeId: string): NodeTypeSchema | undefined =>
      schemas.find(s => s.node_type_id === nodeTypeId),
    [schemas]
  );

  const schemaMap = schemas.reduce<Record<string, NodeTypeSchema>>((acc, s) => {
    acc[s.node_type_id] = s;
    return acc;
  }, {});

  return { schemas, schemaMap, getSchema, loading, refetch: fetchSchemas };
}
