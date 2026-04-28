import { useEffect, useState } from "react";
import { fetchNodeDefinitions, type NodeDefinitionDto } from "../lib/api";

/** Fetches /api/v1/nodes/definitions once. Used by AddNodeModal and the
 *  graph pane context menu so plugin-registered nodes are surfaced in
 *  search without each consumer re-fetching. */
export function useNodeDefinitions(enabled = true) {
  const [defs, setDefs] = useState<NodeDefinitionDto[]>([]);

  useEffect(() => {
    if (!enabled) return;
    const controller = new AbortController();
    fetchNodeDefinitions({ signal: controller.signal })
      .then(data => {
        if (controller.signal.aborted) return;
        setDefs(data.nodes ?? []);
      })
      .catch((err: unknown) => {
        if (err instanceof DOMException && err.name === "AbortError") return;
        console.warn("Failed to fetch node definitions:", err);
      });
    return () => controller.abort();
  }, [enabled]);

  return defs;
}
