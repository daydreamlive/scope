import type { Node } from "@xyflow/react";
import type { FlowNodeData } from "../../../../lib/graphUtils";

/**
 * Previously this hook synced OutputNode configs to the backend via the
 * generic `output_sinks` parameter. That path only supports one sink per
 * type, so multiple Syphon/Spout/NDI outputs were collapsed.
 *
 * OutputNodes are now emitted as proper backend sink nodes (with sink_mode
 * and sink_name) inside flowToGraphConfig(). The backend's multi-sink
 * system (_setup_multi_output_sinks) handles them, supporting multiple
 * outputs of the same type with correct per-node frame routing.
 *
 * This hook is kept as a no-op to avoid breaking callers.
 */
export function useOutputSinkSync(
  _nodes: Node<FlowNodeData>[],
  _onOutputSinkBulkChangeRef: React.RefObject<
    | ((sinks: Record<string, { enabled: boolean; name: string }>) => void)
    | undefined
  >
) {
  // No-op: output sinks are now handled via the graph's sink nodes.
}
