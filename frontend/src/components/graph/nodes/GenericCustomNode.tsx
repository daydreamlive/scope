/**
 * Generic renderer for plugin-provided custom node types.
 *
 * Dynamically generates input/output handles and basic UI controls
 * based on the NodeDefinition fetched from `GET /api/v1/nodes`.
 * Built-in nodes continue to use their dedicated rich components;
 * this component is only used for plugin nodes that don't have a
 * dedicated frontend implementation.
 */

import { Handle, Position } from "@xyflow/react";
import type { NodeProps, Node } from "@xyflow/react";
import type { FlowNodeData } from "../../../lib/graphUtils";
import { buildHandleId } from "../../../lib/graphUtils";
import { NodeCard, NodeHeader, NodeBody, NodeParamRow } from "../ui";

type GenericCustomNodeType = Node<FlowNodeData, string>;

/** Port definition as returned by the backend `/api/v1/nodes` endpoint. */
export interface PortDef {
  name: string;
  port_type: string;
  default_value?: unknown;
  label?: string;
  min_value?: number;
  max_value?: number;
}

/** Node definition from the backend. */
export interface BackendNodeDefinition {
  node_type_id: string;
  display_name: string;
  category: string;
  description?: string;
  inputs: PortDef[];
  outputs: PortDef[];
  is_animated?: boolean;
  is_stream_node?: boolean;
}

const PORT_TYPE_COLORS: Record<string, string> = {
  number: "#38bdf8",
  string: "#fbbf24",
  boolean: "#34d399",
  stream: "#f97316",
  any: "#9ca3af",
};

export function GenericCustomNode({
  data,
  selected,
}: NodeProps<GenericCustomNodeType>) {
  const definition = data._backendDefinition as
    | BackendNodeDefinition
    | undefined;

  const displayName =
    data.customTitle || definition?.display_name || data.nodeType || "Custom";
  const inputs = definition?.inputs ?? [];
  const outputs = definition?.outputs ?? [];

  return (
    <NodeCard selected={selected}>
      <NodeHeader title={displayName} />

      {/* Input handles */}
      {inputs.map(port => (
        <div
          key={`in-${port.name}`}
          className="relative"
          style={{ minHeight: 24 }}
        >
          <Handle
            type="target"
            position={Position.Left}
            id={buildHandleId("param", port.name)}
            style={{
              top: "50%",
              background:
                PORT_TYPE_COLORS[port.port_type] || PORT_TYPE_COLORS.any,
            }}
          />
          <NodeParamRow label={port.label || port.name}>
            <span className="text-xs text-zinc-400 truncate">
              {data[port.name] != null ? String(data[port.name]) : "—"}
            </span>
          </NodeParamRow>
        </div>
      ))}

      {/* Output handles */}
      {outputs.map(port => (
        <div
          key={`out-${port.name}`}
          className="relative"
          style={{ minHeight: 24 }}
        >
          <Handle
            type="source"
            position={Position.Right}
            id={buildHandleId("param", port.name)}
            style={{
              top: "50%",
              background:
                PORT_TYPE_COLORS[port.port_type] || PORT_TYPE_COLORS.any,
            }}
          />
          <NodeParamRow label={port.label || port.name}>
            <span className="text-xs text-zinc-400 truncate">
              {data[port.name] != null ? String(data[port.name]) : "—"}
            </span>
          </NodeParamRow>
        </div>
      ))}

      {inputs.length === 0 && outputs.length === 0 && (
        <NodeBody>
          <span className="text-xs text-zinc-500 italic">
            {definition?.description || "No ports defined"}
          </span>
        </NodeBody>
      )}
    </NodeCard>
  );
}
