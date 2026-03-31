import { Handle, Position } from "@xyflow/react";
import type { NodeProps, Node } from "@xyflow/react";
import { useCallback, useMemo } from "react";
import type { FlowNodeData } from "../../../lib/graphUtils";
import { buildHandleId } from "../../../lib/graphUtils";
import { useNodeData } from "../hooks/node/useNodeData";
import { useNodeCollapse } from "../hooks/node/useNodeCollapse";
import { useHandlePositions } from "../hooks/node/useHandlePositions";
import {
  NodeCard,
  NodeHeader,
  NodeBody,
  NodeParamRow,
  NodePill,
  NodePillInput,
  NODE_TOKENS,
  collapsedHandleStyle,
} from "../ui";
import { PARAM_TYPE_COLORS, COLOR_DEFAULT, COLOR_BOOLEAN } from "../nodeColors";
import type { ConnectorSchema } from "../../../hooks/useNodeSchemas";

type BackendNodeType = Node<FlowNodeData, "backend_node">;

const CONNECTOR_TYPE_COLOR: Record<string, string> = {
  float: PARAM_TYPE_COLORS.float ?? COLOR_DEFAULT,
  int: PARAM_TYPE_COLORS.int ?? COLOR_DEFAULT,
  string: PARAM_TYPE_COLORS.string ?? COLOR_DEFAULT,
  bool: COLOR_BOOLEAN,
  trigger: "#f97316", // orange-500
};

function colorFor(type: string): string {
  return CONNECTOR_TYPE_COLOR[type] ?? COLOR_DEFAULT;
}

export function BackendNode({
  id,
  data,
  selected,
}: NodeProps<BackendNodeType>) {
  const { updateData } = useNodeData(id);
  const { collapsed, toggleCollapse } = useNodeCollapse();

  const schema = data.backendNodeSchema;
  const nodeState = (data.backendNodeState ?? {}) as Record<string, unknown>;
  const sendInput = data.onBackendNodeInput as
    | ((name: string, value: unknown) => void)
    | undefined;

  const inputPorts: ConnectorSchema[] = useMemo(
    () =>
      (schema?.inputs ?? []).filter(
        (c: ConnectorSchema) => c.direction === "input"
      ),
    [schema]
  );
  const outputPorts: ConnectorSchema[] = useMemo(
    () =>
      (schema?.outputs ?? []).filter(
        (c: ConnectorSchema) => c.direction === "output"
      ),
    [schema]
  );

  const allRows = useMemo(
    () => [...inputPorts.map(p => p.name), ...outputPorts.map(p => p.name)],
    [inputPorts, outputPorts]
  );
  const { setRowRef, rowPositions } = useHandlePositions(allRows);

  const handleTrigger = useCallback(
    (name: string) => {
      sendInput?.(name, true);
    },
    [sendInput]
  );

  const handleValueChange = useCallback(
    (name: string, value: unknown) => {
      sendInput?.(name, value);
    },
    [sendInput]
  );

  const nodeName = schema?.node_name ?? data.label ?? "Backend Node";

  return (
    <NodeCard selected={selected} collapsed={collapsed}>
      <NodeHeader
        title={data.customTitle || nodeName}
        onTitleChange={t => updateData({ customTitle: t })}
        collapsed={collapsed}
        onCollapseToggle={toggleCollapse}
      />

      {!collapsed && (
        <NodeBody withGap>
          {/* Input ports */}
          {inputPorts.map(port => (
            <div key={port.name} ref={setRowRef(port.name)}>
              <NodeParamRow label={port.name}>
                {port.type === "trigger" ? (
                  <button
                    className="px-2 py-0.5 rounded text-[10px] font-medium bg-orange-500/20 text-orange-400 hover:bg-orange-500/30 transition-colors"
                    onClick={() => handleTrigger(port.name)}
                  >
                    {port.ui?.widget === "play_button" ? "▶ Play" : "Fire"}
                  </button>
                ) : port.type === "bool" ? (
                  <button
                    className="px-2 py-0.5 rounded text-[10px] font-medium transition-colors"
                    style={{
                      backgroundColor: nodeState[port.name]
                        ? `${COLOR_BOOLEAN}33`
                        : "#333",
                      color: nodeState[port.name] ? COLOR_BOOLEAN : "#666",
                    }}
                    onClick={() =>
                      handleValueChange(port.name, !nodeState[port.name])
                    }
                  >
                    {nodeState[port.name] ? "ON" : "OFF"}
                  </button>
                ) : (
                  <NodePillInput
                    type={
                      port.type === "float" || port.type === "int"
                        ? "number"
                        : "text"
                    }
                    value={
                      (nodeState[port.name] ?? port.default ?? "") as
                        | string
                        | number
                    }
                    onChange={v =>
                      handleValueChange(
                        port.name,
                        port.type === "float"
                          ? parseFloat(String(v))
                          : port.type === "int"
                            ? parseInt(String(v), 10)
                            : v
                      )
                    }
                  />
                )}
              </NodeParamRow>
            </div>
          ))}

          {/* Output ports */}
          {outputPorts.map(port => (
            <div key={port.name} ref={setRowRef(port.name)}>
              <div className={NODE_TOKENS.paramRow}>
                <span className={NODE_TOKENS.labelText}>{port.name}</span>
                <NodePill className="opacity-75">
                  {formatValue(nodeState[port.name])}
                </NodePill>
              </div>
            </div>
          ))}
        </NodeBody>
      )}

      {/* Input handles */}
      {inputPorts.map(port => (
        <Handle
          key={`in-${port.name}`}
          type="target"
          position={Position.Left}
          id={buildHandleId("param", port.name)}
          className={
            collapsed
              ? "!w-0 !h-0 !border-0 !min-w-0 !min-h-0"
              : "!w-2.5 !h-2.5 !border-0"
          }
          style={
            collapsed
              ? { ...collapsedHandleStyle("left"), opacity: 0 }
              : {
                  top: rowPositions[port.name] ?? 0,
                  left: 0,
                  backgroundColor: colorFor(port.type),
                }
          }
        />
      ))}

      {/* Output handles */}
      {outputPorts.map(port => (
        <Handle
          key={`out-${port.name}`}
          type="source"
          position={Position.Right}
          id={buildHandleId("param", port.name)}
          className={
            collapsed
              ? "!w-0 !h-0 !border-0 !min-w-0 !min-h-0"
              : "!w-2.5 !h-2.5 !border-0"
          }
          style={
            collapsed
              ? { ...collapsedHandleStyle("right"), opacity: 0 }
              : {
                  top: rowPositions[port.name] ?? 0,
                  right: 0,
                  backgroundColor: colorFor(port.type),
                }
          }
        />
      ))}
    </NodeCard>
  );
}

function formatValue(v: unknown): string {
  if (v === undefined || v === null) return "—";
  if (typeof v === "boolean") return v ? "true" : "false";
  if (typeof v === "number")
    return Number.isInteger(v) ? String(v) : v.toFixed(3);
  return String(v);
}
