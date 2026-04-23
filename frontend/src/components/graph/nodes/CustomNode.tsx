import { Handle, Position } from "@xyflow/react";
import type { NodeProps, Node } from "@xyflow/react";
import type { FlowNodeData } from "../../../lib/graphUtils";
import {
  customNodeInputHandleId,
  customNodeOutputHandleId,
} from "../../../lib/graphUtils";
import { useNodeData } from "../hooks/node/useNodeData";
import { useNodeCollapse } from "../hooks/node/useNodeCollapse";
import { useHandlePositions } from "../hooks/node/useHandlePositions";
import { NodeCard, NodeHeader, NodeBody, collapsedHandleStyle } from "../ui";

type CustomNodeType = Node<FlowNodeData, "custom_node">;

/* Port type -> color mapping for custom types */
const PORT_COLORS: Record<string, string> = {
  audio: "#22c55e",
  video: "#eeeeee",
  number: "#38bdf8",
  string: "#fbbf24",
  boolean: "#34d399",
  trigger: "#f97316",
  latent: "#a855f7",
  model: "#f59e0b",
  vae: "#f59e0b",
  clip: "#f59e0b",
  conditioning: "#3b82f6",
  semantic_hints: "#06b6d4",
  config: "#6b7280",
  curve: "#ec4899",
  mask: "#ef4444",
  lora: "#f472b6",
};

function portColor(portType: string): string {
  return PORT_COLORS[portType] ?? "#9ca3af";
}

export function CustomNode({ id, data, selected }: NodeProps<CustomNodeType>) {
  const { updateData } = useNodeData(id);
  const { collapsed, toggleCollapse } = useNodeCollapse();

  const inputs = data.customNodeInputs ?? [];
  const outputs = data.customNodeOutputs ?? [];
  const params = data.customNodeParamDefs ?? [];

  // Edit-time helper: update the React Flow node data (so the widget
  // reflects the new value) and, when the node is connected to a running
  // backend, push the same change through so the worker picks it up.
  const setParam = (name: string, value: unknown) => {
    updateData({
      customNodeParams: { ...data.customNodeParams, [name]: value },
    });
    data.onCustomNodeParamChange?.(name, value);
  };
  const displayName =
    data.customTitle ||
    data.customNodeDisplayName ||
    data.customNodeTypeId ||
    "Custom Node";

  // Measure each port row so handles can be positioned from DOM offsetTop
  // rather than hard-coded pixel math. Re-measures when port lists change.
  const { setRowRef, rowPositions } = useHandlePositions([
    collapsed,
    inputs.map(p => p.name).join("|"),
    outputs.map(p => p.name).join("|"),
    params.length,
  ]);

  return (
    <NodeCard
      selected={selected}
      collapsed={collapsed}
      autoMinHeight={!collapsed}
    >
      <NodeHeader
        title={displayName}
        onTitleChange={t => updateData({ customTitle: t })}
        collapsed={collapsed}
        onCollapseToggle={toggleCollapse}
      />
      {!collapsed && (
        <NodeBody>
          {/* Show input ports */}
          {inputs.length > 0 && (
            <div className="flex flex-col gap-0.5 px-2 py-1">
              {inputs.map(p => (
                <div
                  key={p.name}
                  ref={setRowRef(`in_${p.name}`)}
                  className="text-[11px] text-zinc-400 flex items-center gap-1"
                >
                  {p.name}
                </div>
              ))}
            </div>
          )}
          {/* Show output ports */}
          {outputs.length > 0 && (
            <div className="flex flex-col gap-0.5 px-2 py-1">
              {outputs.map(p => (
                <div
                  key={p.name}
                  ref={setRowRef(`out_${p.name}`)}
                  className="text-[11px] text-zinc-400 flex items-center gap-1 justify-end"
                >
                  {p.name}
                </div>
              ))}
            </div>
          )}
          {/* Parameter widgets (ComfyUI-style editable params) */}
          {params.length > 0 && (
            <div className="flex flex-col gap-1 px-2 py-1 border-t border-zinc-800">
              {params.map(p => {
                const val = data.customNodeParams?.[p.name] ?? p.default ?? "";
                return (
                  <div
                    key={p.name}
                    className="flex items-center justify-between gap-2 text-[11px]"
                  >
                    <span
                      className="text-zinc-500 shrink-0"
                      title={p.description || p.name}
                    >
                      {p.description || p.name}
                    </span>
                    {p.param_type === "select" &&
                    Array.isArray(p.ui?.options) ? (
                      <select
                        className="bg-zinc-900 text-zinc-200 rounded px-1 py-0.5 text-[11px] max-w-[130px]"
                        value={String(val)}
                        onChange={e => setParam(p.name, e.target.value)}
                      >
                        {(p.ui?.options as string[]).map(o => (
                          <option key={o} value={o}>
                            {o}
                          </option>
                        ))}
                      </select>
                    ) : p.param_type === "boolean" ? (
                      <input
                        type="checkbox"
                        checked={Boolean(val)}
                        onChange={e => setParam(p.name, e.target.checked)}
                        className="accent-blue-500"
                      />
                    ) : p.param_type === "number" ? (
                      <input
                        type="number"
                        className="bg-zinc-900 text-zinc-200 rounded px-1 py-0.5 text-[11px] w-[80px]"
                        value={Number(val)}
                        min={(p.ui?.min as number | undefined) ?? undefined}
                        max={(p.ui?.max as number | undefined) ?? undefined}
                        step={(p.ui?.step as number | undefined) ?? undefined}
                        onChange={e => setParam(p.name, Number(e.target.value))}
                      />
                    ) : (
                      <input
                        type="text"
                        className="bg-zinc-900 text-zinc-200 rounded px-1 py-0.5 text-[11px] max-w-[130px]"
                        value={String(val)}
                        onChange={e => setParam(p.name, e.target.value)}
                      />
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </NodeBody>
      )}

      {/* Input handles (left side). When expanded, positions come from the
          port row's measured offsetTop; when collapsed, all handles overlap
          at the vertical centre of the collapsed pill. */}
      {inputs.map(p => (
        <Handle
          key={`in-${p.name}`}
          type="target"
          position={Position.Left}
          id={customNodeInputHandleId(p.name)}
          className="!w-2.5 !h-2.5 !border-0"
          style={
            collapsed
              ? collapsedHandleStyle("left")
              : {
                  backgroundColor: portColor(p.port_type),
                  top: rowPositions[`in_${p.name}`] ?? 0,
                  left: 0,
                }
          }
        />
      ))}

      {/* Output handles (right side). */}
      {outputs.map(p => (
        <Handle
          key={`out-${p.name}`}
          type="source"
          position={Position.Right}
          id={customNodeOutputHandleId(p.name)}
          className="!w-2.5 !h-2.5 !border-0"
          style={
            collapsed
              ? collapsedHandleStyle("right")
              : {
                  backgroundColor: portColor(p.port_type),
                  top: rowPositions[`out_${p.name}`] ?? 0,
                  right: 0,
                }
          }
        />
      ))}
    </NodeCard>
  );
}
