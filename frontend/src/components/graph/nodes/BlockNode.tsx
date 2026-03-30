import { Handle, Position } from "@xyflow/react";
import type { NodeProps, Node } from "@xyflow/react";
import type { FlowNodeData } from "../../../lib/graphUtils";
import { buildHandleId } from "../../../lib/graphUtils";
import type { TypedPort } from "../../../lib/api";
import { useNodeCollapse } from "../hooks/node/useNodeCollapse";
import { useHandlePositions } from "../hooks/node/useHandlePositions";
import { NodeCard, NodeHeader, NodeBody, collapsedHandleStyle } from "../ui";
import { TYPE_HINT_COLORS, COLOR_DEFAULT } from "../nodeColors";

type BlockNodeType = Node<FlowNodeData, "block">;

function getTypeColor(typeHint: string): string {
  // Check direct match first
  if (typeHint in TYPE_HINT_COLORS) return TYPE_HINT_COLORS[typeHint];
  // Check if any key is contained in the type hint (e.g. "list[Tensor]" matches "Tensor")
  for (const [key, color] of Object.entries(TYPE_HINT_COLORS)) {
    if (typeHint.includes(key)) return color;
  }
  return COLOR_DEFAULT;
}

function portHandleKind(port: TypedPort): "stream" | "param" {
  // Tensor-typed and video-like ports are treated as stream handles
  const hint = port.type_hint.toLowerCase();
  if (
    hint.includes("tensor") ||
    port.name === "video" ||
    port.name === "output_video"
  ) {
    return "stream";
  }
  return "param";
}

export function BlockNode({ data, selected }: NodeProps<BlockNodeType>) {
  const { collapsed, toggleCollapse } = useNodeCollapse();

  const blockSchema = data.blockSchema;
  const inputs: TypedPort[] = blockSchema?.inputs ?? [];
  const outputs: TypedPort[] = blockSchema?.outputs ?? [];

  const { setRowRef, rowPositions } = useHandlePositions([
    inputs.length,
    outputs.length,
  ]);

  return (
    <NodeCard
      selected={selected}
      autoMinHeight={!collapsed}
      collapsed={collapsed}
      className="!h-auto min-h-full !bg-[#1a2332] !border-[rgba(74,222,128,0.25)]"
    >
      {/* ── Input Handles ── */}
      {inputs.map((port, i) => {
        const kind = portHandleKind(port);
        const handleId = buildHandleId(kind, port.name);
        const color = getTypeColor(port.type_hint);
        const isStream = kind === "stream";
        return (
          <Handle
            key={`in-${port.name}`}
            type="target"
            position={Position.Left}
            id={handleId}
            style={
              collapsed
                ? collapsedHandleStyle("left")
                : {
                    top: rowPositions[`in:${port.name}`] ?? 30 + i * 26,
                    backgroundColor: color,
                    width: isStream ? 10 : 8,
                    height: isStream ? 10 : 8,
                  }
            }
          />
        );
      })}

      {/* ── Output Handles ── */}
      {outputs.map((port, i) => {
        const kind = portHandleKind(port);
        const handleId = buildHandleId(kind, port.name);
        const color = getTypeColor(port.type_hint);
        const isStream = kind === "stream";
        return (
          <Handle
            key={`out-${port.name}`}
            type="source"
            position={Position.Right}
            id={handleId}
            style={
              collapsed
                ? collapsedHandleStyle("right")
                : {
                    top: rowPositions[`out:${port.name}`] ?? 30 + i * 26,
                    backgroundColor: color,
                    width: isStream ? 10 : 8,
                    height: isStream ? 10 : 8,
                  }
            }
          />
        );
      })}

      {/* ── Header ── */}
      <NodeHeader
        title={data.customTitle || data.label || "Block"}
        collapsed={collapsed}
        onCollapseToggle={toggleCollapse}
      />

      {!collapsed && (
        <NodeBody withGap>
          {/* Description */}
          {blockSchema?.description && (
            <div className="text-[9px] text-[#666] px-2 pb-1 leading-tight">
              {blockSchema.description.length > 80
                ? blockSchema.description.slice(0, 80) + "…"
                : blockSchema.description}
            </div>
          )}

          {/* Inputs */}
          {inputs.map(port => (
            <div
              key={`in-${port.name}`}
              ref={setRowRef(`in:${port.name}`)}
              className="flex items-center gap-1.5 px-2 py-0.5"
            >
              <span
                className="inline-block w-2 h-2 rounded-full shrink-0"
                style={{ backgroundColor: getTypeColor(port.type_hint) }}
              />
              <span className="text-[10px] text-[#aaa] truncate flex-1">
                {port.name}
              </span>
              <span className="text-[9px] text-[#555] font-mono shrink-0">
                {port.type_hint}
              </span>
            </div>
          ))}

          {/* Separator between inputs and outputs */}
          {inputs.length > 0 && outputs.length > 0 && (
            <div className="border-t border-[rgba(255,255,255,0.05)] mx-2" />
          )}

          {/* Outputs */}
          {outputs.map(port => (
            <div
              key={`out-${port.name}`}
              ref={setRowRef(`out:${port.name}`)}
              className="flex items-center justify-end gap-1.5 px-2 py-0.5"
            >
              <span className="text-[9px] text-[#555] font-mono shrink-0">
                {port.type_hint}
              </span>
              <span className="text-[10px] text-[#aaa] truncate flex-1 text-right">
                {port.name}
              </span>
              <span
                className="inline-block w-2 h-2 rounded-full shrink-0"
                style={{ backgroundColor: getTypeColor(port.type_hint) }}
              />
            </div>
          ))}

          {/* Components badge */}
          {blockSchema?.components && blockSchema.components.length > 0 && (
            <div className="text-[9px] text-[#555] px-2 pt-1">
              uses: {blockSchema.components.join(", ")}
            </div>
          )}
        </NodeBody>
      )}
    </NodeCard>
  );
}
