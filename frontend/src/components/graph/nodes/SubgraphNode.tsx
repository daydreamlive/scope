import { Handle, Position, useEdges, useNodes } from "@xyflow/react";
import type { NodeProps, Node } from "@xyflow/react";
import { useMemo } from "react";
import { FolderOpen } from "lucide-react";
import type { FlowNodeData, SubgraphPort } from "../../../lib/graphUtils";
import { buildHandleId } from "../../../lib/graphUtils";
import { useNodeData } from "../hooks/node/useNodeData";
import { useNodeCollapse } from "../hooks/node/useNodeCollapse";
import { useHandlePositions } from "../hooks/node/useHandlePositions";
import { getAnyValueFromNode } from "../utils/getValueFromNode";
import {
  NodeCard,
  NodeHeader,
  NodeBody,
  NODE_TOKENS,
  collapsedHandleStyle,
} from "../ui";

type SubgraphNodeType = Node<FlowNodeData, "subgraph">;

const PORT_TYPE_COLORS: Record<string, string> = {
  stream: "#06b6d4", // cyan-500
  string: "#fbbf24",
  number: "#38bdf8",
  boolean: "#34d399",
  list_number: "#38bdf8",
};

function getPortColor(port: SubgraphPort): string {
  if (port.portType === "stream") return PORT_TYPE_COLORS.stream;
  return PORT_TYPE_COLORS[port.paramType || "stream"] || "#9ca3af";
}

function formatPortValue(val: unknown): string {
  if (val === null || val === undefined) return "—";
  if (typeof val === "number") {
    return Number.isInteger(val) ? String(val) : val.toFixed(3);
  }
  if (typeof val === "string") {
    return val.length > 12 ? val.slice(0, 12) + "…" : val;
  }
  if (typeof val === "boolean") return val ? "true" : "false";
  return "—";
}

export function SubgraphNode({
  id,
  data,
  selected,
}: NodeProps<SubgraphNodeType>) {
  const { updateData } = useNodeData(id);
  const { collapsed, toggleCollapse } = useNodeCollapse();

  const inputs: SubgraphPort[] = data.subgraphInputs ?? [];
  const outputs: SubgraphPort[] = data.subgraphOutputs ?? [];
  const innerCount = data.subgraphNodes?.length ?? 0;

  const portValues = (data.portValues ?? {}) as Record<string, unknown>;

  const onEnterSubgraph = data.onEnterSubgraph as
    | ((nodeId: string) => void)
    | undefined;

  const allEdges = useEdges();
  const allNodes = useNodes() as Node<FlowNodeData>[];

  const inputValues = useMemo(() => {
    const vals: Record<string, unknown> = {};
    for (const port of inputs) {
      if (port.portType !== "param") continue;
      const handleId = buildHandleId("param", port.name);
      const edge = allEdges.find(
        e => e.target === id && e.targetHandle === handleId
      );
      if (!edge) continue;
      const srcNode = allNodes.find(n => n.id === edge.source);
      if (!srcNode) continue;
      vals[port.name] = getAnyValueFromNode(srcNode, edge.sourceHandle);
    }
    return vals;
  }, [inputs, allEdges, allNodes, id]);

  const { setRowRef, rowPositions } = useHandlePositions([
    inputs.length,
    outputs.length,
  ]);

  const handleBodyDoubleClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    onEnterSubgraph?.(id);
  };

  return (
    <NodeCard
      selected={selected}
      autoMinHeight={!collapsed}
      collapsed={collapsed}
      className="!h-auto min-h-full"
    >
      {/* ── Input Handles ── */}
      {inputs.map((port, i) => {
        const handleId = buildHandleId(port.portType, port.name);
        const color = getPortColor(port);
        const isStream = port.portType === "stream";
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
        const handleId = buildHandleId(port.portType, port.name);
        const color = getPortColor(port);
        const isStream = port.portType === "stream";
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

      {/* ── Header (title is editable via double-click on the text) ── */}
      <NodeHeader
        title={data.customTitle || data.label || "Subgraph"}
        onTitleChange={newTitle => updateData({ customTitle: newTitle })}
        collapsed={collapsed}
        onCollapseToggle={toggleCollapse}
      />

      {!collapsed && (
        <div className="pb-[28px]" onDoubleClick={handleBodyDoubleClick}>
          <NodeBody withGap>
            {/* Info */}
            <div className={`${NODE_TOKENS.pill} text-[10px] text-[#888]`}>
              {innerCount} node{innerCount !== 1 ? "s" : ""} inside
            </div>

            {/* Inputs */}
            {inputs.map(port => {
              const val = inputValues[port.name];
              const isParam = port.portType === "param";
              return (
                <div
                  key={`in-${port.name}`}
                  ref={setRowRef(`in:${port.name}`)}
                  className="flex items-center gap-1.5 px-2 py-0.5"
                >
                  <span
                    className="inline-block w-2 h-2 rounded-full shrink-0"
                    style={{ backgroundColor: getPortColor(port) }}
                  />
                  <span className="text-[10px] text-[#aaa] truncate flex-1">
                    {port.name}
                  </span>
                  {isParam && (
                    <span className="text-[10px] text-[#666] font-mono shrink-0">
                      {formatPortValue(val)}
                    </span>
                  )}
                </div>
              );
            })}

            {/* Outputs */}
            {outputs.map(port => {
              const isParam = port.portType === "param";
              const val = portValues[port.name];
              return (
                <div
                  key={`out-${port.name}`}
                  ref={setRowRef(`out:${port.name}`)}
                  className="flex items-center justify-end gap-1.5 px-2 py-0.5"
                >
                  {isParam && (
                    <span className="text-[10px] text-[#666] font-mono shrink-0">
                      {formatPortValue(val)}
                    </span>
                  )}
                  <span className="text-[10px] text-[#aaa] truncate">
                    {port.name}
                  </span>
                  <span
                    className="inline-block w-2 h-2 rounded-full shrink-0"
                    style={{ backgroundColor: getPortColor(port) }}
                  />
                </div>
              );
            })}
          </NodeBody>
        </div>
      )}

      {/* ── Footer: absolutely pinned to bottom of card ── */}
      {!collapsed && (
        <div
          className="absolute bottom-0 left-0 right-0 bg-[#181717] border-t border-[rgba(119,119,119,0.15)] flex items-center justify-center gap-1.5 px-2 py-1 h-[28px] rounded-b-xl cursor-pointer hover:bg-[#222] transition-colors"
          onClick={e => {
            e.stopPropagation();
            onEnterSubgraph?.(id);
          }}
        >
          <FolderOpen className="h-3 w-3 text-[#888]" />
          <span className="text-[10px] text-[#888] font-normal select-none">
            Edit Subgraph
          </span>
        </div>
      )}
    </NodeCard>
  );
}
