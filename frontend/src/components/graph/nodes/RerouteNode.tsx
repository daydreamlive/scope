import { Handle, Position } from "@xyflow/react";
import type { NodeProps, Node } from "@xyflow/react";
import type { FlowNodeData } from "../../../lib/graphUtils";
import { buildHandleId } from "../../../lib/graphUtils";

type RerouteNodeType = Node<FlowNodeData, "reroute">;

const TYPE_COLORS: Record<string, string> = {
  string: "#fbbf24",
  number: "#38bdf8",
  boolean: "#34d399",
};

const TYPE_BG: Record<string, string> = {
  string: "rgba(251,191,36,0.12)",
  number: "rgba(56,189,248,0.12)",
  boolean: "rgba(52,211,153,0.12)",
};

export function RerouteNode({ data, selected }: NodeProps<RerouteNodeType>) {
  const vt = data.valueType;
  const accent = (vt && TYPE_COLORS[vt]) || "#6b7280";
  const bg = (vt && TYPE_BG[vt]) || "rgba(107,114,128,0.10)";

  return (
    <div
      className={`flex items-center gap-0 ${selected ? "ring-2 ring-blue-400/50 rounded-full" : ""}`}
      style={{
        height: 18,
        minWidth: 36,
        borderRadius: 9,
        background: bg,
        border: `1.5px solid ${accent}`,
        cursor: "grab",
        position: "relative",
      }}
    >
      {/* Input connector */}
      <Handle
        type="target"
        position={Position.Left}
        id={buildHandleId("param", "value")}
        className="!border-0"
        style={{
          width: 8,
          height: 8,
          borderRadius: "50%",
          backgroundColor: accent,
          left: -4,
          top: "50%",
          transform: "translateY(-50%)",
        }}
      />
      {/* Center line */}
      <div
        style={{
          flex: 1,
          height: 2,
          backgroundColor: accent,
          opacity: 0.35,
          margin: "0 6px",
          borderRadius: 1,
        }}
      />
      {/* Output connector */}
      <Handle
        type="source"
        position={Position.Right}
        id={buildHandleId("param", "value")}
        className="!border-0"
        style={{
          width: 8,
          height: 8,
          borderRadius: "50%",
          backgroundColor: accent,
          right: -4,
          top: "50%",
          transform: "translateY(-50%)",
        }}
      />
    </div>
  );
}
