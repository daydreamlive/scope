import { Handle, Position, useReactFlow } from "@xyflow/react";
import type { NodeProps, Node } from "@xyflow/react";
import type { FlowNodeData } from "../../lib/graphUtils";
import { buildHandleId } from "../../lib/graphUtils";
import {
  NodeCard,
  NodeHeader,
  NodeBody,
  NodeParamRow,
  NodePillInput,
  NodePillToggle,
} from "./node-ui";

type ValueNodeType = Node<FlowNodeData, "value">;

const PARAM_TYPE_COLORS: Record<string, string> = {
  string: "#fbbf24", // amber-400
  number: "#38bdf8", // sky-400
  boolean: "#34d399", // emerald-400
};

function getParamTypeColor(type: "string" | "number" | "boolean"): string {
  return PARAM_TYPE_COLORS[type] || "#9ca3af";
}

export function ValueNode({
  id,
  data,
  selected,
}: NodeProps<ValueNodeType>) {
  const { setNodes } = useReactFlow();
  const valueType = data.valueType || "string";
  const currentValue = data.value ?? (valueType === "boolean" ? false : valueType === "number" ? 0 : "");

  const title = valueType.charAt(0).toUpperCase() + valueType.slice(1);
  const color = getParamTypeColor(valueType);
  const dotColorClass = valueType === "string" ? "bg-amber-400" : valueType === "number" ? "bg-sky-400" : "bg-emerald-400";

  const handleValueChange = (newValue: unknown) => {
    setNodes((nds) =>
      nds.map((n) => {
        if (n.id !== id) return n;
        return {
          ...n,
          data: {
            ...n.data,
            value: newValue,
          },
        };
      })
    );
  };

  return (
    <NodeCard selected={selected}>
      <NodeHeader title={title} dotColor={dotColorClass} />
      <NodeBody>
        {valueType === "string" && (
          <NodeParamRow label="Value">
            <NodePillInput
              type="text"
              value={String(currentValue)}
              onChange={handleValueChange}
            />
          </NodeParamRow>
        )}
        {valueType === "number" && (
          <NodeParamRow label="Value">
            <NodePillInput
              type="number"
              value={Number(currentValue)}
              onChange={handleValueChange}
            />
          </NodeParamRow>
        )}
        {valueType === "boolean" && (
          <NodeParamRow label="Value">
            <NodePillToggle
              checked={Boolean(currentValue)}
              onChange={handleValueChange}
            />
          </NodeParamRow>
        )}
      </NodeBody>
      <Handle
        type="source"
        position={Position.Right}
        id={buildHandleId("param", "value")}
        className="!w-2 !h-2 !border-0"
        style={{ top: 44, right: 8, backgroundColor: color }}
      />
    </NodeCard>
  );
}
