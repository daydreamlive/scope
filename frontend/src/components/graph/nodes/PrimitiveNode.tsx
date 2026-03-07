import { Handle, Position, useReactFlow } from "@xyflow/react";
import type { NodeProps, Node } from "@xyflow/react";
import type { FlowNodeData } from "../../../lib/graphUtils";
import { buildHandleId } from "../../../lib/graphUtils";
import {
  NodeCard,
  NodeHeader,
  NodeBody,
  NodeParamRow,
  NodePillInput,
  NodePillToggle,
  NodePillSelect,
} from "../ui";

type PrimitiveNodeType = Node<FlowNodeData, "primitive">;

const PARAM_TYPE_COLORS: Record<string, string> = {
  string: "#fbbf24", // amber-400
  number: "#38bdf8", // sky-400
  boolean: "#34d399", // emerald-400
};

const TYPE_DOT_CLASSES: Record<string, string> = {
  string: "bg-amber-400",
  number: "bg-sky-400",
  boolean: "bg-emerald-400",
};

const TYPE_OPTIONS = [
  { label: "String", value: "string" },
  { label: "Number", value: "number" },
  { label: "Boolean", value: "boolean" },
];

function getDefaultForType(type: string): unknown {
  if (type === "boolean") return false;
  if (type === "number") return 0;
  return "";
}

export function PrimitiveNode({
  id,
  data,
  selected,
}: NodeProps<PrimitiveNodeType>) {
  const { setNodes } = useReactFlow();
  const valueType = data.valueType || "string";
  const currentValue = data.value ?? getDefaultForType(valueType);

  const color = PARAM_TYPE_COLORS[valueType] || "#9ca3af";
  const dotColorClass = TYPE_DOT_CLASSES[valueType] || "bg-gray-400";

  const handleValueChange = (newValue: unknown) => {
    setNodes(nds =>
      nds.map(n => {
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

  const handleTypeChange = (newType: string | number) => {
    const vt = String(newType) as "string" | "number" | "boolean";
    const defaultVal = getDefaultForType(vt);
    setNodes(nds =>
      nds.map(n => {
        if (n.id !== id) return n;
        return {
          ...n,
          data: {
            ...n.data,
            valueType: vt,
            value: defaultVal,
            parameterOutputs: [
              { name: "value", type: vt, defaultValue: defaultVal },
            ],
          },
        };
      })
    );
  };

  return (
    <NodeCard selected={selected}>
      <NodeHeader
        title={data.customTitle || "Primitive"}
        dotColor={dotColorClass}
        onTitleChange={newTitle =>
          setNodes(nds =>
            nds.map(n =>
              n.id === id
                ? { ...n, data: { ...n.data, customTitle: newTitle } }
                : n
            )
          )
        }
      />
      <NodeBody>
        <NodeParamRow label="Type">
          <NodePillSelect
            value={valueType}
            onChange={handleTypeChange}
            options={TYPE_OPTIONS}
          />
        </NodeParamRow>
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
