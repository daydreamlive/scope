import { Handle, Position } from "@xyflow/react";
import type { NodeProps, Node } from "@xyflow/react";
import type { FlowNodeData } from "../../../lib/graphUtils";
import { buildHandleId } from "../../../lib/graphUtils";
import { useNodeData } from "../hooks/node/useNodeData";
import { useNodeCollapse } from "../hooks/node/useNodeCollapse";
import {
  NodeCard,
  NodeHeader,
  NodeBody,
  NodeParamRow,
  NodePillInput,
  NodePillTextarea,
  NodePillToggle,
  NodePillSelect,
  collapsedHandleStyle,
} from "../ui";
import { PARAM_TYPE_COLORS } from "../nodeColors";

type PrimitiveNodeType = Node<FlowNodeData, "primitive">;

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
  const { updateData } = useNodeData(id);
  const { collapsed, toggleCollapse } = useNodeCollapse();
  const valueType = data.valueType || "string";
  const currentValue = data.value ?? getDefaultForType(valueType);

  const color = PARAM_TYPE_COLORS[valueType] || "#9ca3af";

  const handleValueChange = (newValue: unknown) => {
    updateData({ value: newValue });
  };

  const handleTypeChange = (newType: string | number) => {
    const vt = String(newType) as "string" | "number" | "boolean";
    const defaultVal = getDefaultForType(vt);
    updateData({
      valueType: vt,
      value: defaultVal,
      parameterOutputs: [{ name: "value", type: vt, defaultValue: defaultVal }],
    });
  };

  return (
    <NodeCard selected={selected} collapsed={collapsed}>
      <NodeHeader
        title={data.customTitle || "Primitive"}
        onTitleChange={newTitle => updateData({ customTitle: newTitle })}
        collapsed={collapsed}
        onCollapseToggle={toggleCollapse}
      />
      {!collapsed && (
        <NodeBody>
          <NodeParamRow label="Type">
            <NodePillSelect
              value={valueType}
              onChange={handleTypeChange}
              options={TYPE_OPTIONS}
            />
          </NodeParamRow>
          {valueType === "string" && (
            <div className="mt-1">
              <NodePillTextarea
                value={String(currentValue)}
                onChange={handleValueChange}
                placeholder="Enter text…"
              />
            </div>
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
      )}
      <Handle
        type="target"
        position={Position.Left}
        id={buildHandleId("param", "value")}
        className="!w-2.5 !h-2.5 !border-0"
        style={
          collapsed
            ? collapsedHandleStyle("left")
            : { top: 44, left: 0, backgroundColor: color }
        }
      />
      <Handle
        type="source"
        position={Position.Right}
        id={buildHandleId("param", "value")}
        className="!w-2.5 !h-2.5 !border-0"
        style={
          collapsed
            ? collapsedHandleStyle("right")
            : { top: 44, right: 0, backgroundColor: color }
        }
      />
    </NodeCard>
  );
}
