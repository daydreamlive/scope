import { Handle, Position } from "@xyflow/react";
import type { NodeProps, Node } from "@xyflow/react";
import { ArrowUp } from "lucide-react";
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
import { NODE_TOKENS } from "../ui/tokens";
import { useHandlePositions } from "../hooks/node/useHandlePositions";

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
  const { setRowRef, rowPositions } = useHandlePositions([
    collapsed,
    valueType,
    currentValue,
  ]);
  const autoSend = data.primitiveAutoSend !== false;

  const color = PARAM_TYPE_COLORS[valueType] || "#9ca3af";

  const handleValueChange = (newValue: unknown) => {
    if (autoSend) {
      updateData({ value: newValue, committedValue: newValue });
    } else {
      updateData({ value: newValue });
    }
  };

  const handleSend = () => {
    updateData({ committedValue: data.value });
  };

  const handleAutoSendToggle = (checked: boolean) => {
    if (checked) {
      updateData({ primitiveAutoSend: true, committedValue: data.value });
    } else {
      updateData({ primitiveAutoSend: false });
    }
  };

  const handleTypeChange = (newType: string | number) => {
    const vt = String(newType) as "string" | "number" | "boolean";
    const defaultVal = getDefaultForType(vt);
    updateData({
      valueType: vt,
      value: defaultVal,
      committedValue: defaultVal,
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
          <div ref={setRowRef("value")}>
            {valueType === "string" && (
              <div className="mt-1">
                <NodePillTextarea
                  value={String(currentValue)}
                  onChange={handleValueChange}
                  onSubmit={!autoSend ? handleSend : undefined}
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
                  onSubmit={!autoSend ? handleSend : undefined}
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
          </div>
          {valueType !== "boolean" && (
            <div className="flex items-center gap-1 mt-1">
              {!autoSend && (
                <button
                  type="button"
                  onClick={handleSend}
                  className={`${NODE_TOKENS.pill} flex-1 flex items-center justify-center gap-1 cursor-pointer hover:bg-[#2a2a2a] active:bg-[#333] transition-colors`}
                  title="Send value (Enter)"
                >
                  <ArrowUp className="h-3 w-3 text-[#fafafa]" />
                  <span className={NODE_TOKENS.primaryText}>Send</span>
                </button>
              )}
              <button
                type="button"
                onClick={() => handleAutoSendToggle(!autoSend)}
                className={`${NODE_TOKENS.pill} flex items-center gap-1 cursor-pointer hover:bg-[#2a2a2a] active:bg-[#333] transition-colors ${autoSend ? "flex-1 justify-center" : ""}`}
                title={
                  autoSend
                    ? "Auto-send is on (click to disable)"
                    : "Auto-send is off (click to enable)"
                }
              >
                <span
                  className={`inline-block w-1.5 h-1.5 rounded-full ${autoSend ? "bg-emerald-400" : "bg-[#555]"}`}
                />
                <span className={NODE_TOKENS.primaryText}>Auto-send</span>
              </button>
            </div>
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
            : { top: rowPositions["value"] ?? 44, left: 0, backgroundColor: color }
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
            : { top: rowPositions["value"] ?? 44, right: 0, backgroundColor: color }
        }
      />
    </NodeCard>
  );
}
