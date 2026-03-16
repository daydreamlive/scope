import { Handle, Position } from "@xyflow/react";
import type { NodeProps, Node } from "@xyflow/react";
import { useCallback } from "react";
import type { FlowNodeData } from "../../../lib/graphUtils";
import { buildHandleId } from "../../../lib/graphUtils";
import { useNodeData } from "../hooks/node/useNodeData";
import { useNodeCollapse } from "../hooks/node/useNodeCollapse";
import { NodeCard, NodeHeader, NodeBody, collapsedHandleStyle } from "../ui";

type TriggerNodeType = Node<FlowNodeData, "trigger">;

const COLOR = "#f97316"; // orange-500

export function TriggerNode({
  id,
  data,
  selected,
}: NodeProps<TriggerNodeType>) {
  const { updateData } = useNodeData(id);
  const { collapsed, toggleCollapse } = useNodeCollapse();

  const active = Boolean(data.value);

  const press = useCallback(() => updateData({ value: true }), [updateData]);
  const release = useCallback(() => updateData({ value: false }), [updateData]);

  return (
    <NodeCard selected={selected} collapsed={collapsed}>
      <NodeHeader
        title={data.customTitle || "Trigger"}
        onTitleChange={newTitle => updateData({ customTitle: newTitle })}
        collapsed={collapsed}
        onCollapseToggle={toggleCollapse}
      />
      {!collapsed && (
        <NodeBody>
          <div className="flex items-center justify-center py-1">
            <button
              type="button"
              onPointerDown={e => {
                e.stopPropagation();
                press();
              }}
              onPointerUp={release}
              onPointerLeave={release}
              className="w-9 h-9 rounded-md border-2 transition-all duration-75 cursor-pointer focus:outline-none select-none"
              style={{
                borderColor: active ? COLOR : "#555",
                backgroundColor: active ? COLOR : "#222",
                boxShadow: active ? `0 0 12px ${COLOR}` : "none",
              }}
            />
          </div>
        </NodeBody>
      )}

      {/* Output handle (boolean pulse) */}
      <Handle
        type="source"
        position={Position.Right}
        id={buildHandleId("param", "value")}
        className="!w-2.5 !h-2.5 !border-0"
        style={
          collapsed
            ? collapsedHandleStyle("right")
            : {
                top: "50%",
                right: 0,
                backgroundColor: COLOR,
              }
        }
      />
    </NodeCard>
  );
}
