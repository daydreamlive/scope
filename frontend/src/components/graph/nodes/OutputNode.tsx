import { Handle, Position } from "@xyflow/react";
import type { NodeProps, Node } from "@xyflow/react";
import type { FlowNodeData } from "../../../lib/graphUtils";
import { useNodeData } from "../hooks/useNodeData";
import { useNodeCollapse } from "../hooks/useNodeCollapse";
import {
  NodeCard,
  NodeHeader,
  NodeParamRow,
  NodePillSelect,
  NodePillInput,
  NodePillToggle,
  collapsedHandleStyle,
} from "../ui";

type OutputNodeType = Node<FlowNodeData, "output">;

const HEADER_H = 28;
const BODY_PAD = 6;
const ROW_H = 20;

const OUTPUT_TYPE_OPTIONS = [
  { value: "spout", label: "Spout" },
  { value: "ndi", label: "NDI" },
  { value: "syphon", label: "Syphon" },
];

const DEFAULT_NAMES: Record<string, string> = {
  spout: "ScopeOut",
  ndi: "Scope",
  syphon: "Scope",
};

export function OutputNode({ id, data, selected }: NodeProps<OutputNodeType>) {
  const { updateData } = useNodeData(id);
  const { collapsed, toggleCollapse } = useNodeCollapse();

  const sinkType = (data.outputSinkType as string) || "spout";
  const enabled = (data.outputSinkEnabled as boolean) ?? false;
  const senderName =
    (data.outputSinkName as string) || DEFAULT_NAMES[sinkType] || "Scope";

  // Availability flags passed from GraphEditor
  const spoutAvailable = data.spoutAvailable ?? false;
  const ndiAvailable = data.ndiAvailable ?? false;
  const syphonAvailable = data.syphonAvailable ?? false;

  // Filter output type options based on availability
  const filteredOptions = OUTPUT_TYPE_OPTIONS.filter(opt => {
    if (opt.value === "spout") return spoutAvailable;
    if (opt.value === "ndi") return ndiAvailable;
    if (opt.value === "syphon") return syphonAvailable;
    return true;
  });

  const handleTypeChange = (newType: string) => {
    updateData({
      outputSinkType: newType,
      outputSinkName: DEFAULT_NAMES[newType] || "Scope",
    });
  };

  const handleToggle = (checked: boolean) => {
    updateData({ outputSinkEnabled: checked });
  };

  const handleNameChange = (value: string | number) => {
    updateData({ outputSinkName: String(value) });
  };

  const handleY = HEADER_H + BODY_PAD + ROW_H / 2;

  const typeLabel =
    OUTPUT_TYPE_OPTIONS.find(o => o.value === sinkType)?.label ?? sinkType;

  return (
    <NodeCard selected={selected} collapsed={collapsed}>
      <NodeHeader
        title={data.customTitle || `${typeLabel} Output`}
        onTitleChange={newTitle => updateData({ customTitle: newTitle })}
        collapsed={collapsed}
        onCollapseToggle={toggleCollapse}
      />
      {!collapsed && (
        <div className="px-2 py-1.5 flex flex-col gap-1.5">
          <div className="px-2">
            <NodeParamRow label="Type">
              <NodePillSelect
                value={sinkType}
                onChange={handleTypeChange}
                options={
                  filteredOptions.length > 0
                    ? filteredOptions
                    : OUTPUT_TYPE_OPTIONS
                }
              />
            </NodeParamRow>
          </div>
          <div className="px-2">
            <NodeParamRow label="Enabled">
              <NodePillToggle checked={enabled} onChange={handleToggle} />
            </NodeParamRow>
          </div>
          <div className="px-2">
            <NodeParamRow label="Name">
              <NodePillInput
                type="text"
                value={senderName}
                onChange={handleNameChange}
                placeholder={DEFAULT_NAMES[sinkType] || "Scope"}
              />
            </NodeParamRow>
          </div>
        </div>
      )}
      <Handle
        type="target"
        position={Position.Left}
        id="stream:video"
        className="!w-2.5 !h-2.5 !border-0"
        style={
          collapsed
            ? collapsedHandleStyle("left")
            : { top: handleY, left: 0, backgroundColor: "#ffffff" }
        }
      />
    </NodeCard>
  );
}
