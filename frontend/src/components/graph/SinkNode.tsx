import { Handle, Position } from "@xyflow/react";
import type { NodeProps, Node } from "@xyflow/react";
import type { FlowNodeData } from "../../lib/graphUtils";

type SinkNodeType = Node<FlowNodeData, "sink">;

export function SinkNode({ data }: NodeProps<SinkNodeType>) {
  return (
    <div className="rounded-lg border-2 border-orange-500 bg-orange-950/80 px-4 py-3 min-w-[160px]">
      <div className="text-xs text-orange-400 font-medium mb-1">Sink</div>
      <div className="text-sm text-orange-100 font-semibold">{data.label}</div>
      <Handle
        type="target"
        position={Position.Left}
        id="video"
        className="!bg-orange-400 !w-3 !h-3"
      />
    </div>
  );
}
