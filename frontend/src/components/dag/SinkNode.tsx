import { Handle, Position } from "@xyflow/react";
import type { NodeProps, Node } from "@xyflow/react";
import type { FlowNodeData } from "../../lib/dagUtils";
import { useDagPreview } from "./DagPreviewContext";

type SinkNodeType = Node<FlowNodeData, "sink">;

export function SinkNode({ id, data }: NodeProps<SinkNodeType>) {
  const previewUrl = useDagPreview(id);

  return (
    <div className="rounded-lg border-2 border-orange-500 bg-orange-950/80 px-4 py-3 min-w-[160px]">
      <div className="text-xs text-orange-400 font-medium mb-1">Sink</div>
      <div className="text-sm text-orange-100 font-semibold">{data.label}</div>
      {previewUrl ? (
        <img
          src={previewUrl}
          alt="preview"
          className="mt-2 w-full max-w-[180px] rounded border border-orange-700/50"
        />
      ) : null}
      <Handle
        type="target"
        position={Position.Left}
        id="video"
        className="!bg-orange-400 !w-3 !h-3"
      />
    </div>
  );
}
