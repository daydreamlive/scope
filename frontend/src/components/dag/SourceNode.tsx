import { Handle, Position } from "@xyflow/react";
import type { NodeProps, Node } from "@xyflow/react";
import type { FlowNodeData } from "../../lib/dagUtils";
import { useDagPreview } from "./DagPreviewContext";

type SourceNodeType = Node<FlowNodeData, "source">;

export function SourceNode({ data }: NodeProps<SourceNodeType>) {
  const previewUrl = useDagPreview("input");

  return (
    <div className="rounded-lg border-2 border-green-500 bg-green-950/80 px-4 py-3 min-w-[160px]">
      <div className="text-xs text-green-400 font-medium mb-1">Source</div>
      <div className="text-sm text-green-100 font-semibold">{data.label}</div>
      {previewUrl ? (
        <img
          src={previewUrl}
          alt="preview"
          className="mt-2 w-full max-w-[180px] rounded border border-green-700/50"
        />
      ) : null}
      <Handle
        type="source"
        position={Position.Right}
        id="video"
        className="!bg-green-400 !w-3 !h-3"
      />
    </div>
  );
}
