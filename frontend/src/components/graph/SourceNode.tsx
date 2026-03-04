import { Handle, Position } from "@xyflow/react";
import type { NodeProps, Node } from "@xyflow/react";
import type { FlowNodeData } from "../../lib/graphUtils";
import { NodeCard, NodeHeader, NodeBody, NodeParamRow, NodePill } from "./node-ui";

type SourceNodeType = Node<FlowNodeData, "source">;

const ROW_CENTER_Y = 28 + 6 + 10;

export function SourceNode({ data }: NodeProps<SourceNodeType>) {
  return (
    <NodeCard>
      <NodeHeader title="Source" dotColor="bg-green-400" />
      <NodeBody>
        <NodeParamRow label="Label">
          <NodePill>{data.label}</NodePill>
        </NodeParamRow>
      </NodeBody>
      <Handle
        type="source"
        position={Position.Right}
        id="stream:video"
        className="!w-2 !h-2 !border-0"
        style={{ top: ROW_CENTER_Y, right: 8, backgroundColor: "#ffffff" }}
      />
    </NodeCard>
  );
}
