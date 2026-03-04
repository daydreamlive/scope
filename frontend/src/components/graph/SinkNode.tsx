import { Handle, Position } from "@xyflow/react";
import type { NodeProps, Node } from "@xyflow/react";
import type { FlowNodeData } from "../../lib/graphUtils";
import { NodeCard, NodeHeader, NodeBody, NodeParamRow, NodePill } from "./node-ui";

type SinkNodeType = Node<FlowNodeData, "sink">;

const ROW_CENTER_Y = 28 + 6 + 10;

export function SinkNode({ data }: NodeProps<SinkNodeType>) {
  return (
    <NodeCard>
      <NodeHeader title="Sink" dotColor="bg-orange-400" />
      <NodeBody>
        <NodeParamRow label="Label">
          <NodePill>{data.label}</NodePill>
        </NodeParamRow>
      </NodeBody>
      <Handle
        type="target"
        position={Position.Left}
        id="stream:video"
        className="!w-2 !h-2 !border-0"
        style={{ top: ROW_CENTER_Y, left: 8, backgroundColor: "#fb923c" }}
      />
    </NodeCard>
  );
}
