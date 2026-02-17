import { Handle, Position, useReactFlow } from "@xyflow/react";
import type { NodeProps, Node } from "@xyflow/react";
import type { FlowNodeData } from "../../lib/dagUtils";

type PipelineNodeType = Node<FlowNodeData, "pipeline">;

export function PipelineNode({ id, data }: NodeProps<PipelineNodeType>) {
  const { setNodes } = useReactFlow();

  const pipelineIds = data.availablePipelineIds || [];

  const handleChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const newPipelineId = e.target.value;
    setNodes(nds =>
      nds.map(n =>
        n.id === id
          ? {
              ...n,
              data: {
                ...n.data,
                pipelineId: newPipelineId,
                label: newPipelineId || n.id,
              },
            }
          : n
      )
    );
  };

  return (
    <div className="rounded-lg border-2 border-blue-500 bg-blue-950/80 px-4 py-3 min-w-[200px]">
      <div className="text-xs text-blue-400 font-medium mb-1">Pipeline</div>
      <select
        value={data.pipelineId || ""}
        onChange={handleChange}
        className="w-full bg-blue-900/60 text-blue-100 text-sm rounded px-2 py-1 border border-blue-700 focus:outline-none focus:ring-1 focus:ring-blue-400"
      >
        <option value="">Select pipeline...</option>
        {pipelineIds.map(pid => (
          <option key={pid} value={pid}>
            {pid}
          </option>
        ))}
      </select>
      <Handle
        type="target"
        position={Position.Left}
        id="video"
        className="!bg-blue-400 !w-3 !h-3"
        style={{ top: "30%" }}
      />
      <Handle
        type="target"
        position={Position.Left}
        id="vace_input_frames"
        className="!bg-purple-400 !w-3 !h-3"
        style={{ top: "55%" }}
      />
      <Handle
        type="target"
        position={Position.Left}
        id="vace_input_masks"
        className="!bg-pink-400 !w-3 !h-3"
        style={{ top: "80%" }}
      />
      <Handle
        type="source"
        position={Position.Right}
        id="video"
        className="!bg-blue-400 !w-3 !h-3"
        style={{ top: "30%" }}
      />
      <Handle
        type="source"
        position={Position.Right}
        id="vace_input_frames"
        className="!bg-purple-400 !w-3 !h-3"
        style={{ top: "55%" }}
      />
      <Handle
        type="source"
        position={Position.Right}
        id="vace_input_masks"
        className="!bg-pink-400 !w-3 !h-3"
        style={{ top: "80%" }}
      />
    </div>
  );
}
