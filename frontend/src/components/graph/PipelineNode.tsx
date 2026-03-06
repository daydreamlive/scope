import { Handle, Position } from "@xyflow/react";
import type { NodeProps, Node } from "@xyflow/react";
import type { FlowNodeData } from "../../lib/graphUtils";

type PipelineNodeType = Node<FlowNodeData, "pipeline">;

/** Color palette for port handles - each port gets a consistent color */
const PORT_COLORS: Record<string, string> = {
  video: "bg-blue-400",
  video2: "bg-cyan-400",
  vace_input_frames: "bg-purple-400",
  vace_input_masks: "bg-pink-400",
};

function getPortColor(portName: string): string {
  return PORT_COLORS[portName] ?? "bg-gray-400";
}

export function PipelineNode({
  id,
  data,
  selected,
}: NodeProps<PipelineNodeType>) {
  const pipelineIds = data.availablePipelineIds || [];
  const streamInputs = data.streamInputs ?? ["video"];
  const streamOutputs = data.streamOutputs ?? ["video"];
  const onPipelineSelect = data.onPipelineSelect as
    | ((nodeId: string, pipelineId: string | null) => void)
    | undefined;

  const handleChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const newPipelineId = e.target.value || null;
    onPipelineSelect?.(id, newPipelineId);
  };

  // Calculate handle positions evenly distributed
  const inputCount = streamInputs.length;
  const outputCount = streamOutputs.length;

  return (
    <div
      className={`rounded-lg border-2 px-4 py-3 min-w-[220px] ${
        selected
          ? "border-blue-300 bg-blue-900/90 ring-2 ring-blue-400/50"
          : "border-blue-500 bg-blue-950/80"
      }`}
    >
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

      {/* Port labels */}
      <div className="flex justify-between mt-2 gap-4">
        <div className="flex flex-col gap-0.5">
          {streamInputs.map(port => (
            <div key={`in-${port}`} className="text-[10px] text-blue-300/70">
              {port}
            </div>
          ))}
        </div>
        <div className="flex flex-col gap-0.5 text-right">
          {streamOutputs.map(port => (
            <div key={`out-${port}`} className="text-[10px] text-blue-300/70">
              {port}
            </div>
          ))}
        </div>
      </div>

      {/* Input handles (left side) */}
      {streamInputs.map((port, i) => (
        <Handle
          key={`target-${port}`}
          type="target"
          position={Position.Left}
          id={port}
          className={`!${getPortColor(port)} !w-3 !h-3`}
          style={{ top: `${((i + 1) / (inputCount + 1)) * 100}%` }}
        />
      ))}

      {/* Output handles (right side) */}
      {streamOutputs.map((port, i) => (
        <Handle
          key={`source-${port}`}
          type="source"
          position={Position.Right}
          id={port}
          className={`!${getPortColor(port)} !w-3 !h-3`}
          style={{ top: `${((i + 1) / (outputCount + 1)) * 100}%` }}
        />
      ))}
    </div>
  );
}
