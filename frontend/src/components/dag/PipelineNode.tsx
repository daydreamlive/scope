import { Handle, Position, useReactFlow } from "@xyflow/react";
import type { NodeProps, Node } from "@xyflow/react";
import type { FlowNodeData } from "../../lib/dagUtils";
import { useDagPreview } from "./DagPreviewContext";

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

export function PipelineNode({ id, data }: NodeProps<PipelineNodeType>) {
  const { setNodes } = useReactFlow();
  const previewUrl = useDagPreview(id);

  const pipelineIds = data.availablePipelineIds || [];
  const portsMap = data.pipelinePortsMap;
  const streamInputs = data.streamInputs ?? ["video"];
  const streamOutputs = data.streamOutputs ?? ["video"];

  const handleChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const newPipelineId = e.target.value;
    const ports = newPipelineId && portsMap ? portsMap[newPipelineId] : null;
    setNodes(nds =>
      nds.map(n =>
        n.id === id
          ? {
              ...n,
              data: {
                ...n.data,
                pipelineId: newPipelineId || null,
                label: newPipelineId || n.id,
                streamInputs: ports?.inputs ?? ["video"],
                streamOutputs: ports?.outputs ?? ["video"],
              },
            }
          : n
      )
    );
  };

  // Calculate handle positions evenly distributed
  const inputCount = streamInputs.length;
  const outputCount = streamOutputs.length;

  return (
    <div className="rounded-lg border-2 border-blue-500 bg-blue-950/80 px-4 py-3 min-w-[220px]">
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

      {/* Preview thumbnail */}
      {previewUrl ? (
        <img
          src={previewUrl}
          alt="preview"
          className="mt-2 w-full max-w-[180px] rounded border border-blue-700/50"
        />
      ) : null}

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
