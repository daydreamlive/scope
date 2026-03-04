import { Handle, Position, useEdges } from "@xyflow/react";
import type { NodeProps, Node } from "@xyflow/react";
import type { FlowNodeData } from "../../lib/graphUtils";
import { buildHandleId } from "../../lib/graphUtils";
import {
  NodeCard,
  NodeHeader,
  NodeBody,
  NodeParamRow,
  NodePillSelect,
  NodePillSearchableSelect,
  NodePill,
  NodePillInput,
  NodePillToggle,
} from "./node-ui";

type PipelineNodeType = Node<FlowNodeData, "pipeline">;

const PORT_COLORS_HEX: Record<string, string> = {
  video: "#60a5fa",
  video2: "#22d3ee",
  vace_input_frames: "#a78bfa",
  vace_input_masks: "#f472b6",
};

function getPortColorHex(portName: string): string {
  return PORT_COLORS_HEX[portName] ?? "#9ca3af";
}

const PARAM_TYPE_COLORS: Record<string, string> = {
  string: "#fbbf24", // amber-400
  number: "#38bdf8", // sky-400
  boolean: "#34d399", // emerald-400
};

function getParamTypeColor(type: "string" | "number" | "boolean"): string {
  return PARAM_TYPE_COLORS[type] || "#9ca3af";
}

const HEADER_H = 28;
const BODY_PAD = 6;
const ROW_H = 20;
const ROW_GAP = 6;

function rowCenterY(n: number): number {
  return HEADER_H + BODY_PAD + n * (ROW_H + ROW_GAP) + ROW_H / 2;
}

export function PipelineNode({
  id,
  data,
  selected,
}: NodeProps<PipelineNodeType>) {
  const edges = useEdges();
  const pipelineIds = data.availablePipelineIds || [];
  const streamInputs = data.streamInputs ?? ["video"];
  const streamOutputs = data.streamOutputs ?? ["video"];
  const parameterInputs = data.parameterInputs || [];
  const onPipelineSelect = data.onPipelineSelect as
    | ((nodeId: string, pipelineId: string | null) => void)
    | undefined;
  const onParameterChange = data.onParameterChange as
    | ((nodeId: string, key: string, value: unknown) => void)
    | undefined;
  const parameterValues = (data.parameterValues as Record<string, unknown>) || {};

  const pipelineName = data.pipelineId || "Pipeline";

  const selectOptions = [
    { value: "", label: "Select pipeline..." },
    ...pipelineIds.map(pid => ({ value: pid, label: pid })),
  ];

  const isParamConnected = (paramName: string): boolean => {
    const handleId = buildHandleId("param", paramName);
    return edges.some(
      e => e.target === id && e.targetHandle === handleId
    );
  };

  const inputStartRow = 1;
  const outputStartRow = inputStartRow + streamInputs.length;
  const paramStartRow = outputStartRow + streamOutputs.length;

  return (
    <NodeCard selected={selected}>
      <NodeHeader title={pipelineName} dotColor="bg-blue-400" />
      <NodeBody withGap>
        <NodeParamRow label="Pipeline">
          <NodePillSearchableSelect
            value={data.pipelineId || ""}
            onChange={newValue => {
              const newPipelineId = newValue || null;
              onPipelineSelect?.(id, newPipelineId);
            }}
            options={selectOptions}
            placeholder="Select pipeline..."
          />
        </NodeParamRow>

        {streamInputs.map(port => (
          <NodeParamRow key={`in-${port}`} label="Input">
            <NodePill>{port}</NodePill>
          </NodeParamRow>
        ))}
        {streamOutputs.map(port => (
          <NodeParamRow key={`out-${port}`} label="Output">
            <NodePill>{port}</NodePill>
          </NodeParamRow>
        ))}

        {parameterInputs.map((param) => {
          const isConnected = isParamConnected(param.name);
          const currentValue = parameterValues[param.name] ?? param.defaultValue;

          return (
            <NodeParamRow key={`param-${param.name}`} label={param.label || param.name}>
              {isConnected ? (
                <NodePill className="opacity-50">Connected</NodePill>
              ) : param.type === "string" ? (
                param.enum ? (
                  <NodePillSelect
                    value={String(currentValue ?? "")}
                    onChange={val => onParameterChange?.(id, param.name, val)}
                    options={param.enum.map(opt => ({
                      value: String(opt),
                      label: String(opt),
                    }))}
                  />
                ) : (
                  <NodePillInput
                    type="text"
                    value={String(currentValue ?? "")}
                    onChange={val => onParameterChange?.(id, param.name, val)}
                  />
                )
              ) : param.type === "number" ? (
                <NodePillInput
                  type="number"
                  value={Number(currentValue ?? param.defaultValue ?? 0)}
                  onChange={val => onParameterChange?.(id, param.name, Number(val))}
                  min={param.min}
                  max={param.max}
                />
              ) : (
                <NodePillToggle
                  checked={Boolean(currentValue ?? param.defaultValue ?? false)}
                  onChange={val => onParameterChange?.(id, param.name, val)}
                />
              )}
            </NodeParamRow>
          );
        })}
      </NodeBody>

      {streamInputs.map((port, i) => (
        <Handle
          key={`target-${port}`}
          type="target"
          position={Position.Left}
          id={buildHandleId("stream", port)}
          className="!w-2 !h-2 !border-0"
          style={{
            top: rowCenterY(inputStartRow + i),
            left: 8,
            backgroundColor: getPortColorHex(port),
          }}
        />
      ))}

      {streamOutputs.map((port, i) => (
        <Handle
          key={`source-${port}`}
          type="source"
          position={Position.Right}
          id={buildHandleId("stream", port)}
          className="!w-2 !h-2 !border-0"
          style={{
            top: rowCenterY(outputStartRow + i),
            right: 8,
            backgroundColor: getPortColorHex(port),
          }}
        />
      ))}

      {parameterInputs.map((param, i) => (
        <Handle
          key={`param-target-${param.name}`}
          type="target"
          position={Position.Left}
          id={buildHandleId("param", param.name)}
          className="!w-2 !h-2 !border-0"
          style={{
            top: rowCenterY(paramStartRow + i),
            left: 8,
            backgroundColor: getParamTypeColor(param.type),
          }}
        />
      ))}
    </NodeCard>
  );
}
