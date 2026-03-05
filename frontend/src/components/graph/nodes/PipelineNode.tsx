import { useRef, useCallback, useState, useLayoutEffect } from "react";
import { Handle, Position, useEdges } from "@xyflow/react";
import type { NodeProps, Node } from "@xyflow/react";
import type { FlowNodeData } from "../../../lib/graphUtils";
import { buildHandleId } from "../../../lib/graphUtils";
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
  NodePillTextarea,
  NODE_TOKENS,
} from "../ui";

type PipelineNodeType = Node<FlowNodeData, "pipeline">;

const PORT_COLORS_HEX: Record<string, string> = {
  video: "#ffffff",
  video2: "#ffffff",
  vace_input_frames: "#a78bfa",
  vace_input_masks: "#f472b6",
};

function getPortColorHex(portName: string): string {
  return PORT_COLORS_HEX[portName] ?? "#9ca3af";
}

const PARAM_TYPE_COLORS: Record<string, string> = {
  string: "#fbbf24",
  number: "#38bdf8",
  boolean: "#34d399",
};

function getParamTypeColor(
  type: "string" | "number" | "boolean" | "list_number"
): string {
  if (type === "list_number") return PARAM_TYPE_COLORS.number;
  return PARAM_TYPE_COLORS[type] || "#9ca3af";
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
  const parameterValues =
    (data.parameterValues as Record<string, unknown>) || {};
  const supportsPrompts = data.supportsPrompts ?? false;
  const promptText = data.promptText || "";
  const onPromptChange = data.onPromptChange as
    | ((nodeId: string, text: string) => void)
    | undefined;

  const pipelineName = data.pipelineId || "Pipeline";

  const selectOptions = [
    { value: "", label: "Select pipeline..." },
    ...pipelineIds.map(pid => ({ value: pid, label: pid })),
  ];

  const isParamConnected = (paramName: string): boolean => {
    const handleId = buildHandleId("param", paramName);
    return edges.some(e => e.target === id && e.targetHandle === handleId);
  };

  const isPromptConnected = edges.some(
    e =>
      e.target === id && e.targetHandle === buildHandleId("param", "__prompt")
  );

  const listParams = parameterInputs.filter(p => p.type === "list_number");
  const primitiveParams = parameterInputs.filter(p => p.type !== "list_number");

  // Measure DOM positions for handle placement
  const rowRefs = useRef<Map<string, HTMLDivElement>>(new Map());
  const [rowPositions, setRowPositions] = useState<Record<string, number>>({});

  const setRowRef = useCallback(
    (key: string) => (el: HTMLDivElement | null) => {
      if (el) rowRefs.current.set(key, el);
      else rowRefs.current.delete(key);
    },
    []
  );

  useLayoutEffect(() => {
    const positions: Record<string, number> = {};
    rowRefs.current.forEach((el, key) => {
      if (el) {
        positions[key] = el.offsetTop + el.offsetHeight / 2;
      }
    });
    setRowPositions(prev => {
      const keysChanged =
        Object.keys(positions).length !== Object.keys(prev).length ||
        Object.keys(positions).some(
          key => Math.abs((prev[key] ?? 0) - positions[key]) > 0.5
        );
      return keysChanged ? positions : prev;
    });
  }, [
    streamInputs,
    streamOutputs,
    parameterInputs,
    supportsPrompts,
    data.pipelineId,
  ]);

  return (
    <NodeCard selected={selected} autoMinHeight>
      <NodeHeader title={pipelineName} dotColor="bg-blue-400" />
      <NodeBody withGap>
        {/* Pipeline selector */}
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

        {/* Stream inputs */}
        {streamInputs.map(port => (
          <div key={`in-${port}`} ref={setRowRef(`in:${port}`)}>
            <NodeParamRow label="Input">
              <NodePill>{port}</NodePill>
            </NodeParamRow>
          </div>
        ))}

        {/* Stream outputs */}
        {streamOutputs.map(port => (
          <div key={`out-${port}`} ref={setRowRef(`out:${port}`)}>
            <NodeParamRow label="Output">
              <NodePill>{port}</NodePill>
            </NodeParamRow>
          </div>
        ))}

        {/* Primitive parameters (string, number, boolean) */}
        {primitiveParams.map(param => {
          const isConnected = isParamConnected(param.name);
          const currentValue =
            parameterValues[param.name] ?? param.defaultValue;

          return (
            <div
              key={`param-${param.name}`}
              ref={setRowRef(`param:${param.name}`)}
            >
              <NodeParamRow label={param.label || param.name}>
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
                    onChange={val =>
                      onParameterChange?.(id, param.name, Number(val))
                    }
                    min={param.min}
                    max={param.max}
                  />
                ) : (
                  <NodePillToggle
                    checked={Boolean(
                      currentValue ?? param.defaultValue ?? false
                    )}
                    onChange={val => onParameterChange?.(id, param.name, val)}
                  />
                )}
              </NodeParamRow>
            </div>
          );
        })}

        {/* List-number parameters (e.g. denoising_steps) rendered as individual sliders */}
        {listParams.map(param => {
          const isConnected = isParamConnected(param.name);
          const rawValue = parameterValues[param.name] ?? param.defaultValue;
          const values: number[] = Array.isArray(rawValue)
            ? rawValue
            : Array.isArray(param.defaultValue)
              ? param.defaultValue
              : [];

          return (
            <div
              key={`param-${param.name}`}
              ref={setRowRef(`param:${param.name}`)}
            >
              {isConnected ? (
                <NodeParamRow label={param.label || param.name}>
                  <NodePill className="opacity-50">Connected</NodePill>
                </NodeParamRow>
              ) : (
                <div className="flex flex-col gap-1">
                  <p className={`${NODE_TOKENS.labelText} text-[10px]`}>
                    {param.label || param.name}
                  </p>
                  {values.map((stepVal, idx) => (
                    <NodeParamRow key={idx} label={`Step ${idx + 1}`}>
                      <NodePillInput
                        type="number"
                        value={stepVal}
                        onChange={val => {
                          const updated = [...values];
                          updated[idx] = Number(val);
                          onParameterChange?.(id, param.name, updated);
                        }}
                        min={0}
                        max={1000}
                      />
                    </NodeParamRow>
                  ))}
                </div>
              )}
            </div>
          );
        })}

        {/* Prompt textarea (last, with its own handle) */}
        {supportsPrompts && (
          <div ref={setRowRef("prompt")}>
            <div className="flex flex-col gap-1">
              <p className={`${NODE_TOKENS.labelText} text-[10px] mb-0.5`}>
                Prompt
              </p>
              {isPromptConnected ? (
                <NodePill className="opacity-50">Connected</NodePill>
              ) : (
                <NodePillTextarea
                  value={promptText}
                  onChange={text => onPromptChange?.(id, text)}
                  placeholder="Enter prompt..."
                />
              )}
            </div>
          </div>
        )}
      </NodeBody>

      {/* Stream input handles */}
      {streamInputs.map(port => (
        <Handle
          key={`target-${port}`}
          type="target"
          position={Position.Left}
          id={buildHandleId("stream", port)}
          className="!w-2 !h-2 !border-0"
          style={{
            top: rowPositions[`in:${port}`] ?? 0,
            left: 8,
            backgroundColor: getPortColorHex(port),
          }}
        />
      ))}

      {/* Stream output handles */}
      {streamOutputs.map(port => (
        <Handle
          key={`source-${port}`}
          type="source"
          position={Position.Right}
          id={buildHandleId("stream", port)}
          className="!w-2 !h-2 !border-0"
          style={{
            top: rowPositions[`out:${port}`] ?? 0,
            right: 8,
            backgroundColor: getPortColorHex(port),
          }}
        />
      ))}

      {/* Primitive parameter input handles */}
      {primitiveParams.map(param => (
        <Handle
          key={`param-target-${param.name}`}
          type="target"
          position={Position.Left}
          id={buildHandleId("param", param.name)}
          className="!w-2 !h-2 !border-0"
          style={{
            top: rowPositions[`param:${param.name}`] ?? 0,
            left: 8,
            backgroundColor: getParamTypeColor(param.type),
          }}
        />
      ))}

      {/* List-number parameter input handles (e.g. denoising_steps) */}
      {listParams.map(param => (
        <Handle
          key={`param-target-${param.name}`}
          type="target"
          position={Position.Left}
          id={buildHandleId("param", param.name)}
          className="!w-2 !h-2 !border-0"
          style={{
            top: rowPositions[`param:${param.name}`] ?? 0,
            left: 8,
            backgroundColor: getParamTypeColor(param.type),
          }}
        />
      ))}

      {/* Prompt input handle */}
      {supportsPrompts && (
        <Handle
          type="target"
          position={Position.Left}
          id={buildHandleId("param", "__prompt")}
          className="!w-2 !h-2 !border-0"
          style={{
            top: rowPositions["prompt"] ?? 0,
            left: 8,
            backgroundColor: PARAM_TYPE_COLORS.string,
          }}
        />
      )}
    </NodeCard>
  );
}
