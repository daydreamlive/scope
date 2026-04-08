import { Handle, Position, useEdges } from "@xyflow/react";
import type { NodeProps, Node } from "@xyflow/react";
import { RotateCcw, AlertTriangle, ArrowUp, Info } from "lucide-react";
import type { FlowNodeData } from "../../../lib/graphUtils";
import { buildHandleId } from "../../../lib/graphUtils";
import type { PipelineId } from "../../../types";
import {
  getResolutionScaleFactor,
  adjustResolutionForPipeline,
} from "../../../lib/utils";
import { useNodeData } from "../hooks/node/useNodeData";
import { useNodeCollapse } from "../hooks/node/useNodeCollapse";
import { useHandlePositions } from "../hooks/node/useHandlePositions";
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
  collapsedHandleStyle,
} from "../ui";
import { PARAM_TYPE_COLORS, HANDLE_COLORS, COLOR_DEFAULT } from "../nodeColors";

type PipelineNodeType = Node<FlowNodeData, "pipeline">;

function getPortColorHex(portName: string): string {
  return HANDLE_COLORS[portName] ?? COLOR_DEFAULT;
}

function getParamTypeColor(
  type: "string" | "number" | "boolean" | "list_number" | "trigger"
): string {
  return PARAM_TYPE_COLORS[type] || COLOR_DEFAULT;
}

export function PipelineNode({
  id,
  data,
  selected,
}: NodeProps<PipelineNodeType>) {
  const { updateData } = useNodeData(id);
  const { collapsed, toggleCollapse } = useNodeCollapse();
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
  const onPromptSubmit = data.onPromptSubmit as
    | ((nodeId: string) => void)
    | undefined;
  const supportsCacheManagement = data.supportsCacheManagement ?? false;
  const pipelineAvailable = data.pipelineAvailable ?? true;
  const supportsVace = data.supportsVace ?? false;
  const supportsLoRA = data.supportsLoRA ?? false;
  const isStreaming = data.isStreaming ?? false;

  const pipelineName = data.pipelineId || "Model";

  // Inject unavailable pipelineId into options
  const isUnavailable =
    !!data.pipelineId && !pipelineIds.includes(data.pipelineId);

  const selectOptions = [
    { value: "", label: "Select model..." },
    ...pipelineIds.map(pid => ({ value: pid, label: pid })),
    ...(isUnavailable
      ? [
          {
            value: data.pipelineId!,
            label: `${data.pipelineId} (not installed)`,
          },
        ]
      : []),
  ];

  const isParamConnected = (paramName: string): boolean => {
    const handleId = buildHandleId("param", paramName);
    return edges.some(e => e.target === id && e.targetHandle === handleId);
  };

  const isPromptConnected = edges.some(
    e =>
      e.target === id && e.targetHandle === buildHandleId("param", "__prompt")
  );

  const isVaceConnected = edges.some(
    e => e.target === id && e.targetHandle === buildHandleId("param", "__vace")
  );

  const isLoraConnected = edges.some(
    e => e.target === id && e.targetHandle === buildHandleId("param", "__loras")
  );

  // Resolution validation warning
  const pipelineIdStr = data.pipelineId as PipelineId | undefined;
  const scaleFactor = pipelineIdStr
    ? getResolutionScaleFactor(pipelineIdStr)
    : null;
  const resolutionWarning = (() => {
    if (!scaleFactor || !pipelineIdStr) return null;
    const heightParam = parameterInputs.find(p => p.name === "height");
    const widthParam = parameterInputs.find(p => p.name === "width");
    if (!heightParam && !widthParam) return null;
    const h = Number(parameterValues.height ?? heightParam?.defaultValue ?? 0);
    const w = Number(parameterValues.width ?? widthParam?.defaultValue ?? 0);
    if (h === 0 || w === 0) return null;
    if (h % scaleFactor !== 0 || w % scaleFactor !== 0) {
      const { resolution: adj } = adjustResolutionForPipeline(pipelineIdStr, {
        height: h,
        width: w,
      });
      return `Resolution will be adjusted to ${adj.width}\u00d7${adj.height} when starting the stream (must be divisible by ${scaleFactor})`;
    }
    return null;
  })();

  const listParams = parameterInputs.filter(p => p.type === "list_number");
  const primitiveParams = parameterInputs.filter(
    p => p.type !== "list_number" && p.name !== "reset_cache"
  );
  const isResetCacheConnected = isParamConnected("reset_cache");

  const VACE_STREAM_PORTS = ["vace_input_frames", "vace_input_masks"];
  const normalStreamInputs = streamInputs.filter(
    p => !VACE_STREAM_PORTS.includes(p)
  );
  // When supportsVace is true, always show these ports (they are always in the
  // schema for VACE pipelines). Fall back to what streamInputs provides if
  // supportsVace hasn't been resolved yet.
  const vaceStreamInputs = supportsVace
    ? VACE_STREAM_PORTS
    : streamInputs.filter(p => VACE_STREAM_PORTS.includes(p));

  // Measure handle positions when content changes
  const { setRowRef, rowPositions } = useHandlePositions([
    normalStreamInputs,
    vaceStreamInputs,
    streamOutputs,
    primitiveParams.length,
    listParams.length,
    supportsPrompts,
    supportsVace,
    supportsLoRA,
  ]);

  return (
    <NodeCard
      selected={selected}
      autoMinHeight={!collapsed}
      collapsed={collapsed}
    >
      <NodeHeader
        title={data.customTitle || pipelineName}
        onTitleChange={newTitle => updateData({ customTitle: newTitle })}
        collapsed={collapsed}
        onCollapseToggle={toggleCollapse}
      />
      {!collapsed && (
        <NodeBody withGap>
          {/* Pipeline selector */}
          <NodeParamRow label="Model">
            <NodePillSearchableSelect
              value={data.pipelineId || ""}
              onChange={newValue => {
                const newPipelineId = newValue || null;
                onPipelineSelect?.(id, newPipelineId);
              }}
              options={selectOptions}
              placeholder="Select model..."
              disabled={isStreaming}
            />
          </NodeParamRow>

          {/* Warning banner for unavailable pipeline */}
          {!pipelineAvailable && data.pipelineId && (
            <div className="flex items-center gap-1.5 px-2 py-1 rounded-md bg-amber-500/10 border border-amber-500/20">
              <AlertTriangle className="h-3 w-3 text-amber-400 shrink-0" />
              <span className="text-[10px] text-amber-400">
                Model not installed
              </span>
            </div>
          )}

          {/* Stream inputs (normal, e.g. video) */}
          {normalStreamInputs.map(port => (
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

          {/* Reset Cache button for pipelines that support cache management */}
          {supportsCacheManagement && (
            <div ref={setRowRef("param:reset_cache")}>
              <NodeParamRow label="Reset Cache">
                {isResetCacheConnected ? (
                  <NodePill className="opacity-50">connected</NodePill>
                ) : (
                  <button
                    type="button"
                    onClick={() => onParameterChange?.(id, "reset_cache", true)}
                    className={`${NODE_TOKENS.pill} flex items-center justify-center gap-1 w-[110px] cursor-pointer hover:bg-[#2a2a2a] active:bg-[#333] transition-colors`}
                    title="Clear longlive cache to regenerate fresh frames"
                  >
                    <RotateCcw className="h-3 w-3 text-[#fafafa]" />
                    <span className={NODE_TOKENS.primaryText}>Reset</span>
                  </button>
                )}
              </NodeParamRow>
            </div>
          )}

          {/* Primitive parameters (string, number, boolean) */}
          {primitiveParams.map(param => {
            const isConnected = isParamConnected(param.name);
            const currentValue =
              parameterValues[param.name] ?? param.defaultValue;
            const isLoadParam = param.isLoadParam === true;
            const disabled = isStreaming && isLoadParam;

            return (
              <div
                key={`param-${param.name}`}
                ref={setRowRef(`param:${param.name}`)}
              >
                <NodeParamRow label={param.label || param.name}>
                  {isConnected ? (
                    <NodePill className="opacity-50">
                      {param.type === "boolean"
                        ? currentValue !== undefined
                          ? String(Boolean(currentValue))
                          : "—"
                        : currentValue !== undefined
                          ? param.type === "number"
                            ? typeof currentValue === "number"
                              ? Number.isInteger(currentValue)
                                ? currentValue
                                : currentValue.toFixed(3)
                              : String(currentValue)
                            : String(currentValue)
                          : "—"}
                    </NodePill>
                  ) : param.type === "string" ? (
                    param.enum ? (
                      <NodePillSelect
                        value={
                          currentValue === null || currentValue === undefined
                            ? "__none__"
                            : String(currentValue)
                        }
                        onChange={val =>
                          onParameterChange?.(
                            id,
                            param.name,
                            val === "__none__" ? null : val
                          )
                        }
                        options={param.enum.map(opt => ({
                          value: opt === null ? "__none__" : String(opt),
                          label: opt === null ? "None" : String(opt),
                        }))}
                        disabled={disabled}
                      />
                    ) : (
                      <NodePillInput
                        type="text"
                        value={String(currentValue ?? "")}
                        onChange={val =>
                          onParameterChange?.(id, param.name, val)
                        }
                        disabled={disabled}
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
                      step={param.step}
                      disabled={disabled}
                    />
                  ) : (
                    <NodePillToggle
                      checked={Boolean(
                        currentValue ?? param.defaultValue ?? false
                      )}
                      onChange={val => onParameterChange?.(id, param.name, val)}
                      disabled={disabled}
                    />
                  )}
                </NodeParamRow>
              </div>
            );
          })}

          {/* List-number parameters as individual sliders.
             These are always editable during streaming (e.g. denoising_steps
             is runtime-modifiable despite is_load_param in the schema). */}
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
                  <div className="flex flex-col gap-1">
                    <p className={`${NODE_TOKENS.labelText} text-[10px]`}>
                      {param.label || param.name}
                    </p>
                    <NodePill className="opacity-50">
                      {values.length > 0
                        ? `[${values.map(v => (Number.isInteger(v) ? v : v.toFixed(2))).join(", ")}]`
                        : "—"}
                    </NodePill>
                  </div>
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

          {/* Resolution adjustment warning */}
          {resolutionWarning && (
            <div className="flex items-start gap-1.5 px-1">
              <Info className="h-3 w-3 mt-0.5 shrink-0 text-amber-500" />
              <span className="text-[10px] text-amber-500">
                {resolutionWarning}
              </span>
            </div>
          )}

          {/* LoRA input */}
          {supportsLoRA && (
            <div ref={setRowRef("lora")}>
              <NodeParamRow label="LoRA">
                <NodePill className={isLoraConnected ? "" : "opacity-40"}>
                  {isLoraConnected ? "Connected" : "Not connected"}
                </NodePill>
              </NodeParamRow>
            </div>
          )}

          {/* VACE input */}
          {supportsVace && (
            <div ref={setRowRef("vace")}>
              <NodeParamRow label="VACE">
                <NodePill className={isVaceConnected ? "" : "opacity-40"}>
                  {isVaceConnected ? "Connected" : "Not connected"}
                </NodePill>
              </NodeParamRow>
            </div>
          )}

          {/* VACE stream inputs (vace_input_frames, vace_input_masks) */}
          {vaceStreamInputs.map(port => (
            <div key={`in-${port}`} ref={setRowRef(`in:${port}`)}>
              <NodeParamRow
                label={
                  port === "vace_input_frames" ? "VACE Frames" : "VACE Masks"
                }
              >
                <NodePill>{port}</NodePill>
              </NodeParamRow>
            </div>
          ))}

          {/* Prompt textarea */}
          {supportsPrompts && (
            <div ref={setRowRef("prompt")}>
              <div className="flex flex-col gap-1">
                <p className={`${NODE_TOKENS.labelText} text-[10px] mb-0.5`}>
                  Prompt
                </p>
                {isPromptConnected ? (
                  <NodePillTextarea
                    value={promptText || "—"}
                    onChange={() => {}}
                    disabled
                    className="opacity-50"
                  />
                ) : (
                  <div className="flex flex-col gap-1">
                    <NodePillTextarea
                      value={promptText}
                      onChange={text => onPromptChange?.(id, text)}
                      onSubmit={() => onPromptSubmit?.(id)}
                      placeholder="Enter prompt..."
                    />
                    <button
                      type="button"
                      onClick={() => onPromptSubmit?.(id)}
                      className={`${NODE_TOKENS.pill} flex items-center justify-center gap-1 w-full cursor-pointer hover:bg-[#2a2a2a] active:bg-[#333] transition-colors`}
                      title="Send prompt (Enter)"
                    >
                      <ArrowUp className="h-3 w-3 text-[#fafafa]" />
                      <span className={NODE_TOKENS.primaryText}>Send</span>
                    </button>
                  </div>
                )}
              </div>
            </div>
          )}
        </NodeBody>
      )}

      {/* Stream input handles (normal) */}
      {normalStreamInputs.map((port, idx) => (
        <Handle
          key={`target-${port}`}
          type="target"
          position={Position.Left}
          id={buildHandleId("stream", port)}
          className={
            collapsed && idx > 0
              ? "!w-0 !h-0 !border-0 !min-w-0 !min-h-0"
              : "!w-2.5 !h-2.5 !border-0"
          }
          style={
            collapsed
              ? idx === 0
                ? collapsedHandleStyle("left")
                : { ...collapsedHandleStyle("left"), opacity: 0 }
              : {
                  top: rowPositions[`in:${port}`] ?? 0,
                  left: 0,
                  backgroundColor: getPortColorHex(port),
                }
          }
        />
      ))}

      {/* VACE stream input handles (vace_input_frames, vace_input_masks) */}
      {vaceStreamInputs.map(port => (
        <Handle
          key={`target-${port}`}
          type="target"
          position={Position.Left}
          id={buildHandleId("stream", port)}
          className={
            collapsed
              ? "!w-0 !h-0 !border-0 !min-w-0 !min-h-0"
              : "!w-2.5 !h-2.5 !border-0"
          }
          style={
            collapsed
              ? { ...collapsedHandleStyle("left"), opacity: 0 }
              : {
                  top: rowPositions[`in:${port}`] ?? 0,
                  left: 0,
                  backgroundColor: getPortColorHex(port),
                }
          }
        />
      ))}

      {/* Primitive parameter input handles */}
      {primitiveParams.map(param => (
        <Handle
          key={`param-target-${param.name}`}
          type="target"
          position={Position.Left}
          id={buildHandleId("param", param.name)}
          className={
            collapsed
              ? "!w-0 !h-0 !border-0 !min-w-0 !min-h-0"
              : "!w-2.5 !h-2.5 !border-0"
          }
          style={
            collapsed
              ? { ...collapsedHandleStyle("left"), opacity: 0 }
              : {
                  top: rowPositions[`param:${param.name}`] ?? 0,
                  left: 0,
                  backgroundColor: getParamTypeColor(param.type),
                }
          }
        />
      ))}

      {/* List-number parameter input handles (e.g. denoising_steps) */}
      {listParams.map(param => (
        <Handle
          key={`param-target-${param.name}`}
          type="target"
          position={Position.Left}
          id={buildHandleId("param", param.name)}
          className={
            collapsed
              ? "!w-0 !h-0 !border-0 !min-w-0 !min-h-0"
              : "!w-2.5 !h-2.5 !border-0"
          }
          style={
            collapsed
              ? { ...collapsedHandleStyle("left"), opacity: 0 }
              : {
                  top: rowPositions[`param:${param.name}`] ?? 0,
                  left: 0,
                  backgroundColor: getParamTypeColor(param.type),
                }
          }
        />
      ))}

      {/* Prompt input handle */}
      {supportsPrompts && (
        <Handle
          type="target"
          position={Position.Left}
          id={buildHandleId("param", "__prompt")}
          className={
            collapsed
              ? "!w-0 !h-0 !border-0 !min-w-0 !min-h-0"
              : "!w-2.5 !h-2.5 !border-0"
          }
          style={
            collapsed
              ? { ...collapsedHandleStyle("left"), opacity: 0 }
              : {
                  top: rowPositions["prompt"] ?? 0,
                  left: 0,
                  backgroundColor: PARAM_TYPE_COLORS.string,
                }
          }
        />
      )}

      {/* LoRA compound input handle */}
      {supportsLoRA && (
        <Handle
          type="target"
          position={Position.Left}
          id={buildHandleId("param", "__loras")}
          className={
            collapsed
              ? "!w-0 !h-0 !border-0 !min-w-0 !min-h-0"
              : "!w-2.5 !h-2.5 !border-0"
          }
          style={
            collapsed
              ? { ...collapsedHandleStyle("left"), opacity: 0 }
              : {
                  top: rowPositions["lora"] ?? 0,
                  left: 0,
                  backgroundColor: "#f472b6",
                }
          }
        />
      )}

      {/* VACE compound input handle */}
      {supportsVace && (
        <Handle
          type="target"
          position={Position.Left}
          id={buildHandleId("param", "__vace")}
          className={
            collapsed
              ? "!w-0 !h-0 !border-0 !min-w-0 !min-h-0"
              : "!w-2.5 !h-2.5 !border-0"
          }
          style={
            collapsed
              ? { ...collapsedHandleStyle("left"), opacity: 0 }
              : {
                  top: rowPositions["vace"] ?? 0,
                  left: 0,
                  backgroundColor: "#a78bfa",
                }
          }
        />
      )}

      {/* Reset cache input handle */}
      {supportsCacheManagement && (
        <Handle
          type="target"
          position={Position.Left}
          id={buildHandleId("param", "reset_cache")}
          className={
            collapsed
              ? "!w-0 !h-0 !border-0 !min-w-0 !min-h-0"
              : "!w-2.5 !h-2.5 !border-0"
          }
          style={
            collapsed
              ? { ...collapsedHandleStyle("left"), opacity: 0 }
              : {
                  top: rowPositions["param:reset_cache"] ?? 0,
                  left: 0,
                  backgroundColor: PARAM_TYPE_COLORS.boolean,
                }
          }
        />
      )}

      {/* Stream output handles */}
      {streamOutputs.map((port, idx) => (
        <Handle
          key={`source-${port}`}
          type="source"
          position={Position.Right}
          id={buildHandleId("stream", port)}
          className={
            collapsed && idx > 0
              ? "!w-0 !h-0 !border-0 !min-w-0 !min-h-0"
              : "!w-2.5 !h-2.5 !border-0"
          }
          style={
            collapsed
              ? idx === 0
                ? collapsedHandleStyle("right")
                : { ...collapsedHandleStyle("right"), opacity: 0 }
              : {
                  top: rowPositions[`out:${port}`] ?? 0,
                  right: 0,
                  backgroundColor: getPortColorHex(port),
                }
          }
        />
      ))}
    </NodeCard>
  );
}
