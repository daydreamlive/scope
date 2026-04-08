import { Handle, Position } from "@xyflow/react";
import type { NodeProps, Node } from "@xyflow/react";
import { useCallback } from "react";
import type { FlowNodeData } from "../../../lib/graphUtils";
import { buildHandleId } from "../../../lib/graphUtils";
import { useNodeData } from "../hooks/node/useNodeData";
import { useNodeCollapse } from "../hooks/node/useNodeCollapse";
import { useHandlePositions } from "../hooks/node/useHandlePositions";
import { useConnectedNumber } from "../hooks/node/useConnectedValue";
import { useSlider } from "../hooks/node/useSlider";
import {
  NodeCard,
  NodeHeader,
  NodeBody,
  NodeParamRow,
  NodePillInput,
  NodePill,
  NODE_TOKENS,
  collapsedHandleStyle,
} from "../ui";
import { COLOR_NUMBER as COLOR } from "../nodeColors";

type SliderNodeType = Node<FlowNodeData, "slider">;

export function SliderNode({ id, data, selected }: NodeProps<SliderNodeType>) {
  const { updateData } = useNodeData(id);
  const { collapsed, toggleCollapse } = useNodeCollapse();

  const min = data.sliderMin ?? 0;
  const max = data.sliderMax ?? 1;
  const rawStep = data.sliderStep ?? 0.01;
  const step = Number.isFinite(rawStep) && rawStep > 0 ? rawStep : 0.01;

  const minIn = useConnectedNumber(id, "min", min);
  const maxIn = useConnectedNumber(id, "max", max);
  const effectiveMin = minIn.value;
  const effectiveMax = maxIn.value;

  const value = typeof data.value === "number" ? data.value : effectiveMin;

  const onSliderChange = useCallback(
    (v: number) => updateData({ value: v }),
    [updateData]
  );

  const { sliderRef, clampedValue, pct, handlePointerDown } = useSlider({
    min: effectiveMin,
    max: effectiveMax,
    step,
    value,
    onChange: onSliderChange,
  });

  const { setRowRef, rowPositions } = useHandlePositions([
    minIn.connected,
    maxIn.connected,
    collapsed,
  ]);

  return (
    <NodeCard
      selected={selected}
      autoMinHeight={!collapsed}
      collapsed={collapsed}
    >
      <NodeHeader
        title={data.customTitle || "Slider"}
        onTitleChange={newTitle => updateData({ customTitle: newTitle })}
        collapsed={collapsed}
        onCollapseToggle={toggleCollapse}
      />
      {!collapsed && (
        <NodeBody withGap>
          {/* Slider track */}
          <div ref={setRowRef("slider")}>
          <div
            ref={sliderRef}
            className="relative w-full h-5 rounded-full cursor-pointer select-none"
            style={{
              background: "#1b1a1a",
              border: "1px solid rgba(119,119,119,0.15)",
            }}
            onPointerDown={handlePointerDown}
          >
            {/* Filled portion */}
            <div
              className="absolute left-0 top-0 h-full rounded-full pointer-events-none"
              style={{ width: `${pct}%`, background: COLOR, opacity: 0.35 }}
            />
            {/* Thumb */}
            <div
              className="absolute top-1/2 -translate-y-1/2 w-3 h-3 rounded-full pointer-events-none"
              style={{
                left: `calc(${pct}% - 6px)`,
                background: COLOR,
                boxShadow: `0 0 4px ${COLOR}`,
              }}
            />
          </div>
          </div>

          {/* Current value display */}
          <div className="flex justify-center">
            <span className={NODE_TOKENS.primaryText}>
              {clampedValue.toFixed(step < 1 ? 2 : 0)}
            </span>
          </div>

          {/* Min / Max / Step */}
          <div ref={setRowRef("min")}>
            <NodeParamRow label="Min">
              {minIn.connected ? (
                <NodePill className="opacity-50">{effectiveMin}</NodePill>
              ) : (
                <NodePillInput
                  type="number"
                  value={min}
                  onChange={v => updateData({ sliderMin: Number(v) })}
                />
              )}
            </NodeParamRow>
          </div>
          <div ref={setRowRef("max")}>
            <NodeParamRow label="Max">
              {maxIn.connected ? (
                <NodePill className="opacity-50">{effectiveMax}</NodePill>
              ) : (
                <NodePillInput
                  type="number"
                  value={max}
                  onChange={v => updateData({ sliderMax: Number(v) })}
                />
              )}
            </NodeParamRow>
          </div>
          <NodeParamRow label="Step">
            <NodePillInput
              type="number"
              value={step}
              onChange={v => updateData({ sliderStep: Number(v) })}
            />
          </NodeParamRow>
        </NodeBody>
      )}

      {/* Input handle (left) */}
      <Handle
        type="target"
        position={Position.Left}
        id={buildHandleId("param", "value")}
        className="!w-2.5 !h-2.5 !border-0"
        style={
          collapsed
            ? collapsedHandleStyle("left")
            : { top: rowPositions["slider"] ?? 44, left: 0, backgroundColor: COLOR }
        }
      />

      {/* Min input handle */}
      <Handle
        type="target"
        position={Position.Left}
        id={buildHandleId("param", "min")}
        className={
          collapsed
            ? "!w-0 !h-0 !border-0 !min-w-0 !min-h-0"
            : "!w-2.5 !h-2.5 !border-0"
        }
        style={
          collapsed
            ? { ...collapsedHandleStyle("left"), opacity: 0 }
            : {
                top: rowPositions["min"] ?? 100,
                left: 0,
                backgroundColor: COLOR,
              }
        }
      />
      {/* Max input handle */}
      <Handle
        type="target"
        position={Position.Left}
        id={buildHandleId("param", "max")}
        className={
          collapsed
            ? "!w-0 !h-0 !border-0 !min-w-0 !min-h-0"
            : "!w-2.5 !h-2.5 !border-0"
        }
        style={
          collapsed
            ? { ...collapsedHandleStyle("left"), opacity: 0 }
            : {
                top: rowPositions["max"] ?? 122,
                left: 0,
                backgroundColor: COLOR,
              }
        }
      />

      {/* Output handle (right) */}
      <Handle
        type="source"
        position={Position.Right}
        id={buildHandleId("param", "value")}
        className="!w-2.5 !h-2.5 !border-0"
        style={
          collapsed
            ? collapsedHandleStyle("right")
            : { top: rowPositions["slider"] ?? 44, right: 0, backgroundColor: COLOR }
        }
      />
    </NodeCard>
  );
}
