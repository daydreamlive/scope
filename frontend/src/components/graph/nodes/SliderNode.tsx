import { Handle, Position, useReactFlow } from "@xyflow/react";
import type { NodeProps, Node } from "@xyflow/react";
import { useCallback, useRef } from "react";
import type { FlowNodeData } from "../../../lib/graphUtils";
import { buildHandleId } from "../../../lib/graphUtils";
import {
  NodeCard,
  NodeHeader,
  NodeBody,
  NodeParamRow,
  NodePillInput,
  NODE_TOKENS,
} from "../ui";

type SliderNodeType = Node<FlowNodeData, "slider">;

const COLOR = "#a78bfa"; // violet-400

export function SliderNode({ id, data, selected }: NodeProps<SliderNodeType>) {
  const { setNodes } = useReactFlow();
  const sliderRef = useRef<HTMLDivElement>(null);

  const min = data.sliderMin ?? 0;
  const max = data.sliderMax ?? 1;
  const step = data.sliderStep ?? 0.01;
  const value = typeof data.value === "number" ? data.value : min;

  const updateField = useCallback(
    (field: string, v: unknown) => {
      setNodes(nds =>
        nds.map(n =>
          n.id === id ? { ...n, data: { ...n.data, [field]: v } } : n
        )
      );
    },
    [id, setNodes]
  );

  const clampedValue = Math.min(Math.max(value, min), max);
  const pct = max > min ? ((clampedValue - min) / (max - min)) * 100 : 0;

  const setValueFromMouse = useCallback(
    (clientX: number) => {
      if (!sliderRef.current) return;
      const rect = sliderRef.current.getBoundingClientRect();
      let ratio = (clientX - rect.left) / rect.width;
      ratio = Math.min(Math.max(ratio, 0), 1);
      let newVal = min + ratio * (max - min);
      // snap to step
      newVal = Math.round(newVal / step) * step;
      newVal = Math.min(Math.max(newVal, min), max);
      updateField("value", parseFloat(newVal.toFixed(10)));
    },
    [min, max, step, updateField]
  );

  const handlePointerDown = useCallback(
    (e: React.PointerEvent) => {
      e.preventDefault();
      e.stopPropagation();
      const target = e.currentTarget as HTMLElement;
      target.setPointerCapture(e.pointerId);
      setValueFromMouse(e.clientX);

      const onMove = (ev: PointerEvent) => setValueFromMouse(ev.clientX);
      const onUp = () => {
        target.removeEventListener("pointermove", onMove);
        target.removeEventListener("pointerup", onUp);
      };
      target.addEventListener("pointermove", onMove);
      target.addEventListener("pointerup", onUp);
    },
    [setValueFromMouse]
  );

  return (
    <NodeCard selected={selected} autoMinHeight>
      <NodeHeader title="Slider" dotColor="bg-violet-400" />
      <NodeBody withGap>
        {/* Slider track */}
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

        {/* Current value display */}
        <div className="flex justify-center">
          <span className={NODE_TOKENS.primaryText}>
            {clampedValue.toFixed(step < 1 ? 2 : 0)}
          </span>
        </div>

        {/* Min / Max / Step */}
        <NodeParamRow label="Min">
          <NodePillInput
            type="number"
            value={min}
            onChange={v => updateField("sliderMin", Number(v))}
          />
        </NodeParamRow>
        <NodeParamRow label="Max">
          <NodePillInput
            type="number"
            value={max}
            onChange={v => updateField("sliderMax", Number(v))}
          />
        </NodeParamRow>
        <NodeParamRow label="Step">
          <NodePillInput
            type="number"
            value={step}
            onChange={v => updateField("sliderStep", Number(v))}
          />
        </NodeParamRow>
      </NodeBody>

      {/* Input handle (left) */}
      <Handle
        type="target"
        position={Position.Left}
        id={buildHandleId("param", "value")}
        className="!w-2 !h-2 !border-0"
        style={{ top: 44, left: 8, backgroundColor: COLOR }}
      />

      {/* Output handle (right) */}
      <Handle
        type="source"
        position={Position.Right}
        id={buildHandleId("param", "value")}
        className="!w-2 !h-2 !border-0"
        style={{ top: 44, right: 8, backgroundColor: COLOR }}
      />
    </NodeCard>
  );
}
