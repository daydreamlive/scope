import { Handle, Position, useUpdateNodeInternals } from "@xyflow/react";
import type { NodeProps, Node } from "@xyflow/react";
import { useCallback, useEffect, useRef } from "react";
import type { FlowNodeData } from "../../../lib/graphUtils";
import { buildHandleId } from "../../../lib/graphUtils";
import { useNodeData } from "../hooks/node/useNodeData";
import { useNodeCollapse } from "../hooks/node/useNodeCollapse";
import {
  NodeCard,
  NodeHeader,
  NodeBody,
  NODE_TOKENS,
  collapsedHandleStyle,
} from "../ui";
import { COLOR_NUMBER as COLOR } from "../nodeColors";

type KnobsNodeType = Node<FlowNodeData, "knobs">;
const KNOB_SIZE = 40;
const KNOB_RADIUS = 16;
const ARC_DEG = 270;
const ARC_START = (360 - ARC_DEG) / 2 + 90; // 135 deg
const ARC_END = ARC_START + ARC_DEG; // 405 deg

/** Height of a single knob row (used for handle positioning). */
const ROW_HEIGHT = 48;
const HEADER_HEIGHT = 28;
const BODY_PAD_TOP = 6;

interface KnobDef {
  label: string;
  min: number;
  max: number;
  value: number;
}

function defaultKnob(index: number): KnobDef {
  return { label: `Knob ${index + 1}`, min: 0, max: 1, value: 0 };
}

function polarToXY(cx: number, cy: number, r: number, deg: number) {
  const rad = (deg * Math.PI) / 180;
  return { x: cx + r * Math.cos(rad), y: cy + r * Math.sin(rad) };
}

function describeArc(
  cx: number,
  cy: number,
  r: number,
  startDeg: number,
  endDeg: number
) {
  const start = polarToXY(cx, cy, r, endDeg);
  const end = polarToXY(cx, cy, r, startDeg);
  const large = endDeg - startDeg > 180 ? 1 : 0;
  return `M ${start.x} ${start.y} A ${r} ${r} 0 ${large} 0 ${end.x} ${end.y}`;
}

/** A single horizontal row: [knob SVG + value] [label] [min/max] [×] */
function SingleKnobRow({
  knob,
  index,
  onValueChange,
  onFieldChange,
  onRemove,
  canRemove,
}: {
  knob: KnobDef;
  index: number;
  onValueChange: (index: number, value: number) => void;
  onFieldChange: (
    index: number,
    field: keyof KnobDef,
    value: string | number
  ) => void;
  onRemove: (index: number) => void;
  canRemove: boolean;
}) {
  const dragRef = useRef<{ startY: number; startValue: number } | null>(null);
  const { min, max, value } = knob;
  const clamped = Math.min(Math.max(value, min), max);
  const pct = max > min ? (clamped - min) / (max - min) : 0;
  const angleDeg = ARC_START + pct * ARC_DEG;
  const cx = KNOB_SIZE / 2;
  const cy = KNOB_SIZE / 2;

  const handlePointerDown = useCallback(
    (e: React.PointerEvent) => {
      e.preventDefault();
      e.stopPropagation();
      const target = e.currentTarget as HTMLElement;
      target.setPointerCapture(e.pointerId);
      dragRef.current = { startY: e.clientY, startValue: clamped };

      const onMove = (ev: PointerEvent) => {
        if (!dragRef.current) return;
        const dy = dragRef.current.startY - ev.clientY;
        const range = max - min;
        const sensitivity = range / 120;
        let newVal = dragRef.current.startValue + dy * sensitivity;
        newVal = Math.min(Math.max(newVal, min), max);
        onValueChange(index, parseFloat(newVal.toFixed(10)));
      };
      const onUp = () => {
        dragRef.current = null;
        target.removeEventListener("pointermove", onMove);
        target.removeEventListener("pointerup", onUp);
      };
      target.addEventListener("pointermove", onMove);
      target.addEventListener("pointerup", onUp);
    },
    [clamped, min, max, index, onValueChange]
  );

  const indEnd = polarToXY(cx, cy, KNOB_RADIUS - 3, angleDeg);

  return (
    <div className="flex items-center gap-1.5" style={{ height: ROW_HEIGHT }}>
      {/* Knob SVG */}
      <svg
        width={KNOB_SIZE}
        height={KNOB_SIZE}
        className="cursor-ns-resize select-none shrink-0"
        onPointerDown={handlePointerDown}
      >
        <path
          d={describeArc(cx, cy, KNOB_RADIUS, ARC_START, ARC_END)}
          fill="none"
          stroke="rgba(119,119,119,0.25)"
          strokeWidth={2.5}
          strokeLinecap="round"
        />
        {pct > 0.001 && (
          <path
            d={describeArc(cx, cy, KNOB_RADIUS, ARC_START, angleDeg)}
            fill="none"
            stroke={COLOR}
            strokeWidth={2.5}
            strokeLinecap="round"
          />
        )}
        <circle
          cx={cx}
          cy={cy}
          r={5}
          fill="#2a2a2a"
          stroke="rgba(119,119,119,0.35)"
          strokeWidth={1}
        />
        <line
          x1={cx}
          y1={cy}
          x2={indEnd.x}
          y2={indEnd.y}
          stroke={COLOR}
          strokeWidth={2}
          strokeLinecap="round"
        />
      </svg>

      {/* Value */}
      <span
        className={`${NODE_TOKENS.primaryText} w-[32px] text-center shrink-0`}
      >
        {clamped.toFixed(max - min >= 10 ? 0 : 2)}
      </span>

      {/* Label */}
      <input
        className={`${NODE_TOKENS.pillInput} ${NODE_TOKENS.pillInputText} !w-auto flex-1 min-w-0 !text-[9px] !px-1 !py-0`}
        value={knob.label}
        onChange={e => onFieldChange(index, "label", e.target.value)}
        onMouseDown={e => e.stopPropagation()}
      />

      {/* Min / Max */}
      <input
        className={`${NODE_TOKENS.pillInput} ${NODE_TOKENS.pillInputNumber} !w-[28px] !text-[8px] !px-0.5 !py-0 shrink-0`}
        type="number"
        value={knob.min}
        onChange={e =>
          onFieldChange(index, "min", parseFloat(e.target.value) || 0)
        }
        onMouseDown={e => e.stopPropagation()}
        title="Min"
      />
      <input
        className={`${NODE_TOKENS.pillInput} ${NODE_TOKENS.pillInputNumber} !w-[28px] !text-[8px] !px-0.5 !py-0 shrink-0`}
        type="number"
        value={knob.max}
        onChange={e =>
          onFieldChange(index, "max", parseFloat(e.target.value) || 1)
        }
        onMouseDown={e => e.stopPropagation()}
        title="Max"
      />

      {/* Remove button */}
      {canRemove && (
        <button
          className="w-4 h-4 rounded text-[#555] hover:text-red-400 text-[10px] flex items-center justify-center leading-none shrink-0"
          onClick={() => onRemove(index)}
        >
          ×
        </button>
      )}
    </div>
  );
}

export function KnobsNode({ id, data, selected }: NodeProps<KnobsNodeType>) {
  const { updateData } = useNodeData(id);
  const { collapsed, toggleCollapse } = useNodeCollapse();
  const updateNodeInternals = useUpdateNodeInternals();
  const knobs: KnobDef[] =
    data.knobs && data.knobs.length > 0 ? data.knobs : [defaultKnob(0)];
  const prevKnobCount = useRef(knobs.length);

  useEffect(() => {
    if (knobs.length !== prevKnobCount.current) {
      prevKnobCount.current = knobs.length;
      updateData({
        parameterOutputs: knobs.map((_, i) => ({
          name: `knob_${i}`,
          type: "number" as const,
          defaultValue: 0,
        })),
      });
      updateNodeInternals(id);
    }
  }, [knobs.length, id, updateData, updateNodeInternals]);

  const updateKnobs = useCallback(
    (newKnobs: KnobDef[]) => {
      updateData({ knobs: newKnobs });
    },
    [updateData]
  );

  const handleValueChange = useCallback(
    (index: number, value: number) => {
      const updated = [...knobs];
      updated[index] = { ...updated[index], value };
      updateKnobs(updated);
    },
    [knobs, updateKnobs]
  );

  const handleFieldChange = useCallback(
    (index: number, field: keyof KnobDef, value: string | number) => {
      const updated = [...knobs];
      updated[index] = { ...updated[index], [field]: value };
      updateKnobs(updated);
    },
    [knobs, updateKnobs]
  );

  const handleRemove = useCallback(
    (index: number) => {
      if (knobs.length <= 1) return;
      const updated = knobs.filter((_, i) => i !== index);
      updateKnobs(updated);
    },
    [knobs, updateKnobs]
  );

  const handleAdd = useCallback(() => {
    updateKnobs([...knobs, defaultKnob(knobs.length)]);
  }, [knobs, updateKnobs]);

  return (
    <NodeCard
      selected={selected}
      autoMinHeight={!collapsed}
      collapsed={collapsed}
    >
      <NodeHeader
        title={data.customTitle || "Knobs"}
        onTitleChange={newTitle => updateData({ customTitle: newTitle })}
        collapsed={collapsed}
        onCollapseToggle={toggleCollapse}
      />
      {!collapsed && (
        <NodeBody>
          <div className="flex flex-col gap-0.5 py-0.5">
            {knobs.map((knob, i) => (
              <SingleKnobRow
                key={i}
                knob={knob}
                index={i}
                onValueChange={handleValueChange}
                onFieldChange={handleFieldChange}
                onRemove={handleRemove}
                canRemove={knobs.length > 1}
              />
            ))}
          </div>
          {/* Add button */}
          <button
            className="w-full py-0.5 mt-0.5 rounded bg-[#1b1a1a] border border-[rgba(119,119,119,0.15)] text-[#888] hover:text-[#ccc] hover:border-[rgba(119,119,119,0.4)] text-[10px] transition-colors"
            onClick={handleAdd}
          >
            + Add Knob
          </button>
        </NodeBody>
      )}

      {/* Input & Output handles for each knob, vertically aligned to each row */}
      {knobs.map((_, i) => {
        const yOffset =
          HEADER_HEIGHT + BODY_PAD_TOP + ROW_HEIGHT * i + ROW_HEIGHT / 2;
        const isFirst = i === 0;
        return (
          <span key={`handles-${i}`}>
            <Handle
              type="target"
              position={Position.Left}
              id={buildHandleId("param", `knob_${i}`)}
              className={
                collapsed && !isFirst
                  ? "!w-0 !h-0 !border-0 !min-w-0 !min-h-0"
                  : "!w-2.5 !h-2.5 !border-0"
              }
              style={
                collapsed
                  ? isFirst
                    ? collapsedHandleStyle("left")
                    : { ...collapsedHandleStyle("left"), opacity: 0 }
                  : { top: yOffset, left: 0, backgroundColor: COLOR }
              }
            />
            <Handle
              type="source"
              position={Position.Right}
              id={buildHandleId("param", `knob_${i}`)}
              className={
                collapsed && !isFirst
                  ? "!w-0 !h-0 !border-0 !min-w-0 !min-h-0"
                  : "!w-2.5 !h-2.5 !border-0"
              }
              style={
                collapsed
                  ? isFirst
                    ? collapsedHandleStyle("right")
                    : { ...collapsedHandleStyle("right"), opacity: 0 }
                  : { top: yOffset, right: 0, backgroundColor: COLOR }
              }
            />
          </span>
        );
      })}
    </NodeCard>
  );
}
