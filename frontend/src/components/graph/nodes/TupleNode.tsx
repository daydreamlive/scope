import { Handle, Position } from "@xyflow/react";
import type { NodeProps, Node } from "@xyflow/react";
import { useCallback } from "react";
import type { FlowNodeData } from "../../../lib/graphUtils";
import { buildHandleId } from "../../../lib/graphUtils";
import { useNodeData } from "../hooks/node/useNodeData";
import { useNodeCollapse } from "../hooks/node/useNodeCollapse";
import {
  NodeCard,
  NodeHeader,
  NodeBody,
  NodePillToggle,
  NODE_TOKENS,
  collapsedHandleStyle,
} from "../ui";

type TupleNodeType = Node<FlowNodeData, "tuple">;

const COLOR = "#fb923c";

const ROW_HEIGHT = 22;
const SETTINGS_HEIGHT = 48;
const HEADER_HEIGHT = 28;
const BODY_PAD = 6;

export function TupleNode({ id, data, selected }: NodeProps<TupleNodeType>) {
  const { updateData } = useNodeData(id);
  const { collapsed, toggleCollapse } = useNodeCollapse();

  const values =
    data.tupleValues && data.tupleValues.length > 0 ? data.tupleValues : [0];
  const tMin = data.tupleMin ?? 0;
  const tMax = data.tupleMax ?? 1000;
  const tStep = data.tupleStep ?? 1;
  const enforceOrder = data.tupleEnforceOrder ?? true;
  const orderDir = data.tupleOrderDirection ?? "desc";

  const updateValues = useCallback(
    (newValues: number[]) => {
      updateData({ tupleValues: newValues });
    },
    [updateData]
  );

  const setValueAt = useCallback(
    (index: number, raw: number) => {
      let v = Math.min(Math.max(raw, tMin), tMax);
      v = Math.round(v / tStep) * tStep;
      v = parseFloat(v.toFixed(10));

      const newValues = [...values];

      if (enforceOrder && newValues.length > 1) {
        if (orderDir === "desc") {
          const upper = index > 0 ? newValues[index - 1] : tMax;
          const lower =
            index < newValues.length - 1 ? newValues[index + 1] : tMin;
          v = Math.min(Math.max(v, lower), upper);
        } else {
          const lower = index > 0 ? newValues[index - 1] : tMin;
          const upper =
            index < newValues.length - 1 ? newValues[index + 1] : tMax;
          v = Math.min(Math.max(v, lower), upper);
        }
      }

      newValues[index] = v;
      updateValues(newValues);
    },
    [values, tMin, tMax, tStep, enforceOrder, orderDir, updateValues]
  );

  const addRow = useCallback(() => {
    const last = values[values.length - 1];
    let newVal: number;
    if (enforceOrder && orderDir === "desc") {
      newVal = Math.max(last - tStep, tMin);
    } else if (enforceOrder && orderDir === "asc") {
      newVal = Math.min(last + tStep, tMax);
    } else {
      newVal = last;
    }
    updateValues([...values, parseFloat(newVal.toFixed(10))]);
  }, [values, enforceOrder, orderDir, tMin, tMax, tStep, updateValues]);

  const removeRow = useCallback(
    (index: number) => {
      if (values.length <= 1) return;
      updateValues(values.filter((_, i) => i !== index));
    },
    [values, updateValues]
  );

  const rangeSize = tMax - tMin;
  const decimalPlaces = tStep < 1 ? 2 : 0;

  const rowsSectionTop = HEADER_HEIGHT + BODY_PAD + SETTINGS_HEIGHT;

  return (
    <NodeCard
      selected={selected}
      autoMinHeight={!collapsed}
      collapsed={collapsed}
    >
      <NodeHeader
        title={data.customTitle || "Tuple"}
        onTitleChange={newTitle => updateData({ customTitle: newTitle })}
        collapsed={collapsed}
        onCollapseToggle={toggleCollapse}
      />
      {!collapsed && (
        <NodeBody>
          <div className="flex flex-col gap-1">
            {/* Min / Max / Step row */}
            <div className="flex items-center gap-1">
              <span className={`${NODE_TOKENS.labelText} shrink-0`}>Min</span>
              <input
                className={`${NODE_TOKENS.pillInput} ${NODE_TOKENS.pillInputNumber} !w-[36px] !text-[8px] !px-1 !py-0`}
                type="number"
                value={tMin}
                onChange={e => updateData({ tupleMin: Number(e.target.value) })}
                onMouseDown={e => e.stopPropagation()}
              />
              <span className={`${NODE_TOKENS.labelText} shrink-0`}>Max</span>
              <input
                className={`${NODE_TOKENS.pillInput} ${NODE_TOKENS.pillInputNumber} !w-[36px] !text-[8px] !px-1 !py-0`}
                type="number"
                value={tMax}
                onChange={e => updateData({ tupleMax: Number(e.target.value) })}
                onMouseDown={e => e.stopPropagation()}
              />
              <span className={`${NODE_TOKENS.labelText} shrink-0`}>Step</span>
              <input
                className={`${NODE_TOKENS.pillInput} ${NODE_TOKENS.pillInputNumber} !w-[36px] !text-[8px] !px-1 !py-0`}
                type="number"
                value={tStep}
                onChange={e =>
                  updateData({ tupleStep: Number(e.target.value) })
                }
                onMouseDown={e => e.stopPropagation()}
              />
            </div>

            {/* Enforce order */}
            <div className="flex items-center justify-between gap-2">
              <div className="flex items-center gap-1.5">
                <span className={NODE_TOKENS.labelText}>Order</span>
                <NodePillToggle
                  checked={enforceOrder}
                  onChange={v => updateData({ tupleEnforceOrder: v })}
                />
              </div>
              {enforceOrder && (
                <button
                  className={`${NODE_TOKENS.pill} text-[9px] text-[#aaa] hover:text-[#fff] cursor-pointer transition-colors`}
                  onClick={() =>
                    updateData({
                      tupleOrderDirection: orderDir === "desc" ? "asc" : "desc",
                    })
                  }
                >
                  {orderDir === "desc" ? "\u2193 Desc" : "\u2191 Asc"}
                </button>
              )}
            </div>

            {/* Value rows */}
            <div className="flex flex-col gap-1">
              {values.map((v, i) => {
                const pct = rangeSize > 0 ? ((v - tMin) / rangeSize) * 100 : 0;
                return (
                  <div key={i} className="flex items-center gap-1">
                    <span
                      className={`${NODE_TOKENS.labelText} w-3 text-right shrink-0`}
                    >
                      {i}
                    </span>
                    {/* Mini slider */}
                    <div
                      className="relative flex-1 h-4 rounded-full cursor-pointer select-none"
                      style={{
                        background: "#1b1a1a",
                        border: "1px solid rgba(119,119,119,0.15)",
                      }}
                      onPointerDown={e => {
                        e.preventDefault();
                        e.stopPropagation();
                        const target = e.currentTarget as HTMLElement;
                        target.setPointerCapture(e.pointerId);
                        const setFromMouse = (clientX: number) => {
                          const rect = target.getBoundingClientRect();
                          let ratio = (clientX - rect.left) / rect.width;
                          ratio = Math.min(Math.max(ratio, 0), 1);
                          setValueAt(i, tMin + ratio * rangeSize);
                        };
                        setFromMouse(e.clientX);
                        const onMove = (ev: PointerEvent) =>
                          setFromMouse(ev.clientX);
                        const onUp = () => {
                          target.removeEventListener("pointermove", onMove);
                          target.removeEventListener("pointerup", onUp);
                        };
                        target.addEventListener("pointermove", onMove);
                        target.addEventListener("pointerup", onUp);
                      }}
                    >
                      <div
                        className="absolute left-0 top-0 h-full rounded-full pointer-events-none"
                        style={{
                          width: `${pct}%`,
                          background: COLOR,
                          opacity: 0.3,
                        }}
                      />
                      <div
                        className="absolute top-1/2 -translate-y-1/2 w-2 h-2 rounded-full pointer-events-none"
                        style={{
                          left: `calc(${pct}% - 4px)`,
                          background: COLOR,
                        }}
                      />
                    </div>
                    {/* Numeric input */}
                    <input
                      className={`${NODE_TOKENS.pillInput} ${NODE_TOKENS.pillInputNumber} !w-[40px] !text-[9px] !px-1`}
                      type="number"
                      value={v.toFixed(decimalPlaces)}
                      onChange={e =>
                        setValueAt(i, parseFloat(e.target.value) || 0)
                      }
                      onMouseDown={e => e.stopPropagation()}
                      step={tStep}
                      min={tMin}
                      max={tMax}
                    />
                    {/* Remove */}
                    {values.length > 1 && (
                      <button
                        className="w-4 h-4 rounded text-[#555] hover:text-red-400 text-[10px] flex items-center justify-center leading-none shrink-0"
                        onClick={() => removeRow(i)}
                      >
                        ×
                      </button>
                    )}
                  </div>
                );
              })}
            </div>

            {/* Add row */}
            <button
              className="w-full py-0.5 rounded bg-[#1b1a1a] border border-[rgba(119,119,119,0.15)] text-[#888] hover:text-[#ccc] hover:border-[rgba(119,119,119,0.4)] text-[10px] transition-colors"
              onClick={addRow}
            >
              + Add
            </button>
          </div>
        </NodeBody>
      )}

      {/* Tuple-level input (receives entire list_number) */}
      <Handle
        type="target"
        position={Position.Left}
        id={buildHandleId("param", "value")}
        className={
          collapsed
            ? "!w-0 !h-0 !border-0 !min-w-0 !min-h-0"
            : "!w-2.5 !h-2.5 !border-0"
        }
        style={
          collapsed
            ? { ...collapsedHandleStyle("left"), opacity: 0 }
            : {
                top: HEADER_HEIGHT + BODY_PAD + 8,
                left: 0,
                backgroundColor: COLOR,
              }
        }
      />

      {/* Per-row inputs */}
      {values.map((_, i) => {
        const yOffset = rowsSectionTop + i * ROW_HEIGHT + ROW_HEIGHT / 2;
        return (
          <Handle
            key={`row-in-${i}`}
            type="target"
            position={Position.Left}
            id={buildHandleId("param", `row_${i}`)}
            className={
              collapsed
                ? "!w-0 !h-0 !border-0 !min-w-0 !min-h-0"
                : "!w-2.5 !h-2.5 !border-0"
            }
            style={
              collapsed
                ? { ...collapsedHandleStyle("left"), opacity: 0 }
                : { top: yOffset, left: 0, backgroundColor: "#38bdf8" }
            }
          />
        );
      })}

      {/* Output */}
      <Handle
        type="source"
        position={Position.Right}
        id={buildHandleId("param", "value")}
        className="!w-2.5 !h-2.5 !border-0"
        style={
          collapsed
            ? collapsedHandleStyle("right")
            : {
                top: HEADER_HEIGHT + BODY_PAD + 8,
                right: 0,
                backgroundColor: COLOR,
              }
        }
      />
    </NodeCard>
  );
}
