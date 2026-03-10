import { Handle, Position } from "@xyflow/react";
import type { NodeProps, Node } from "@xyflow/react";
import { useCallback, useRef } from "react";
import type { FlowNodeData } from "../../../lib/graphUtils";
import { buildHandleId } from "../../../lib/graphUtils";
import { useNodeData } from "../hooks/useNodeData";
import { NodeCard, NodeHeader, NodeBody, NODE_TOKENS } from "../ui";

type XYPadNodeType = Node<FlowNodeData, "xypad">;

const COLOR_X = "#38bdf8"; // sky-400
const COLOR_Y = "#4ade80"; // green-400
const DOT_COLOR = "#fafafa";
const PAD_SIZE = 160;
const HEADER_HEIGHT = 28;
const BODY_PAD = 6; // py-1.5 ≈ 6px

export function XYPadNode({ id, data, selected }: NodeProps<XYPadNodeType>) {
  const { updateData } = useNodeData(id);
  const padRef = useRef<HTMLDivElement>(null);

  const minX = data.padMinX ?? 0;
  const maxX = data.padMaxX ?? 1;
  const minY = data.padMinY ?? 0;
  const maxY = data.padMaxY ?? 1;
  const padX = typeof data.padX === "number" ? data.padX : (minX + maxX) / 2;
  const padY = typeof data.padY === "number" ? data.padY : (minY + maxY) / 2;

  const clampedX = Math.min(Math.max(padX, minX), maxX);
  const clampedY = Math.min(Math.max(padY, minY), maxY);
  const pctX = maxX > minX ? (clampedX - minX) / (maxX - minX) : 0.5;
  const pctY = maxY > minY ? (clampedY - minY) / (maxY - minY) : 0.5;

  const setValuesFromMouse = useCallback(
    (clientX: number, clientY: number) => {
      if (!padRef.current) return;
      const rect = padRef.current.getBoundingClientRect();
      let ratioX = (clientX - rect.left) / rect.width;
      let ratioY = (clientY - rect.top) / rect.height;
      ratioX = Math.min(Math.max(ratioX, 0), 1);
      ratioY = Math.min(Math.max(ratioY, 0), 1);
      const newX = parseFloat((minX + ratioX * (maxX - minX)).toFixed(6));
      const newY = parseFloat((maxY - ratioY * (maxY - minY)).toFixed(6));
      updateData({ padX: newX, padY: newY });
    },
    [minX, maxX, minY, maxY, updateData]
  );

  const handlePointerDown = useCallback(
    (e: React.PointerEvent) => {
      e.preventDefault();
      e.stopPropagation();
      const target = e.currentTarget as HTMLElement;
      target.setPointerCapture(e.pointerId);
      setValuesFromMouse(e.clientX, e.clientY);

      const onMove = (ev: PointerEvent) =>
        setValuesFromMouse(ev.clientX, ev.clientY);
      const onUp = () => {
        target.removeEventListener("pointermove", onMove);
        target.removeEventListener("pointerup", onUp);
      };
      target.addEventListener("pointermove", onMove);
      target.addEventListener("pointerup", onUp);
    },
    [setValuesFromMouse]
  );

  const dotLeft = pctX * 100;
  const dotTop = (1 - pctY) * 100;

  const rangeX = maxX - minX;
  const rangeY = maxY - minY;
  const dp = (r: number) => (r >= 10 ? 0 : 2);

  // Center Y positions on pad
  const padCenterY = HEADER_HEIGHT + BODY_PAD + PAD_SIZE / 2;
  const handleSpacing = 16;

  return (
    <NodeCard selected={selected} autoMinHeight>
      <NodeHeader
        title={data.customTitle || "XY Pad"}
        dotColor="bg-sky-400"
        onTitleChange={newTitle => updateData({ customTitle: newTitle })}
      />
      <NodeBody>
        <div className="flex flex-col gap-1">
          {/* The pad */}
          <div
            ref={padRef}
            className="relative rounded-lg cursor-crosshair select-none overflow-hidden"
            style={{
              width: PAD_SIZE,
              height: PAD_SIZE,
              background: "#1b1a1a",
              border: "1px solid rgba(119,119,119,0.15)",
            }}
            onPointerDown={handlePointerDown}
          >
            {/* Grid lines */}
            <div
              className="absolute inset-0 pointer-events-none"
              style={{ opacity: 0.08 }}
            >
              <div className="absolute left-1/4 top-0 bottom-0 w-px bg-white" />
              <div className="absolute left-1/2 top-0 bottom-0 w-px bg-white" />
              <div className="absolute left-3/4 top-0 bottom-0 w-px bg-white" />
              <div className="absolute top-1/4 left-0 right-0 h-px bg-white" />
              <div className="absolute top-1/2 left-0 right-0 h-px bg-white" />
              <div className="absolute top-3/4 left-0 right-0 h-px bg-white" />
            </div>
            {/* Crosshairs */}
            <div
              className="absolute pointer-events-none"
              style={{
                left: `${dotLeft}%`,
                top: 0,
                bottom: 0,
                width: 1,
                background: `${COLOR_X}33`,
              }}
            />
            <div
              className="absolute pointer-events-none"
              style={{
                top: `${dotTop}%`,
                left: 0,
                right: 0,
                height: 1,
                background: `${COLOR_Y}33`,
              }}
            />
            {/* Dot */}
            <div
              className="absolute w-3 h-3 rounded-full pointer-events-none -translate-x-1/2 -translate-y-1/2"
              style={{
                left: `${dotLeft}%`,
                top: `${dotTop}%`,
                background: DOT_COLOR,
                boxShadow: `0 0 6px ${DOT_COLOR}`,
              }}
            />
          </div>

          {/* Value display */}
          <div className="flex justify-between px-1">
            <span className={NODE_TOKENS.primaryText}>
              X:{" "}
              <span style={{ color: COLOR_X }}>
                {clampedX.toFixed(dp(rangeX))}
              </span>
            </span>
            <span className={NODE_TOKENS.primaryText}>
              Y:{" "}
              <span style={{ color: COLOR_Y }}>
                {clampedY.toFixed(dp(rangeY))}
              </span>
            </span>
          </div>

          {/* Range settings — stacked vertically */}
          <div className="flex flex-col gap-0.5">
            <div className="flex items-center gap-1">
              <span className={`${NODE_TOKENS.labelText} w-3 shrink-0`}>X</span>
              <input
                className={`${NODE_TOKENS.pillInput} ${NODE_TOKENS.pillInputNumber} !w-[36px] !text-[8px] !px-0.5 !py-0`}
                type="number"
                value={minX}
                onChange={e => updateData({ padMinX: Number(e.target.value) })}
                onMouseDown={e => e.stopPropagation()}
                title="Min X"
              />
              <span className={`${NODE_TOKENS.labelText} shrink-0`}>–</span>
              <input
                className={`${NODE_TOKENS.pillInput} ${NODE_TOKENS.pillInputNumber} !w-[36px] !text-[8px] !px-0.5 !py-0`}
                type="number"
                value={maxX}
                onChange={e => updateData({ padMaxX: Number(e.target.value) })}
                onMouseDown={e => e.stopPropagation()}
                title="Max X"
              />
            </div>
            <div className="flex items-center gap-1">
              <span className={`${NODE_TOKENS.labelText} w-3 shrink-0`}>Y</span>
              <input
                className={`${NODE_TOKENS.pillInput} ${NODE_TOKENS.pillInputNumber} !w-[36px] !text-[8px] !px-0.5 !py-0`}
                type="number"
                value={minY}
                onChange={e => updateData({ padMinY: Number(e.target.value) })}
                onMouseDown={e => e.stopPropagation()}
                title="Min Y"
              />
              <span className={`${NODE_TOKENS.labelText} shrink-0`}>–</span>
              <input
                className={`${NODE_TOKENS.pillInput} ${NODE_TOKENS.pillInputNumber} !w-[36px] !text-[8px] !px-0.5 !py-0`}
                type="number"
                value={maxY}
                onChange={e => updateData({ padMaxY: Number(e.target.value) })}
                onMouseDown={e => e.stopPropagation()}
                title="Max Y"
              />
            </div>
          </div>
        </div>
      </NodeBody>

      {/* Input handles (left) — centered on pad */}
      <Handle
        type="target"
        position={Position.Left}
        id={buildHandleId("param", "x")}
        className="!w-2 !h-2 !border-0"
        style={{
          top: padCenterY - handleSpacing / 2,
          left: 8,
          backgroundColor: COLOR_X,
        }}
      />
      <Handle
        type="target"
        position={Position.Left}
        id={buildHandleId("param", "y")}
        className="!w-2 !h-2 !border-0"
        style={{
          top: padCenterY + handleSpacing / 2,
          left: 8,
          backgroundColor: COLOR_Y,
        }}
      />

      {/* Output handles (right) — centered on pad */}
      <Handle
        type="source"
        position={Position.Right}
        id={buildHandleId("param", "x")}
        className="!w-2 !h-2 !border-0"
        style={{
          top: padCenterY - handleSpacing / 2,
          right: 8,
          backgroundColor: COLOR_X,
        }}
      />
      <Handle
        type="source"
        position={Position.Right}
        id={buildHandleId("param", "y")}
        className="!w-2 !h-2 !border-0"
        style={{
          top: padCenterY + handleSpacing / 2,
          right: 8,
          backgroundColor: COLOR_Y,
        }}
      />
    </NodeCard>
  );
}
