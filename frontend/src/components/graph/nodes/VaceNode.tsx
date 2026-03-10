import { useCallback, useRef } from "react";
import { Handle, Position } from "@xyflow/react";
import type { NodeProps, Node } from "@xyflow/react";
import type { FlowNodeData } from "../../../lib/graphUtils";
import { buildHandleId } from "../../../lib/graphUtils";
import { useNodeData } from "../hooks/useNodeData";
import { useHandlePositions } from "../hooks/useHandlePositions";
import {
  NodeCard,
  NodeHeader,
  NodeBody,
  NodeParamRow,
  NodePill,
  NODE_TOKENS,
} from "../ui";

type VaceNodeType = Node<FlowNodeData, "vace">;

const VACE_COLOR = "#a78bfa"; // violet-400
const IMAGE_COLOR = "#f472b6"; // pink-400

export function VaceNode({ id, data, selected }: NodeProps<VaceNodeType>) {
  const { updateData } = useNodeData(id);
  const sliderRef = useRef<HTMLDivElement>(null);

  const contextScale =
    typeof data.vaceContextScale === "number" ? data.vaceContextScale : 1.0;
  const min = 0;
  const max = 2;
  const step = 0.01;

  const clampedValue = Math.min(Math.max(contextScale, min), max);
  const pct = max > min ? ((clampedValue - min) / (max - min)) * 100 : 0;

  const setValueFromMouse = useCallback(
    (clientX: number) => {
      if (!sliderRef.current) return;
      const rect = sliderRef.current.getBoundingClientRect();
      let ratio = (clientX - rect.left) / rect.width;
      ratio = Math.min(Math.max(ratio, 0), 1);
      let newVal = min + ratio * (max - min);
      newVal = min + Math.round((newVal - min) / step) * step;
      newVal = Math.min(Math.max(newVal, min), max);
      updateData({ vaceContextScale: parseFloat(newVal.toFixed(10)) });
    },
    [updateData]
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

  // Summarize connected images
  const refImage = (data.vaceRefImage as string) || "";
  const firstFrame = (data.vaceFirstFrame as string) || "";
  const lastFrame = (data.vaceLastFrame as string) || "";

  const shortName = (path: string) =>
    path ? path.split(/[/\\]/).pop() || path : "—";

  // Measure handle positions when image state changes
  const { setRowRef, rowPositions } = useHandlePositions([
    refImage,
    firstFrame,
    lastFrame,
  ]);

  return (
    <NodeCard selected={selected} autoMinHeight>
      <NodeHeader
        title={data.customTitle || "VACE"}
        dotColor="bg-violet-400"
        onTitleChange={newTitle => updateData({ customTitle: newTitle })}
      />
      <NodeBody withGap>
        {/* Context Scale slider */}
        <div className="flex flex-col gap-1">
          <p className={`${NODE_TOKENS.labelText} text-[10px]`}>
            Context Scale
          </p>
          <div
            ref={sliderRef}
            className="relative w-full h-5 rounded-full cursor-pointer select-none nodrag"
            style={{
              background: "#1b1a1a",
              border: "1px solid rgba(119,119,119,0.15)",
            }}
            onPointerDown={handlePointerDown}
          >
            <div
              className="absolute left-0 top-0 h-full rounded-full pointer-events-none"
              style={{
                width: `${pct}%`,
                background: VACE_COLOR,
                opacity: 0.35,
              }}
            />
            <div
              className="absolute top-1/2 -translate-y-1/2 w-3 h-3 rounded-full pointer-events-none"
              style={{
                left: `calc(${pct}% - 6px)`,
                background: VACE_COLOR,
                boxShadow: `0 0 4px ${VACE_COLOR}`,
              }}
            />
          </div>
          <div className="flex justify-center">
            <span className={NODE_TOKENS.primaryText}>
              {clampedValue.toFixed(2)}
            </span>
          </div>
        </div>

        {/* Image input indicators — each wrapped with rowRef for handle alignment */}
        <div ref={setRowRef("ref_image")}>
          <NodeParamRow label="Ref Image">
            <NodePill className={refImage ? "" : "opacity-40"}>
              {shortName(refImage)}
            </NodePill>
          </NodeParamRow>
        </div>
        <div ref={setRowRef("first_frame")}>
          <NodeParamRow label="First Frame">
            <NodePill className={firstFrame ? "" : "opacity-40"}>
              {shortName(firstFrame)}
            </NodePill>
          </NodeParamRow>
        </div>
        <div ref={setRowRef("last_frame")}>
          <NodeParamRow label="Last Frame">
            <NodePill className={lastFrame ? "" : "opacity-40"}>
              {shortName(lastFrame)}
            </NodePill>
          </NodeParamRow>
        </div>
      </NodeBody>

      {/* Image input handles (left) — positioned by measured row offsets */}
      <Handle
        type="target"
        position={Position.Left}
        id={buildHandleId("param", "ref_image")}
        className="!w-2 !h-2 !border-0"
        style={{
          top: rowPositions["ref_image"] ?? 0,
          left: 8,
          backgroundColor: IMAGE_COLOR,
        }}
      />
      <Handle
        type="target"
        position={Position.Left}
        id={buildHandleId("param", "first_frame")}
        className="!w-2 !h-2 !border-0"
        style={{
          top: rowPositions["first_frame"] ?? 0,
          left: 8,
          backgroundColor: IMAGE_COLOR,
        }}
      />
      <Handle
        type="target"
        position={Position.Left}
        id={buildHandleId("param", "last_frame")}
        className="!w-2 !h-2 !border-0"
        style={{
          top: rowPositions["last_frame"] ?? 0,
          left: 8,
          backgroundColor: IMAGE_COLOR,
        }}
      />

      {/* VACE compound output handle (right) */}
      <Handle
        type="source"
        position={Position.Right}
        id={buildHandleId("param", "__vace")}
        className="!w-2 !h-2 !border-0"
        style={{ top: "50%", right: 8, backgroundColor: VACE_COLOR }}
      />
    </NodeCard>
  );
}
