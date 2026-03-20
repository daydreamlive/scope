import { useCallback, useRef } from "react";
import { Handle, Position } from "@xyflow/react";
import type { NodeProps, Node } from "@xyflow/react";
import type { FlowNodeData } from "../../../lib/graphUtils";
import { buildHandleId } from "../../../lib/graphUtils";
import { useNodeData } from "../hooks/node/useNodeData";
import { useNodeCollapse } from "../hooks/node/useNodeCollapse";
import { useHandlePositions } from "../hooks/node/useHandlePositions";
import {
  NodeCard,
  NodeHeader,
  NodeBody,
  NodeParamRow,
  NodePill,
  NODE_TOKENS,
  collapsedHandleStyle,
} from "../ui";

type VaceNodeType = Node<FlowNodeData, "vace">;

const VACE_COLOR = "#a78bfa"; // violet-400
const IMAGE_COLOR = "#f472b6"; // pink-400
const VIDEO_COLOR = "#38bdf8"; // sky-400

export function VaceNode({ id, data, selected }: NodeProps<VaceNodeType>) {
  const { updateData } = useNodeData(id);
  const { collapsed, toggleCollapse } = useNodeCollapse();
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

  // Connected inputs
  const refImage = (data.vaceRefImage as string) || "";
  const firstFrame = (data.vaceFirstFrame as string) || "";
  const lastFrame = (data.vaceLastFrame as string) || "";
  const vaceVideo = (data.vaceVideo as string) || "";

  const shortName = (path: string) =>
    path ? path.split(/[/\\]/).pop() || path : "—";

  // Mutual exclusion: images OR video
  const hasImages = !!(refImage || firstFrame || lastFrame);
  const hasVideo = !!vaceVideo;
  const imagesDimmed = hasVideo; // dim images when video is connected
  const videoDimmed = hasImages; // dim video when images are connected

  // Measure handle positions
  const { setRowRef, rowPositions } = useHandlePositions([
    refImage,
    firstFrame,
    lastFrame,
    vaceVideo,
  ]);

  return (
    <NodeCard
      selected={selected}
      autoMinHeight={!collapsed}
      collapsed={collapsed}
    >
      <NodeHeader
        title={data.customTitle || "VACE"}
        onTitleChange={newTitle => updateData({ customTitle: newTitle })}
        collapsed={collapsed}
        onCollapseToggle={toggleCollapse}
      />
      {!collapsed && (
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

          {/* Video input indicator */}
          <div
            ref={setRowRef("video")}
            className={`transition-opacity ${videoDimmed ? "opacity-30 pointer-events-none" : ""}`}
          >
            <NodeParamRow label="Video">
              <NodePill className={vaceVideo ? "" : "opacity-40"}>
                {shortName(vaceVideo)}
              </NodePill>
            </NodeParamRow>
          </div>

          {/* Image input indicators — each wrapped with rowRef for handle alignment */}
          <div
            ref={setRowRef("ref_image")}
            className={`transition-opacity ${imagesDimmed ? "opacity-30 pointer-events-none" : ""}`}
          >
            <NodeParamRow label="Ref Image">
              <NodePill className={refImage ? "" : "opacity-40"}>
                {shortName(refImage)}
              </NodePill>
            </NodeParamRow>
          </div>
          <div
            ref={setRowRef("first_frame")}
            className={`transition-opacity ${imagesDimmed ? "opacity-30 pointer-events-none" : ""}`}
          >
            <NodeParamRow label="First Frame">
              <NodePill className={firstFrame ? "" : "opacity-40"}>
                {shortName(firstFrame)}
              </NodePill>
            </NodeParamRow>
          </div>
          <div
            ref={setRowRef("last_frame")}
            className={`transition-opacity ${imagesDimmed ? "opacity-30 pointer-events-none" : ""}`}
          >
            <NodeParamRow label="Last Frame">
              <NodePill className={lastFrame ? "" : "opacity-40"}>
                {shortName(lastFrame)}
              </NodePill>
            </NodeParamRow>
          </div>
        </NodeBody>
      )}

      {/* Video input handle (left) */}
      <Handle
        type="target"
        position={Position.Left}
        id={buildHandleId("param", "video")}
        className="!w-2.5 !h-2.5 !border-0"
        style={
          collapsed
            ? collapsedHandleStyle("left")
            : {
                top: rowPositions["video"] ?? 0,
                left: 0,
                backgroundColor: VIDEO_COLOR,
                opacity: videoDimmed ? 0.3 : 1,
              }
        }
      />

      {/* Image input handles (left) — positioned by measured row offsets */}
      <Handle
        type="target"
        position={Position.Left}
        id={buildHandleId("param", "ref_image")}
        className={
          collapsed
            ? "!w-0 !h-0 !border-0 !min-w-0 !min-h-0"
            : "!w-2.5 !h-2.5 !border-0"
        }
        style={
          collapsed
            ? { ...collapsedHandleStyle("left"), opacity: 0 }
            : {
                top: rowPositions["ref_image"] ?? 0,
                left: 0,
                backgroundColor: IMAGE_COLOR,
                opacity: imagesDimmed ? 0.3 : 1,
              }
        }
      />
      <Handle
        type="target"
        position={Position.Left}
        id={buildHandleId("param", "first_frame")}
        className={
          collapsed
            ? "!w-0 !h-0 !border-0 !min-w-0 !min-h-0"
            : "!w-2.5 !h-2.5 !border-0"
        }
        style={
          collapsed
            ? { ...collapsedHandleStyle("left"), opacity: 0 }
            : {
                top: rowPositions["first_frame"] ?? 0,
                left: 0,
                backgroundColor: IMAGE_COLOR,
                opacity: imagesDimmed ? 0.3 : 1,
              }
        }
      />
      <Handle
        type="target"
        position={Position.Left}
        id={buildHandleId("param", "last_frame")}
        className={
          collapsed
            ? "!w-0 !h-0 !border-0 !min-w-0 !min-h-0"
            : "!w-2.5 !h-2.5 !border-0"
        }
        style={
          collapsed
            ? { ...collapsedHandleStyle("left"), opacity: 0 }
            : {
                top: rowPositions["last_frame"] ?? 0,
                left: 0,
                backgroundColor: IMAGE_COLOR,
                opacity: imagesDimmed ? 0.3 : 1,
              }
        }
      />

      {/* VACE compound output handle (right) */}
      <Handle
        type="source"
        position={Position.Right}
        id={buildHandleId("param", "__vace")}
        className="!w-2.5 !h-2.5 !border-0"
        style={
          collapsed
            ? collapsedHandleStyle("right")
            : { top: "50%", right: 0, backgroundColor: VACE_COLOR }
        }
      />
    </NodeCard>
  );
}
