import { useCallback } from "react";
import { Handle, Position } from "@xyflow/react";
import type { NodeProps, Node } from "@xyflow/react";
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
  NodePill,
  NODE_TOKENS,
  collapsedHandleStyle,
} from "../ui";
import {
  COLOR_VACE as VACE_COLOR,
  COLOR_IMAGE as IMAGE_COLOR,
} from "../nodeColors";

type VaceNodeType = Node<FlowNodeData, "vace">;

export function VaceNode({ id, data, selected }: NodeProps<VaceNodeType>) {
  const { updateData } = useNodeData(id);
  const { collapsed, toggleCollapse } = useNodeCollapse();

  const contextScale =
    typeof data.vaceContextScale === "number" ? data.vaceContextScale : 1.0;
  const contextScaleIn = useConnectedNumber(id, "context_scale", contextScale);
  const effectiveScale = contextScaleIn.value;

  const onSliderChange = useCallback(
    (v: number) => updateData({ vaceContextScale: v }),
    [updateData]
  );

  const { sliderRef, clampedValue, pct, handlePointerDown } = useSlider({
    min: 0,
    max: 2,
    step: 0.01,
    value: effectiveScale,
    onChange: onSliderChange,
  });

  // Connected inputs
  const refImage = (data.vaceRefImage as string) || "";
  const firstFrame = (data.vaceFirstFrame as string) || "";
  const lastFrame = (data.vaceLastFrame as string) || "";

  const shortName = (path: string) =>
    path ? path.split(/[/\\]/).pop() || path : "—";

  // Measure handle positions
  const { setRowRef, rowPositions } = useHandlePositions([
    refImage,
    firstFrame,
    lastFrame,
    contextScaleIn.connected,
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
          <div ref={setRowRef("context_scale")} className="flex flex-col gap-1">
            <p className={`${NODE_TOKENS.labelText} text-[10px]`}>
              Context Scale
            </p>
            {contextScaleIn.connected ? (
              <div className="flex justify-center">
                <NodePill className="opacity-50">
                  {effectiveScale.toFixed(2)}
                </NodePill>
              </div>
            ) : (
              <>
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
              </>
            )}
          </div>

          {/* Image input indicators — each wrapped with rowRef for handle alignment */}
          <div ref={setRowRef("ref_image")} className="transition-opacity">
            <NodeParamRow label="Ref Image">
              <NodePill className={refImage ? "" : "opacity-40"}>
                {shortName(refImage)}
              </NodePill>
            </NodeParamRow>
          </div>
          <div ref={setRowRef("first_frame")} className="transition-opacity">
            <NodeParamRow label="First Frame">
              <NodePill className={firstFrame ? "" : "opacity-40"}>
                {shortName(firstFrame)}
              </NodePill>
            </NodeParamRow>
          </div>
          <div ref={setRowRef("last_frame")} className="transition-opacity">
            <NodeParamRow label="Last Frame">
              <NodePill className={lastFrame ? "" : "opacity-40"}>
                {shortName(lastFrame)}
              </NodePill>
            </NodeParamRow>
          </div>
        </NodeBody>
      )}

      {/* Context scale input handle */}
      <Handle
        type="target"
        position={Position.Left}
        id={buildHandleId("param", "context_scale")}
        className={
          collapsed
            ? "!w-0 !h-0 !border-0 !min-w-0 !min-h-0"
            : "!w-2.5 !h-2.5 !border-0"
        }
        style={
          collapsed
            ? { ...collapsedHandleStyle("left"), opacity: 0 }
            : {
                top: rowPositions["context_scale"] ?? 50,
                left: 0,
                backgroundColor: "#38bdf8",
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
