import { Handle, Position, useEdges, useNodes } from "@xyflow/react";
import type { NodeProps, Node } from "@xyflow/react";
import { useEffect, useRef, useCallback } from "react";
import type { FlowNodeData } from "../../../lib/graphUtils";
import { buildHandleId } from "../../../lib/graphUtils";
import { getNumberFromNode } from "../utils/getValueFromNode";
import { useNodeData } from "../hooks/node/useNodeData";
import { useNodeCollapse } from "../hooks/node/useNodeCollapse";
import { useHandlePositions } from "../hooks/node/useHandlePositions";
import {
  NodeCard,
  NodeHeader,
  NodeBody,
  NodeParamRow,
  NodePillInput,
  NodePillSelect,
  NodePill,
  NODE_TOKENS,
  collapsedHandleStyle,
} from "../ui";

type TriggerActionNodeType = Node<FlowNodeData, "trigger_action">;

// ── Easing functions ──
function easeLinear(t: number): number {
  return t;
}
function easeIn(t: number): number {
  return t * t;
}
function easeOut(t: number): number {
  return t * (2 - t);
}
function easeInOut(t: number): number {
  return t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t;
}

const EASING_FNS: Record<string, (t: number) => number> = {
  linear: easeLinear,
  ease_in: easeIn,
  ease_out: easeOut,
  ease_in_out: easeInOut,
};

/** Interpolate y from a sorted array of {x,y} breakpoints (both normalized 0-1) */
function interpolateCurve(
  points: Array<{ x: number; y: number }>,
  t: number
): number {
  if (points.length === 0) return t;
  if (points.length === 1) return points[0].y;
  if (t <= points[0].x) return points[0].y;
  if (t >= points[points.length - 1].x) return points[points.length - 1].y;
  // Find the two surrounding points
  for (let i = 0; i < points.length - 1; i++) {
    if (t >= points[i].x && t <= points[i + 1].x) {
      const segLen = points[i + 1].x - points[i].x;
      const localT = segLen > 0 ? (t - points[i].x) / segLen : 0;
      return points[i].y + localT * (points[i + 1].y - points[i].y);
    }
  }
  return points[points.length - 1].y;
}

const ACTION_TYPE_OPTIONS = [
  { value: "set_number", label: "Set Number" },
  { value: "set_string", label: "Set String" },
  { value: "set_bool", label: "Set Bool" },
  { value: "animate_number", label: "Animate Number" },
  { value: "toggle_bool", label: "Toggle Bool" },
  { value: "cycle_strings", label: "Cycle Strings" },
];

const CURVE_OPTIONS = [
  { value: "linear", label: "Linear" },
  { value: "ease_in", label: "Ease In" },
  { value: "ease_out", label: "Ease Out" },
  { value: "ease_in_out", label: "Ease In/Out" },
];

/** Infer output type from action type */
function getOutputType(
  actionType: string
): "number" | "string" | "boolean" {
  switch (actionType) {
    case "set_string":
    case "cycle_strings":
      return "string";
    case "set_bool":
    case "toggle_bool":
      return "boolean";
    default:
      return "number";
  }
}

/** Color for the output handle based on type */
function getOutputColor(outputType: string): string {
  switch (outputType) {
    case "string":
      return "#fbbf24"; // amber-400
    case "boolean":
      return "#34d399"; // emerald-400
    default:
      return "#38bdf8"; // sky-400
  }
}

export function TriggerActionNode({
  id,
  data,
  selected,
}: NodeProps<TriggerActionNodeType>) {
  const { updateData } = useNodeData(id);
  const { collapsed, toggleCollapse } = useNodeCollapse();

  const actionType = data.triggerActionType || "set_number";
  const setValue = data.triggerSetValue;
  const animateFrom = data.triggerAnimateFrom ?? 0;
  const animateTo = data.triggerAnimateTo ?? 1;
  const animateDuration = data.triggerAnimateDuration ?? 1;
  const animateCurve = data.triggerAnimateCurve || "linear";
  const toggleState = data.triggerToggleState ?? false;
  const cycleItems = data.triggerCycleItems || ["item1", "item2", "item3"];
  const cycleIndex = data.triggerCycleIndex ?? 0;
  const currentValue = data.currentValue;

  const edges = useEdges();
  const allNodes = useNodes() as Node<FlowNodeData>[];

  const lastTriggerValueRef = useRef<number>(0);
  const animationFrameRef = useRef<number | undefined>(undefined);
  const animStartRef = useRef<number>(0);

  const outputType = getOutputType(actionType);
  const outputColor = getOutputColor(outputType);

  // ── Detect connected curve node ──
  const curveHandleId = buildHandleId("param", "curve");
  const curveEdge = edges.find(
    e => e.target === id && e.targetHandle === curveHandleId
  );
  const curveSourceNode = curveEdge
    ? allNodes.find(n => n.id === curveEdge.source)
    : undefined;
  const connectedCurvePoints =
    curveSourceNode?.data?.curvePoints as
      | Array<{ x: number; y: number }>
      | undefined;
  const hasCurve =
    !!connectedCurvePoints && connectedCurvePoints.length >= 2;
  const connectedCurveMin = (curveSourceNode?.data?.curveMin as number) ?? 0;
  const connectedCurveMax = (curveSourceNode?.data?.curveMax as number) ?? 1;

  // ── Initialize currentValue based on action type ──
  useEffect(() => {
    if (currentValue !== undefined) return;
    let initial: number | string;
    switch (actionType) {
      case "set_number":
        initial = Number(setValue) || 0;
        break;
      case "set_string":
        initial = String(setValue ?? "");
        break;
      case "set_bool":
        initial = setValue ? 1 : 0;
        break;
      case "animate_number":
        initial = animateFrom;
        break;
      case "toggle_bool":
        initial = toggleState ? 1 : 0;
        break;
      case "cycle_strings":
        initial = cycleItems[0] || "";
        break;
      default:
        initial = 0;
    }
    updateData({ currentValue: initial });
  }, [actionType]); // eslint-disable-line react-hooks/exhaustive-deps

  // ── Trigger detection: watch rising edge on trigger input ──
  useEffect(() => {
    const triggerHandleId = buildHandleId("param", "trigger");
    const triggerEdge = edges.find(
      e => e.target === id && e.targetHandle === triggerHandleId
    );
    if (!triggerEdge) return;

    const sourceNode = allNodes.find(n => n.id === triggerEdge.source);
    if (!sourceNode) return;

    const val =
      getNumberFromNode(sourceNode, triggerEdge.sourceHandle) ?? 0;
    const prev = lastTriggerValueRef.current;
    lastTriggerValueRef.current = val;

    // Rising edge: prev <= 0 and val > 0
    if (!(prev <= 0 && val > 0)) return;

    // Fire the action
    switch (actionType) {
      case "set_number":
        updateData({ currentValue: Number(setValue) || 0 });
        break;
      case "set_string":
        updateData({ currentValue: String(setValue ?? "") });
        break;
      case "set_bool":
        updateData({ currentValue: setValue ? 1 : 0 });
        break;
      case "animate_number":
        startAnimation();
        break;
      case "toggle_bool": {
        const newState = !toggleState;
        updateData({
          triggerToggleState: newState,
          currentValue: newState ? 1 : 0,
        });
        break;
      }
      case "cycle_strings": {
        const nextIndex = (cycleIndex + 1) % cycleItems.length;
        updateData({
          triggerCycleIndex: nextIndex,
          currentValue: cycleItems[nextIndex] || "",
        });
        break;
      }
    }
  }); // intentionally no deps - runs every render

  const startAnimation = useCallback(() => {
    // Cancel any existing animation
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
    }

    animStartRef.current = performance.now();
    const dur = Math.max(0.01, animateDuration) * 1000; // ms

    // Snapshot curve state at animation start (so mid-animation edits don't glitch)
    const useCurve = hasCurve ? connectedCurvePoints! : null;
    const cMin = connectedCurveMin;
    const cMax = connectedCurveMax;
    const easeFn = useCurve ? null : (EASING_FNS[animateCurve] || easeLinear);

    const animate = () => {
      const elapsed = performance.now() - animStartRef.current;
      const progress = Math.min(1, elapsed / dur);

      let value: number;
      if (useCurve) {
        // Curve defines full value trajectory: y (0-1) maps to [curveMin, curveMax]
        const curveY = interpolateCurve(useCurve, progress);
        value = cMin + (cMax - cMin) * curveY;
      } else {
        // Preset easing between From and To
        const easedProgress = easeFn!(progress);
        value = animateFrom + (animateTo - animateFrom) * easedProgress;
      }

      updateData({ currentValue: value });

      if (progress < 1) {
        animationFrameRef.current = requestAnimationFrame(animate);
      } else {
        animationFrameRef.current = undefined;
      }
    };

    animationFrameRef.current = requestAnimationFrame(animate);
  }, [animateFrom, animateTo, animateDuration, animateCurve, updateData, hasCurve, connectedCurvePoints, connectedCurveMin, connectedCurveMax]);

  // Cleanup animation on unmount
  useEffect(() => {
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, []);

  const handleActionTypeChange = useCallback(
    (newType: string) => {
      // Reset currentValue when type changes
      let initial: number | string;
      switch (newType) {
        case "set_number":
        case "animate_number":
        case "set_bool":
        case "toggle_bool":
          initial = 0;
          break;
        case "set_string":
        case "cycle_strings":
          initial = "";
          break;
        default:
          initial = 0;
      }
      updateData({
        triggerActionType: newType as FlowNodeData["triggerActionType"],
        currentValue: initial,
        // Update parameterOutputs so type resolution works
        parameterOutputs: [
          {
            name: "value",
            type: getOutputType(newType),
            defaultValue: initial,
          },
        ],
      });
    },
    [updateData]
  );

  const { setRowRef, rowPositions } = useHandlePositions([
    actionType,
    cycleItems.length,
  ]);

  // ── Display the current value ──
  const displayValue = (() => {
    if (currentValue === undefined || currentValue === null) return "—";
    if (
      (actionType === "set_bool" || actionType === "toggle_bool") &&
      typeof currentValue === "number"
    ) {
      return currentValue ? "true" : "false";
    }
    if (typeof currentValue === "number") return currentValue.toFixed(3);
    const s = String(currentValue);
    return s.length > 20 ? s.slice(0, 20) + "…" : s;
  })();

  return (
    <NodeCard
      selected={selected}
      autoMinHeight={!collapsed}
      collapsed={collapsed}
    >
      <NodeHeader
        title={data.customTitle || "TriggerAction"}
        onTitleChange={newTitle => updateData({ customTitle: newTitle })}
        collapsed={collapsed}
        onCollapseToggle={toggleCollapse}
      />

      {!collapsed && (
        <NodeBody withGap>
          {/* Action type selector */}
          <NodeParamRow label="Action">
            <NodePillSelect
              value={actionType}
              onChange={handleActionTypeChange}
              options={ACTION_TYPE_OPTIONS}
            />
          </NodeParamRow>

          {/* Action-specific controls */}
          {actionType === "set_number" && (
            <NodeParamRow label="Value">
              <NodePillInput
                type="number"
                value={Number(setValue) || 0}
                onChange={v =>
                  updateData({ triggerSetValue: Number(v) })
                }
              />
            </NodeParamRow>
          )}

          {actionType === "set_string" && (
            <NodeParamRow label="Value">
              <NodePillInput
                type="text"
                value={String(setValue ?? "")}
                onChange={v => updateData({ triggerSetValue: String(v) })}
              />
            </NodeParamRow>
          )}

          {actionType === "set_bool" && (
            <NodeParamRow label="Value">
              <label className="flex items-center gap-1 cursor-pointer">
                <input
                  type="checkbox"
                  checked={Boolean(setValue)}
                  onChange={e =>
                    updateData({ triggerSetValue: e.target.checked })
                  }
                  className="w-3 h-3 rounded accent-emerald-500"
                />
                <span className="text-[9px] text-[#fafafa]">
                  {Boolean(setValue) ? "true" : "false"}
                </span>
              </label>
            </NodeParamRow>
          )}

          {actionType === "animate_number" && (
            <>
              {/* From/To only shown when no curve is connected */}
              {!hasCurve && (
                <>
                  <NodeParamRow label="From">
                    <NodePillInput
                      type="number"
                      value={animateFrom}
                      onChange={v =>
                        updateData({ triggerAnimateFrom: Number(v) })
                      }
                    />
                  </NodeParamRow>
                  <NodeParamRow label="To">
                    <NodePillInput
                      type="number"
                      value={animateTo}
                      onChange={v =>
                        updateData({ triggerAnimateTo: Number(v) })
                      }
                    />
                  </NodeParamRow>
                </>
              )}
              <NodeParamRow label="Duration">
                <NodePillInput
                  type="number"
                  value={animateDuration}
                  onChange={v =>
                    updateData({
                      triggerAnimateDuration: Math.max(0.01, Number(v)),
                    })
                  }
                  min={0.01}
                />
              </NodeParamRow>
              <NodeParamRow label="Curve">
                {hasCurve ? (
                  <NodePill className="opacity-75">
                    <span className="text-[9px]" style={{ color: "#f59e0b" }}>
                      Custom ({connectedCurveMin}–{connectedCurveMax})
                    </span>
                  </NodePill>
                ) : (
                  <NodePillSelect
                    value={animateCurve}
                    onChange={v =>
                      updateData({
                        triggerAnimateCurve:
                          v as FlowNodeData["triggerAnimateCurve"],
                      })
                    }
                    options={CURVE_OPTIONS}
                  />
                )}
              </NodeParamRow>
            </>
          )}

          {actionType === "toggle_bool" && (
            <NodeParamRow label="State">
              <NodePill className="opacity-75">
                {toggleState ? "true" : "false"}
              </NodePill>
            </NodeParamRow>
          )}

          {actionType === "cycle_strings" && (
            <>
              <NodeParamRow label="Items">
                <NodePillInput
                  type="text"
                  value={cycleItems.join(", ")}
                  onChange={v => {
                    const items = String(v)
                      .split(",")
                      .map(s => s.trim())
                      .filter(s => s.length > 0);
                    updateData({
                      triggerCycleItems:
                        items.length > 0 ? items : ["item1"],
                    });
                  }}
                />
              </NodeParamRow>
              <NodeParamRow label="Index">
                <NodePill className="opacity-75">
                  {cycleIndex} / {cycleItems.length}
                </NodePill>
              </NodeParamRow>
            </>
          )}

          {/* Current value display */}
          <div ref={setRowRef("value")} className={NODE_TOKENS.paramRow}>
            <span className={NODE_TOKENS.labelText}>Output</span>
            <NodePill className="opacity-75">{displayValue}</NodePill>
          </div>
        </NodeBody>
      )}

      {/* ── Handles ── */}

      {/* Trigger input (left) */}
      <Handle
        type="target"
        position={Position.Left}
        id={buildHandleId("param", "trigger")}
        className="!w-2.5 !h-2.5 !border-0"
        style={
          collapsed
            ? collapsedHandleStyle("left")
            : {
                top: rowPositions["value"] ?? 44,
                left: 0,
                backgroundColor: "#38bdf8",
              }
        }
      />

      {/* Curve input (left, below trigger — only when animate_number) */}
      {actionType === "animate_number" && (
        <Handle
          type="target"
          position={Position.Left}
          id={curveHandleId}
          className="!w-2.5 !h-2.5 !border-0"
          style={
            collapsed
              ? { ...collapsedHandleStyle("left"), opacity: 0 }
              : {
                  top: (rowPositions["value"] ?? 44) + 16,
                  left: 0,
                  backgroundColor: "#f59e0b",
                }
          }
        />
      )}

      {/* Value output (right) */}
      <Handle
        type="source"
        position={Position.Right}
        id={buildHandleId("param", "value")}
        className="!w-2.5 !h-2.5 !border-0"
        style={
          collapsed
            ? collapsedHandleStyle("right")
            : {
                top: rowPositions["value"] ?? 44,
                right: 0,
                backgroundColor: outputColor,
              }
        }
      />
    </NodeCard>
  );
}
