import { Handle, Position } from "@xyflow/react";
import type { NodeProps, Node } from "@xyflow/react";
import { useCallback, useRef } from "react";
import type { FlowNodeData } from "../../../lib/graphUtils";
import { buildHandleId } from "../../../lib/graphUtils";
import { useNodeData } from "../hooks/node/useNodeData";
import { useNodeCollapse } from "../hooks/node/useNodeCollapse";
import {
  NodeCard,
  NodeHeader,
  NodeBody,
  NodeParamRow,
  NodePillInput,
  NodePillSelect,
  collapsedHandleStyle,
} from "../ui";

type CurveNodeType = Node<FlowNodeData, "curve">;

type CurvePoint = { x: number; y: number };

const CURVE_PRESETS: Record<string, CurvePoint[]> = {
  linear: [
    { x: 0, y: 0 },
    { x: 1, y: 1 },
  ],
  ease_in: [
    { x: 0, y: 0 },
    { x: 0.4, y: 0.05 },
    { x: 0.7, y: 0.2 },
    { x: 1, y: 1 },
  ],
  ease_out: [
    { x: 0, y: 0 },
    { x: 0.3, y: 0.8 },
    { x: 0.6, y: 0.95 },
    { x: 1, y: 1 },
  ],
  ease_in_out: [
    { x: 0, y: 0 },
    { x: 0.3, y: 0.05 },
    { x: 0.5, y: 0.5 },
    { x: 0.7, y: 0.95 },
    { x: 1, y: 1 },
  ],
  bell: [
    { x: 0, y: 0 },
    { x: 0.25, y: 0.7 },
    { x: 0.5, y: 1 },
    { x: 0.75, y: 0.7 },
    { x: 1, y: 0 },
  ],
  ramp_down: [
    { x: 0, y: 1 },
    { x: 1, y: 0 },
  ],
  dip: [
    { x: 0, y: 1 },
    { x: 0.25, y: 0.3 },
    { x: 0.5, y: 0 },
    { x: 0.75, y: 0.3 },
    { x: 1, y: 1 },
  ],
  step: [
    { x: 0, y: 0 },
    { x: 0.49, y: 0 },
    { x: 0.51, y: 1 },
    { x: 1, y: 1 },
  ],
  zigzag: [
    { x: 0, y: 0 },
    { x: 0.25, y: 1 },
    { x: 0.5, y: 0 },
    { x: 0.75, y: 1 },
    { x: 1, y: 0 },
  ],
  bounce: [
    { x: 0, y: 0 },
    { x: 0.2, y: 0.8 },
    { x: 0.4, y: 0.3 },
    { x: 0.55, y: 0.65 },
    { x: 0.7, y: 0.45 },
    { x: 0.85, y: 0.55 },
    { x: 1, y: 0.5 },
  ],
};

const PRESET_OPTIONS = [
  { value: "linear", label: "Linear" },
  { value: "ease_in", label: "Ease In" },
  { value: "ease_out", label: "Ease Out" },
  { value: "ease_in_out", label: "Ease In/Out" },
  { value: "bell", label: "Bell" },
  { value: "ramp_down", label: "Ramp Down" },
  { value: "dip", label: "Dip" },
  { value: "step", label: "Step" },
  { value: "zigzag", label: "Zigzag" },
  { value: "bounce", label: "Bounce" },
];

const CANVAS_W = 180;
const CANVAS_H = 100;
const POINT_RADIUS = 5;
const HIT_RADIUS = 10; // larger hit area for pointer
const CURVE_COLOR = "#f59e0b"; // amber-400
const FILL_COLOR = "rgba(245, 158, 11, 0.08)";
const GRID_OPACITY = 0.08;
const HEADER_HEIGHT = 28;
const BODY_PAD = 6; // py-1.5 ≈ 6px

export function CurveNode({ id, data, selected }: NodeProps<CurveNodeType>) {
  const { updateData } = useNodeData(id);
  const { collapsed, toggleCollapse } = useNodeCollapse();
  const canvasRef = useRef<HTMLDivElement>(null);
  const draggingRef = useRef<number | null>(null);

  const points: Array<{ x: number; y: number }> = data.curvePoints ?? [
    { x: 0, y: 0 },
    { x: 1, y: 1 },
  ];

  const curveMin = data.curveMin ?? 0;
  const curveMax = data.curveMax ?? 1;

  /** Convert client coords to normalized (0-1) canvas coords (y flipped) */
  const clientToNorm = useCallback(
    (clientX: number, clientY: number): { x: number; y: number } => {
      if (!canvasRef.current) return { x: 0, y: 0 };
      const rect = canvasRef.current.getBoundingClientRect();
      const x = Math.min(Math.max((clientX - rect.left) / rect.width, 0), 1);
      const y = Math.min(
        Math.max(1 - (clientY - rect.top) / rect.height, 0),
        1
      );
      return { x: round6(x), y: round6(y) };
    },
    []
  );

  /** Find the index of the point closest to (cx, cy) within HIT_RADIUS, or -1 */
  const hitTest = useCallback(
    (clientX: number, clientY: number): number => {
      if (!canvasRef.current) return -1;
      const rect = canvasRef.current.getBoundingClientRect();
      let closest = -1;
      let closestDist = Infinity;
      for (let i = 0; i < points.length; i++) {
        const px = rect.left + points[i].x * rect.width;
        const py = rect.top + (1 - points[i].y) * rect.height;
        const dist = Math.hypot(clientX - px, clientY - py);
        if (dist < HIT_RADIUS && dist < closestDist) {
          closest = i;
          closestDist = dist;
        }
      }
      return closest;
    },
    [points]
  );

  const updatePoints = useCallback(
    (newPts: Array<{ x: number; y: number }>) => {
      updateData({ curvePoints: newPts });
    },
    [updateData]
  );

  /** Pointer down: either start dragging an existing point or add a new one */
  const handlePointerDown = useCallback(
    (e: React.PointerEvent) => {
      e.preventDefault();
      e.stopPropagation();

      const idx = hitTest(e.clientX, e.clientY);

      if (idx >= 0) {
        // Start dragging an existing point
        draggingRef.current = idx;
      } else {
        // Add a new point
        const norm = clientToNorm(e.clientX, e.clientY);
        const newPts = [...points, norm].sort((a, b) => a.x - b.x);
        updatePoints(newPts);
        // Find the index of the newly inserted point and start dragging it
        const newIdx = newPts.findIndex(
          (p) => p.x === norm.x && p.y === norm.y
        );
        draggingRef.current = newIdx >= 0 ? newIdx : null;
      }

      const target = e.currentTarget as HTMLElement;
      target.setPointerCapture(e.pointerId);

      const onMove = (ev: PointerEvent) => {
        if (draggingRef.current === null) return;
        const di = draggingRef.current;
        const norm = clientToNorm(ev.clientX, ev.clientY);

        // Get the latest points from the data — but since we're in a closure,
        // we work with the points array we have and update it immutably
        const currentPts = [...points];
        if (di < 0 || di >= currentPts.length) return;

        // First and last points: x is fixed at 0 and 1 respectively
        let newX = norm.x;
        if (di === 0) {
          newX = 0;
        } else if (di === currentPts.length - 1) {
          newX = 1;
        } else {
          // Constrain x between neighbors
          const prevX = currentPts[di - 1].x + 0.001;
          const nextX = currentPts[di + 1].x - 0.001;
          newX = Math.min(Math.max(newX, prevX), nextX);
        }

        currentPts[di] = { x: round6(newX), y: norm.y };
        updatePoints(currentPts);
      };

      const onUp = () => {
        draggingRef.current = null;
        target.removeEventListener("pointermove", onMove);
        target.removeEventListener("pointerup", onUp);
      };

      target.addEventListener("pointermove", onMove);
      target.addEventListener("pointerup", onUp);
    },
    [hitTest, clientToNorm, points, updatePoints]
  );

  /** Double-click: remove point (except first and last) */
  const handleDoubleClick = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault();
      e.stopPropagation();
      const idx = hitTest(e.clientX, e.clientY);
      if (idx > 0 && idx < points.length - 1) {
        const newPts = points.filter((_, i) => i !== idx);
        updatePoints(newPts);
      }
    },
    [hitTest, points, updatePoints]
  );

  // Build the SVG path for the curve
  const buildPath = (): string => {
    if (points.length === 0) return "";
    const cmds: string[] = [];
    for (let i = 0; i < points.length; i++) {
      const px = points[i].x * CANVAS_W;
      const py = (1 - points[i].y) * CANVAS_H;
      if (i === 0) {
        cmds.push(`M ${px} ${py}`);
      } else {
        cmds.push(`L ${px} ${py}`);
      }
    }
    return cmds.join(" ");
  };

  // Build the filled area path (curve + bottom edge)
  const buildFillPath = (): string => {
    if (points.length === 0) return "";
    const curvePath = buildPath();
    // Close by going to bottom-right, bottom-left, then back to start
    const lastX = points[points.length - 1].x * CANVAS_W;
    const firstX = points[0].x * CANVAS_W;
    return `${curvePath} L ${lastX} ${CANVAS_H} L ${firstX} ${CANVAS_H} Z`;
  };

  // Handle position
  const handleY = collapsed
    ? undefined
    : HEADER_HEIGHT + BODY_PAD + CANVAS_H / 2;

  return (
    <NodeCard
      selected={selected}
      autoMinHeight={!collapsed}
      collapsed={collapsed}
      minWidth={CANVAS_W + 16}
    >
      <NodeHeader
        title={data.customTitle || "Curve"}
        onTitleChange={(newTitle) => updateData({ customTitle: newTitle })}
        collapsed={collapsed}
        onCollapseToggle={toggleCollapse}
      />

      {!collapsed && (
        <NodeBody>
          <div className="flex flex-col gap-1">
            {/* Canvas */}
            <div
              ref={canvasRef}
              className="relative rounded-lg cursor-crosshair select-none overflow-hidden"
              style={{
                width: CANVAS_W,
                height: CANVAS_H,
                background: "#1b1a1a",
                border: "1px solid rgba(119,119,119,0.15)",
              }}
              onPointerDown={handlePointerDown}
              onDoubleClick={handleDoubleClick}
            >
              {/* Grid lines */}
              <div
                className="absolute inset-0 pointer-events-none"
                style={{ opacity: GRID_OPACITY }}
              >
                <div className="absolute left-1/4 top-0 bottom-0 w-px bg-white" />
                <div className="absolute left-1/2 top-0 bottom-0 w-px bg-white" />
                <div className="absolute left-3/4 top-0 bottom-0 w-px bg-white" />
                <div className="absolute top-1/4 left-0 right-0 h-px bg-white" />
                <div className="absolute top-1/2 left-0 right-0 h-px bg-white" />
                <div className="absolute top-3/4 left-0 right-0 h-px bg-white" />
              </div>

              {/* SVG overlay for curve + fill + points */}
              <svg
                className="absolute inset-0 pointer-events-none"
                width={CANVAS_W}
                height={CANVAS_H}
                viewBox={`0 0 ${CANVAS_W} ${CANVAS_H}`}
              >
                {/* Filled area under curve */}
                <path d={buildFillPath()} fill={FILL_COLOR} />
                {/* Curve line */}
                <path
                  d={buildPath()}
                  fill="none"
                  stroke={CURVE_COLOR}
                  strokeWidth={1.5}
                  strokeLinejoin="round"
                />
                {/* Breakpoints */}
                {points.map((p, i) => (
                  <circle
                    key={i}
                    cx={p.x * CANVAS_W}
                    cy={(1 - p.y) * CANVAS_H}
                    r={POINT_RADIUS}
                    fill={
                      i === 0 || i === points.length - 1
                        ? "#fafafa"
                        : CURVE_COLOR
                    }
                    stroke="#000"
                    strokeWidth={1}
                  />
                ))}
              </svg>
            </div>

            {/* Point count + hint */}
            <div className="flex justify-between px-1">
              <span className="text-[9px] text-[#8c8c8d]">
                {points.length} point{points.length !== 1 ? "s" : ""}
              </span>
              <span className="text-[9px] text-[#555]">dbl-click to remove</span>
            </div>

            {/* Preset selector */}
            <NodeParamRow label="Preset">
              <NodePillSelect
                value=""
                onChange={v => {
                  const preset = CURVE_PRESETS[v];
                  if (preset) {
                    updateData({ curvePoints: preset.map(p => ({ ...p })) });
                  }
                }}
                options={[
                  { value: "", label: "Load..." },
                  ...PRESET_OPTIONS,
                ]}
              />
            </NodeParamRow>

            {/* Min / Max controls */}
            <NodeParamRow label="Min">
              <NodePillInput
                type="number"
                value={curveMin}
                onChange={v => updateData({ curveMin: Number(v) })}
              />
            </NodeParamRow>
            <NodeParamRow label="Max">
              <NodePillInput
                type="number"
                value={curveMax}
                onChange={v => updateData({ curveMax: Number(v) })}
              />
            </NodeParamRow>
          </div>
        </NodeBody>
      )}

      {/* Output handle (right) */}
      <Handle
        type="source"
        position={Position.Right}
        id={buildHandleId("param", "value")}
        className="!w-2.5 !h-2.5 !border-0"
        style={
          collapsed
            ? collapsedHandleStyle("right")
            : {
                top: handleY,
                right: 0,
                backgroundColor: CURVE_COLOR,
              }
        }
      />
    </NodeCard>
  );
}

function round6(n: number): number {
  return parseFloat(n.toFixed(6));
}
