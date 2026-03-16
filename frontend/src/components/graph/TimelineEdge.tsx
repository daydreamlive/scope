/**
 * TimelineEdge
 *
 * Custom edge for connections originating from timeline trigger handles
 * (Position.Top). The path goes:
 *   1. Straight up from the source handle
 *   2. Rounded 90° corner turning toward the target
 *   3. Horizontal segment arriving directly at the target handle
 *
 * The edge is rendered as a dashed line to visually distinguish trigger
 * connections from regular data connections.
 */

import { BaseEdge } from "@xyflow/react";
import type { EdgeProps } from "@xyflow/react";

/** Radius of the rounded 90° corner. */
const R = 8;
/** Minimum vertical gap above the source before the turn. */
const MIN_RISE = 20;

/**
 * Build an SVG path that goes straight up from `(sx, sy)`, makes a single
 * rounded 90° turn, and arrives horizontally at `(tx, ty)`.
 */
function buildTimelinePath(
  sx: number,
  sy: number,
  tx: number,
  ty: number
): { path: string; labelX: number; labelY: number } {
  const dx = tx - sx;
  const dirX = dx >= 0 ? 1 : -1; // horizontal direction toward target
  const absDx = Math.abs(dx);

  // The turn happens at the target's Y level so the edge arrives horizontally.
  // If the target is below the source we still rise a bit to make the turn.
  const turnY = Math.min(sy - MIN_RISE, ty);

  // Clamp radius so it doesn't exceed available space
  const r = Math.min(R, absDx, Math.abs(sy - turnY));

  let path: string;

  if (absDx < 1) {
    // Target is directly above — straight vertical line
    path = `M ${sx} ${sy} L ${tx} ${ty}`;
  } else {
    // Up from source to the turn level, single 90° corner, horizontal to target
    path = [
      `M ${sx} ${sy}`,
      // Vertical up to just before the corner
      `L ${sx} ${turnY + r}`,
      // Rounded 90° corner
      `Q ${sx} ${turnY} ${sx + dirX * r} ${turnY}`,
      // Horizontal into the target handle
      `L ${tx} ${ty}`,
    ].join(" ");
  }

  // Label position: midpoint of the path
  const labelX = (sx + tx) / 2;
  const labelY = turnY;

  return { path, labelX, labelY };
}

export function TimelineEdge({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  style,
  markerEnd,
  data,
}: EdgeProps) {
  const { path: edgePath, labelX, labelY } = buildTimelinePath(
    sourceX,
    sourceY,
    targetX,
    targetY
  );

  const edgeColor = (style?.stroke as string) || "#9ca3af";

  const dashedStyle = {
    ...style,
    strokeDasharray: "6 3",
  };

  return (
    <>
      <BaseEdge id={id} path={edgePath} markerEnd={markerEnd} style={dashedStyle} />
      {/* Clickable delete-dot anchored to the midpoint of the edge */}
      <g transform={`translate(${labelX}, ${labelY})`}>
        {/* Invisible larger hit area for easier clicking */}
        <circle
          r={12}
          fill="transparent"
          style={{ pointerEvents: "all", cursor: "pointer" }}
          onClick={e => {
            e.stopPropagation();
            if (data && typeof data.onDelete === "function") {
              data.onDelete(id);
            }
          }}
        />
        {/* Visible dot */}
        <circle
          r={5}
          fill={edgeColor}
          stroke="rgba(0, 0, 0, 0.4)"
          strokeWidth={1}
          style={{ pointerEvents: "none" }}
        />
      </g>
    </>
  );
}
