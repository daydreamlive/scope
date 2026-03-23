import { BaseEdge, getBezierPath } from "@xyflow/react";
import type { EdgeProps } from "@xyflow/react";

export function CustomEdge({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  style,
  markerEnd,
  data,
}: EdgeProps) {
  const [edgePath, labelX, labelY] = getBezierPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
  });

  const edgeColor = (style?.stroke as string) || "#9ca3af";

  return (
    <>
      <BaseEdge id={id} path={edgePath} markerEnd={markerEnd} style={style} />
      <g
        className="edge-delete-dot"
        transform={`translate(${labelX}, ${labelY})`}
      >
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
