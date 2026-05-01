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

  const flashing = data?.flashing === true;
  const edgeColor = flashing
    ? "#ffffff"
    : (style?.stroke as string) || "#6b7280";

  const edgeStyle = flashing
    ? {
        ...style,
        stroke: "#ffffff",
        filter: "drop-shadow(0 0 4px rgba(255,255,255,0.6))",
        transition: "stroke 150ms, filter 150ms",
      }
    : {
        ...style,
        transition: "stroke 150ms, filter 150ms",
      };

  return (
    <>
      <BaseEdge
        id={id}
        path={edgePath}
        markerEnd={markerEnd}
        style={edgeStyle}
      />
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
          r={4}
          fill={edgeColor}
          stroke="rgba(0, 0, 0, 0.5)"
          strokeWidth={1.5}
          style={{ pointerEvents: "none" }}
        />
      </g>
    </>
  );
}
