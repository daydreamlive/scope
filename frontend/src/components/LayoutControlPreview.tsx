import { useEffect, useRef, useState } from "react";

interface LayoutControlPreviewProps {
  /** Current circle position from controller input [x, y] normalized 0-1 */
  position?: [number, number];
  /** Whether controller input is active (pointer locked) */
  isActive?: boolean;
  /** Callback when user clicks to request pointer lock */
  onRequestPointerLock?: () => void;
  /** Width of the preview canvas */
  width?: number;
  /** Height of the preview canvas */
  height?: number;
}

/**
 * Visual preview of the layout control circle position.
 * Shows a white canvas with a black circle contour matching
 * what the LayoutControlPreprocessor generates for VACE.
 */
export function LayoutControlPreview({
  position = [0.5, 0.35],
  isActive = false,
  onRequestPointerLock,
  width = 160,
  height = 120,
}: LayoutControlPreviewProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [localPosition, setLocalPosition] =
    useState<[number, number]>(position);

  // Update local position when prop changes
  useEffect(() => {
    setLocalPosition(position);
  }, [position]);

  // Draw the circle on canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Clear with white background
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, width, height);

    // Calculate circle position and radius
    const [x, y] = localPosition;
    const px = x * width;
    const py = y * height;
    const radius = Math.min(width, height) * 0.15; // ~15% of smallest dimension

    // Draw black circle contour
    ctx.beginPath();
    ctx.arc(px, py, radius, 0, Math.PI * 2);
    ctx.strokeStyle = "#000000";
    ctx.lineWidth = 2;
    ctx.stroke();

    // Draw center dot
    ctx.beginPath();
    ctx.arc(px, py, 3, 0, Math.PI * 2);
    ctx.fillStyle = "#000000";
    ctx.fill();
  }, [localPosition, width, height]);

  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-center justify-between">
        <span className="text-xs text-muted-foreground">Preview</span>
        <span className="text-xs text-muted-foreground">
          {isActive ? (
            <span className="text-green-500">● Active</span>
          ) : (
            <span className="text-yellow-500">○ Click video to control</span>
          )}
        </span>
      </div>
      <div
        className="relative border border-border rounded overflow-hidden cursor-pointer"
        style={{ width, height }}
        onClick={onRequestPointerLock}
        title={
          isActive
            ? "Controller active"
            : "Click video output to enable WASD control"
        }
      >
        <canvas
          ref={canvasRef}
          width={width}
          height={height}
          className="block"
        />
        {!isActive && (
          <div className="absolute inset-0 bg-black/20 flex items-center justify-center">
            <span className="text-xs text-white bg-black/50 px-2 py-1 rounded">
              WASD to move
            </span>
          </div>
        )}
      </div>
      <div className="text-xs text-muted-foreground text-center">
        Position: ({(localPosition[0] * 100).toFixed(0)}%,{" "}
        {(localPosition[1] * 100).toFixed(0)}%)
      </div>
    </div>
  );
}
