import { useEffect, useRef, useState, useCallback } from "react";

interface LayoutControlPreviewProps {
  /** Current circle position [x, y] normalized 0-1 */
  position?: [number, number];
  /** Callback when position changes (for syncing with parent) */
  onPositionChange?: (position: [number, number]) => void;
  /** Whether streaming is active (shows different status) */
  isStreaming?: boolean;
  /** Width of the preview canvas */
  width?: number;
  /** Height of the preview canvas */
  height?: number;
}

/**
 * Interactive layout control preview.
 * Click to focus, then use WASD/arrows to move the circle.
 * Works independently of streaming state.
 */
export function LayoutControlPreview({
  position,
  onPositionChange,
  isStreaming = false,
  width = 160,
  height = 120,
}: LayoutControlPreviewProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [localPosition, setLocalPosition] = useState<[number, number]>(
    position || [0.5, 0.35]
  );
  const [isFocused, setIsFocused] = useState(false);
  const pressedKeysRef = useRef<Set<string>>(new Set());
  const animationFrameRef = useRef<number | null>(null);

  // Sync with external position when provided
  useEffect(() => {
    if (position) {
      setLocalPosition(position);
    }
  }, [position]);

  // Notify parent of position changes
  useEffect(() => {
    onPositionChange?.(localPosition);
  }, [localPosition, onPositionChange]);

  // Handle keyboard input
  const handleKeyDown = useCallback((e: KeyboardEvent) => {
    const validKeys = new Set([
      "KeyW",
      "KeyA",
      "KeyS",
      "KeyD",
      "ArrowUp",
      "ArrowDown",
      "ArrowLeft",
      "ArrowRight",
    ]);
    if (validKeys.has(e.code)) {
      e.preventDefault();
      pressedKeysRef.current.add(e.code);
    }
  }, []);

  const handleKeyUp = useCallback((e: KeyboardEvent) => {
    pressedKeysRef.current.delete(e.code);
  }, []);

  // Animation loop for smooth movement
  useEffect(() => {
    if (!isFocused) {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = null;
      }
      return;
    }

    const moveSpeed = 0.02;
    let lastTime = performance.now();

    const animate = (currentTime: number) => {
      const deltaTime = (currentTime - lastTime) / 16.67; // Normalize to ~60fps
      lastTime = currentTime;

      const keys = pressedKeysRef.current;
      if (keys.size > 0) {
        setLocalPosition(prev => {
          let [x, y] = prev;
          const speed = moveSpeed * deltaTime;

          if (keys.has("KeyW") || keys.has("ArrowUp")) y -= speed;
          if (keys.has("KeyS") || keys.has("ArrowDown")) y += speed;
          if (keys.has("KeyA") || keys.has("ArrowLeft")) x -= speed;
          if (keys.has("KeyD") || keys.has("ArrowRight")) x += speed;

          x = Math.max(0.1, Math.min(0.9, x));
          y = Math.max(0.1, Math.min(0.9, y));

          return [x, y];
        });
      }

      animationFrameRef.current = requestAnimationFrame(animate);
    };

    animationFrameRef.current = requestAnimationFrame(animate);

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [isFocused]);

  // Set up keyboard listeners when focused
  useEffect(() => {
    if (isFocused) {
      window.addEventListener("keydown", handleKeyDown);
      window.addEventListener("keyup", handleKeyUp);
    }
    const keysRef = pressedKeysRef;
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("keyup", handleKeyUp);
      keysRef.current.clear();
    };
  }, [isFocused, handleKeyDown, handleKeyUp]);

  // Handle focus/blur
  const handleClick = () => {
    setIsFocused(true);
    containerRef.current?.focus();
  };

  const handleBlur = () => {
    setIsFocused(false);
    pressedKeysRef.current.clear();
  };

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
        <span className="text-xs text-muted-foreground">Layout Control</span>
        <span className="text-xs">
          {isFocused ? (
            <span className="text-green-500">● Active</span>
          ) : isStreaming ? (
            <span className="text-blue-500">○ Click to control</span>
          ) : (
            <span className="text-muted-foreground">○ Click to try</span>
          )}
        </span>
      </div>
      <div
        ref={containerRef}
        className={`relative border rounded overflow-hidden cursor-pointer outline-none ${
          isFocused ? "border-green-500 ring-1 ring-green-500" : "border-border"
        }`}
        style={{ width, height }}
        tabIndex={0}
        onClick={handleClick}
        onBlur={handleBlur}
        title="Click to control with WASD"
      >
        <canvas
          ref={canvasRef}
          width={width}
          height={height}
          className="block"
        />
        {!isFocused && (
          <div className="absolute inset-0 bg-black/10 flex items-center justify-center">
            <span className="text-xs text-white bg-black/50 px-2 py-1 rounded">
              Click, then WASD
            </span>
          </div>
        )}
      </div>
      <div className="text-xs text-muted-foreground text-center">
        ({(localPosition[0] * 100).toFixed(0)}%,{" "}
        {(localPosition[1] * 100).toFixed(0)}%)
        {isStreaming && (
          <span className="text-green-500 ml-1">● Streaming</span>
        )}
      </div>
    </div>
  );
}
