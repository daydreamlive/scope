import { useEffect, useRef, useState, useCallback } from "react";

interface UseLayoutControlStreamOptions {
  /** Target width for the canvas (should match pipeline resolution) */
  width: number;
  /** Target height for the canvas (should match pipeline resolution) */
  height: number;
  /** Whether layout control is active (creates the stream) */
  enabled: boolean;
  /** Whether to capture keyboard input (only when streaming) */
  captureInput: boolean;
  /** Frame rate for the stream */
  fps?: number;
  /** Circle radius as fraction of smallest dimension (default 0.1 = 10%) */
  radiusFraction?: number;
  /** Movement speed per frame for WASD keys (0.0-1.0 normalized) */
  moveSpeed?: number;
}

/**
 * Hook that generates a MediaStream from a layout control canvas.
 *
 * Renders a white background with a black circle contour that can be
 * controlled via WASD keys. The canvas is rendered at full resolution
 * and captured as a MediaStream for sending to the backend.
 *
 * This ensures the frontend-rendered frames are exactly what the backend
 * uses for VACE conditioning, eliminating any sync issues.
 */
export function useLayoutControlStream({
  width,
  height,
  enabled,
  captureInput,
  fps = 30,
  radiusFraction = 0.1,
  moveSpeed = 0.004,
}: UseLayoutControlStreamOptions) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const pressedKeysRef = useRef<Set<string>>(new Set());
  const positionRef = useRef<[number, number]>([0.5, 0.35]);

  // Stream as state so component re-renders when it's created
  const [stream, setStream] = useState<MediaStream | null>(null);

  // Circle position state (normalized 0.0-1.0)
  const [position, setPosition] = useState<[number, number]>([0.5, 0.35]);

  // Calculate radius in pixels based on smallest dimension
  const radius = Math.round(Math.min(width, height) * radiusFraction);

  // Keep positionRef in sync with state
  useEffect(() => {
    positionRef.current = position;
  }, [position]);

  // Create canvas and stream when enabled
  useEffect(() => {
    if (!enabled) {
      // Clean up when disabled
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
        setStream(null);
      }
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = null;
      }
      canvasRef.current = null;
      return;
    }

    // Create offscreen canvas at target resolution
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    canvasRef.current = canvas;

    // Capture stream from canvas
    const newStream = canvas.captureStream(fps);
    setStream(newStream);

    return () => {
      newStream.getTracks().forEach(track => track.stop());
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [enabled, width, height, fps]); // eslint-disable-line react-hooks/exhaustive-deps

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

  // Set up keyboard listeners only when captureInput is true (i.e., when streaming)
  useEffect(() => {
    if (!captureInput) {
      // Clear any pressed keys when input capture is disabled
      pressedKeysRef.current.clear();
      return;
    }

    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("keyup", handleKeyUp);

    const keysRef = pressedKeysRef;
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("keyup", handleKeyUp);
      keysRef.current.clear();
    };
  }, [captureInput, handleKeyDown, handleKeyUp]);

  // Draw circle contour on canvas (no fill, just outline)
  const drawCircleContour = useCallback(
    (
      ctx: CanvasRenderingContext2D,
      cx: number,
      cy: number,
      r: number,
      thickness: number = 3
    ) => {
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.strokeStyle = "#000000";
      ctx.lineWidth = thickness;
      ctx.stroke();
    },
    []
  );

  // Animation loop for rendering and position updates
  useEffect(() => {
    if (!enabled || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let lastTime = performance.now();

    const animate = (currentTime: number) => {
      const deltaTime = (currentTime - lastTime) / 16.67; // Normalize to ~60fps
      lastTime = currentTime;

      // Update position based on pressed keys
      const keys = pressedKeysRef.current;
      if (keys.size > 0) {
        let [x, y] = positionRef.current;
        const speed = moveSpeed * deltaTime;

        if (keys.has("KeyW") || keys.has("ArrowUp")) y -= speed;
        if (keys.has("KeyS") || keys.has("ArrowDown")) y += speed;
        if (keys.has("KeyA") || keys.has("ArrowLeft")) x -= speed;
        if (keys.has("KeyD") || keys.has("ArrowRight")) x += speed;

        // Clamp to valid range (leave margin for circle)
        const margin = 0.1;
        x = Math.max(margin, Math.min(1.0 - margin, x));
        y = Math.max(margin, Math.min(1.0 - margin, y));

        const newPos: [number, number] = [x, y];
        positionRef.current = newPos;
        setPosition(newPos);
      }

      // Draw frame using current position from ref (avoids stale closure)
      const [posX, posY] = positionRef.current;

      // White background
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, width, height);

      // Calculate pixel position
      const px = Math.round(posX * width);
      const py = Math.round(posY * height);

      // Clamp to keep circle fully visible
      const clampedPx = Math.max(radius, Math.min(width - radius, px));
      const clampedPy = Math.max(radius, Math.min(height - radius, py));

      // Draw black circle contour
      drawCircleContour(ctx, clampedPx, clampedPy, radius, 3);

      animationFrameRef.current = requestAnimationFrame(animate);
    };

    animationFrameRef.current = requestAnimationFrame(animate);

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [enabled, width, height, radius, moveSpeed, drawCircleContour]);

  // Method to update position externally (for syncing with preview)
  const setPositionExternal = useCallback((pos: [number, number]) => {
    positionRef.current = pos;
    setPosition(pos);
  }, []);

  return {
    /** The MediaStream from the canvas (can be used as video input) */
    stream,
    /** Current circle position [x, y] normalized 0-1 */
    position,
    /** Set position externally (for syncing with preview component) */
    setPosition: setPositionExternal,
    /** Reference to the canvas element (for preview rendering) */
    canvas: canvasRef.current,
  };
}
