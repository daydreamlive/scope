import { type ReactNode, useState, useRef, useEffect } from "react";
import { NodeResizer } from "@xyflow/react";
import { NODE_TOKENS } from "./tokens";
import { useNodeFlags } from "../hooks/useNodeFlags";

interface NodeCardProps {
  children: ReactNode;
  selected?: boolean;
  className?: string;
  // Measures content height and enforces as minHeight on resize
  autoMinHeight?: boolean;
  // Override default min width (240px)
  minWidth?: number;
  // Override default min height (60px)
  minHeight?: number;
}

export function NodeCard({
  children,
  selected,
  className = "",
  autoMinHeight = false,
  minWidth = 240,
  minHeight: minHeightProp = 60,
}: NodeCardProps) {
  const measureRef = useRef<HTMLDivElement>(null);
  const [minH, setMinH] = useState(60);
  const { locked } = useNodeFlags();

  useEffect(() => {
    if (!autoMinHeight || !measureRef.current) return;

    const el = measureRef.current;

    const measure = () => {
      // Use scrollHeight
      const h = el.scrollHeight;
      setMinH(prev => (Math.abs(h - prev) > 2 ? h : prev));
    };

    // Measure
    measure();

    // Watch for size changes
    const ro = new ResizeObserver(measure);
    ro.observe(el);

    return () => ro.disconnect();
  }, [autoMinHeight]);

  // Block pointer-events when locked (NodeHeader overrides to keep buttons clickable)
  const lockStyle: React.CSSProperties | undefined = locked
    ? { pointerEvents: "none" }
    : undefined;

  return (
    <div
      className={`group ${NODE_TOKENS.card} ${selected ? NODE_TOKENS.cardSelected : ""} ${className}`}
    >
      <NodeResizer
        isVisible={!!selected}
        minWidth={minWidth}
        minHeight={
          autoMinHeight ? Math.max(minHeightProp, minH) : minHeightProp
        }
        lineClassName="!border-transparent"
        handleClassName="!w-2 !h-2 !bg-transparent !border !border-blue-400/20 hover:!border-blue-400/40 !rounded-sm"
      />
      {autoMinHeight ? (
        <div
          ref={measureRef}
          className="flex flex-col w-full"
          style={lockStyle}
        >
          {children}
        </div>
      ) : (
        <div className="flex flex-col w-full h-full" style={lockStyle}>
          {children}
        </div>
      )}
    </div>
  );
}
