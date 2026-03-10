import { type ReactNode, useState, useRef, useEffect } from "react";
import { NodeResizer } from "@xyflow/react";
import { NODE_TOKENS } from "./tokens";

interface NodeCardProps {
  children: ReactNode;
  selected?: boolean;
  className?: string;
  /** When true, measures content height and enforces it as minHeight on resize */
  autoMinHeight?: boolean;
  /** Override the default minimum width (240px) for the resize handles */
  minWidth?: number;
  /** Override the default minimum height (60px) for the resize handles */
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

  useEffect(() => {
    if (!autoMinHeight || !measureRef.current) return;

    const el = measureRef.current;

    const measure = () => {
      // Use scrollHeight for natural content height
      const h = el.scrollHeight;
      setMinH(prev => (Math.abs(h - prev) > 2 ? h : prev));
    };

    // Measure
    measure();

    // Watch for size changes (ResizeObserver avoids infinite loops)
    const ro = new ResizeObserver(measure);
    ro.observe(el);

    return () => ro.disconnect();
  }, [autoMinHeight]);

  return (
    <div
      className={`${NODE_TOKENS.card} ${selected ? NODE_TOKENS.cardSelected : ""} ${className}`}
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
        <div ref={measureRef} className="flex flex-col w-full">
          {children}
        </div>
      ) : (
        children
      )}
    </div>
  );
}
