import { type ReactNode, useState, useRef, useEffect } from "react";
import { NodeResizer } from "@xyflow/react";
import { NODE_TOKENS } from "./tokens";

interface NodeCardProps {
  children: ReactNode;
  selected?: boolean;
  className?: string;
  /** When true, measures content height and enforces it as minHeight on resize */
  autoMinHeight?: boolean;
}

export function NodeCard({
  children,
  selected,
  className = "",
  autoMinHeight = false,
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
        minWidth={240}
        minHeight={autoMinHeight ? Math.max(60, minH) : 60}
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
