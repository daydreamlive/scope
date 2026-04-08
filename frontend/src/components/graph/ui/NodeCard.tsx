import { type ReactNode, useState, useRef, useEffect } from "react";
import { NodeResizer } from "@xyflow/react";
import { NODE_TOKENS } from "./tokens";
import { useNodeFlags } from "../hooks/node/useNodeFlags";

interface NodeCardProps {
  children: ReactNode;
  selected?: boolean;
  className?: string;
  /** Enforce min height from content. */
  autoMinHeight?: boolean;
  minWidth?: number;
  minHeight?: number;
  /** Compact pill (no resizer). */
  collapsed?: boolean;
}

export function NodeCard({
  children,
  selected,
  className = "",
  autoMinHeight = true,
  minWidth = 240,
  minHeight: minHeightProp = 60,
  collapsed = false,
}: NodeCardProps) {
  const measureRef = useRef<HTMLDivElement>(null);
  const [minH, setMinH] = useState(60);
  const { locked } = useNodeFlags();

  useEffect(() => {
    if (!autoMinHeight || !measureRef.current) return;

    const el = measureRef.current;

    const measure = () => {
      const h = el.scrollHeight;
      setMinH(prev => (Math.abs(h - prev) > 2 ? h : prev));
    };

    measure();

    const ro = new ResizeObserver(() => {
      requestAnimationFrame(measure);
    });
    ro.observe(el);

    return () => ro.disconnect();
  }, [autoMinHeight]);

  // Locked: block pointer-events
  const lockStyle: React.CSSProperties | undefined = locked
    ? { pointerEvents: "none" }
    : undefined;

  /* Collapsed pill */
  if (collapsed) {
    return (
      <div
        className={`group bg-[#1e1e1e] border border-[rgba(255,255,255,0.06)] rounded-full shadow-[0_1px_4px_rgba(0,0,0,0.25)] relative flex flex-col ${
          selected ? NODE_TOKENS.cardSelected : ""
        } ${className}`}
      >
        <div className="flex flex-col w-full" style={lockStyle}>
          {children}
        </div>
      </div>
    );
  }

  /* Normal card */
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
        handleClassName="!w-2 !h-2 !bg-transparent !border !border-blue-400/30 hover:!border-blue-400/60 !rounded-sm"
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
