import type { ReactNode } from "react";
import { NODE_TOKENS } from "./tokens";

interface NodePillProps {
  children: ReactNode;
  className?: string;
  title?: string;
}

export function NodePill({ children, className = "", title }: NodePillProps) {
  return (
    <div
      className={`${NODE_TOKENS.pill} w-full flex items-center justify-center ${className}`}
      title={title}
    >
      <p
        className={`${NODE_TOKENS.primaryText} leading-[1.55] truncate max-w-full`}
      >
        {children}
      </p>
    </div>
  );
}
