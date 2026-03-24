import type { ReactNode } from "react";
import { NODE_TOKENS } from "./tokens";

interface NodePillProps {
  children: ReactNode;
  className?: string;
}

export function NodePill({ children, className = "" }: NodePillProps) {
  return (
    <div
      className={`${NODE_TOKENS.pill} w-[110px] flex items-center justify-center ${className}`}
    >
      <p className={`${NODE_TOKENS.primaryText} leading-[1.55]`}>{children}</p>
    </div>
  );
}
