import type { ReactNode } from "react";
import { NODE_TOKENS } from "./tokens";

interface NodeBodyProps {
  children: ReactNode;
  withGap?: boolean;
  className?: string;
}

export function NodeBody({
  children,
  withGap = false,
  className = "",
}: NodeBodyProps) {
  return (
    <div
      className={`${withGap ? NODE_TOKENS.bodyWithGap : NODE_TOKENS.body} flex-1 min-h-0 ${className}`}
    >
      {children}
    </div>
  );
}
