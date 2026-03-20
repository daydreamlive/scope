import type { ReactNode } from "react";
import { NODE_TOKENS } from "./tokens";
import { useIsZoomedOut } from "../hooks/node/useIsZoomedOut";
import { useNodeFlags } from "../hooks/node/useNodeFlags";

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
  const zoomedOut = useIsZoomedOut();
  const { locked } = useNodeFlags();

  // Opacity only (pointer-events handled by NodeCard)
  const style: React.CSSProperties | undefined = zoomedOut
    ? { opacity: 0.07, pointerEvents: "none" }
    : locked
      ? { opacity: 0.55 }
      : undefined;

  return (
    <div
      className={`${withGap ? NODE_TOKENS.bodyWithGap : NODE_TOKENS.body} flex-1 min-h-0 ${className} transition-opacity duration-200`}
      style={style}
    >
      {children}
    </div>
  );
}
