import type { ReactNode } from "react";
import { NODE_TOKENS } from "./tokens";

interface NodeParamRowProps {
  label: string;
  children: ReactNode;
  className?: string;
}

export function NodeParamRow({
  label,
  children,
  className = "",
}: NodeParamRowProps) {
  return (
    <div className={`${NODE_TOKENS.paramRow} ${className}`}>
      <p className={`${NODE_TOKENS.labelText} w-[80px] shrink-0 truncate`}>
        {label}
      </p>
      {children}
    </div>
  );
}

