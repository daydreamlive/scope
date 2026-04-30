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
      <p className={`${NODE_TOKENS.labelText} min-w-0 truncate`} title={label}>
        {label}
      </p>
      <div className="flex min-w-0 justify-end [&>div]:min-w-0 [&>div]:w-full">
        {children}
      </div>
    </div>
  );
}
