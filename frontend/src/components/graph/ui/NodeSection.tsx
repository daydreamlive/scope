import type { ReactNode } from "react";
import { NODE_TOKENS } from "./tokens";

interface NodeSectionProps {
  title: string;
  children: ReactNode;
  className?: string;
}

export function NodeSection({
  title,
  children,
  className = "",
}: NodeSectionProps) {
  return (
    <div className={className}>
      <h4 className={NODE_TOKENS.sectionTitle}>{title}</h4>
      <div className="flex flex-col gap-3">{children}</div>
    </div>
  );
}

