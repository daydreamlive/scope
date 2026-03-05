import { NODE_TOKENS } from "./tokens";

interface NodeHeaderProps {
  title: string;
  dotColor: string;
  className?: string;
}

export function NodeHeader({
  title,
  dotColor,
  className = "",
}: NodeHeaderProps) {
  return (
    <div className={`${NODE_TOKENS.header} ${className}`}>
      <div className={`w-[10px] h-[10px] rounded-full ${dotColor} shrink-0`} />
      <p className={NODE_TOKENS.headerText}>{title}</p>
    </div>
  );
}

