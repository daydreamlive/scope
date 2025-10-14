import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "./tooltip";
import { cn } from "../../lib/utils";

interface LabelWithTooltipProps {
  label: string;
  tooltip?: string;
  className?: string;
  htmlFor?: string;
}

/**
 * A reusable label component with optional tooltip.
 * When a tooltip is provided, it shows on hover over the label.
 */
export function LabelWithTooltip({
  label,
  tooltip,
  className,
  htmlFor,
}: LabelWithTooltipProps) {
  if (!tooltip) {
    return (
      <label htmlFor={htmlFor} className={className}>
        {label}
      </label>
    );
  }

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <label htmlFor={htmlFor} className={cn("cursor-help", className)}>
            {label}
          </label>
        </TooltipTrigger>
        <TooltipContent className="max-w-xs">
          <p className="text-xs">{tooltip}</p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}
