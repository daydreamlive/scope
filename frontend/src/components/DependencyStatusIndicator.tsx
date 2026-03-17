import { CheckCircle2, Download, Loader2 } from "lucide-react";
import { Button } from "./ui/button";

interface DependencyStatusIndicatorProps {
  status: string | undefined;
  activeStatus: string;
  doneLabel: string;
  activeLabel: string;
  idleLabel: string;
  onAction: () => void;
}

export function DependencyStatusIndicator({
  status,
  activeStatus,
  doneLabel,
  activeLabel,
  idleLabel,
  onAction,
}: DependencyStatusIndicatorProps) {
  if (status === "done") {
    return (
      <span className="text-xs text-green-500 flex items-center gap-1">
        <CheckCircle2 className="h-3 w-3" />
        {doneLabel}
      </span>
    );
  }

  if (status === activeStatus) {
    return (
      <span className="text-xs text-muted-foreground flex items-center gap-1">
        <Loader2 className="h-3 w-3 animate-spin" />
        {activeLabel}
      </span>
    );
  }

  return (
    <Button
      variant="ghost"
      size="sm"
      className="h-6 text-xs px-2"
      onClick={onAction}
    >
      <Download className="h-3 w-3 mr-1" />
      {status === "error" ? "Retry" : idleLabel}
    </Button>
  );
}
