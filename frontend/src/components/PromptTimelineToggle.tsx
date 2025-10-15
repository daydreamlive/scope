import { Button } from "./ui/button";

interface PromptTimelineToggleProps {
  mode: "text" | "timeline";
  onModeChange: (mode: "text" | "timeline") => void;
  disabled?: boolean;
}

export function PromptTimelineToggle({
  mode,
  onModeChange,
  disabled = false,
}: PromptTimelineToggleProps) {
  return (
    <div className="flex items-center bg-card border border-border rounded-lg p-1">
      <Button
        variant={mode === "text" ? "default" : "ghost"}
        size="sm"
        onClick={() => onModeChange("text")}
        disabled={disabled}
        className="flex-1 text-xs"
      >
        Input
      </Button>
      <div className="w-px h-4 bg-border mx-1" />
      <Button
        variant={mode === "timeline" ? "default" : "ghost"}
        size="sm"
        onClick={() => onModeChange("timeline")}
        disabled={disabled}
        className="flex-1 text-xs"
      >
        Timeline
      </Button>
    </div>
  );
}
