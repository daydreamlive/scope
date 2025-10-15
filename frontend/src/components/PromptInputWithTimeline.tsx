import { useState } from "react";
import { PromptInput } from "./PromptInput";
import { PromptTimeline } from "./PromptTimeline";
import { PromptTimelineToggle } from "./PromptTimelineToggle";
import { useTimelinePlayback } from "../hooks/useTimelinePlayback";

interface PromptInputWithTimelineProps {
  className?: string;
  currentPrompt: string;
  onPromptChange?: (prompt: string) => void;
  onPromptSubmit?: (prompt: string) => void;
  disabled?: boolean;
  isStreaming?: boolean;
}

export function PromptInputWithTimeline({
  className = "",
  currentPrompt,
  onPromptChange,
  onPromptSubmit,
  disabled = false,
  isStreaming = false,
}: PromptInputWithTimelineProps) {
  const [mode, setMode] = useState<"text" | "timeline">("text");

  const {
    prompts,
    setPrompts,
    isPlaying,
    currentTime,
    updateCurrentTime,
    togglePlayback,
  } = useTimelinePlayback({
    onPromptChange: onPromptSubmit,
    isStreaming,
  });

  const handlePromptSubmit = (prompt: string) => {
    if (onPromptSubmit) {
      onPromptSubmit(prompt);
    }
  };

  return (
    <div className={`space-y-3 ${className}`}>
      {/* Toggle */}
      <div className="flex justify-center">
        <PromptTimelineToggle
          mode={mode}
          onModeChange={setMode}
          disabled={disabled}
        />
      </div>

      {/* Content based on mode */}
      {mode === "text" ? (
        <PromptInput
          currentPrompt={currentPrompt}
          onPromptChange={onPromptChange}
          onPromptSubmit={handlePromptSubmit}
          disabled={disabled}
        />
      ) : (
        <PromptTimeline
          prompts={prompts}
          onPromptsChange={setPrompts}
          disabled={disabled}
          isPlaying={isPlaying}
          currentTime={currentTime}
          onPlayPause={togglePlayback}
          onTimeChange={updateCurrentTime}
        />
      )}
    </div>
  );
}
