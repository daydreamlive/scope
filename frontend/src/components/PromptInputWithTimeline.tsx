import React, { useState } from "react";
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
  isVideoPaused?: boolean;
  timelineRef?: React.RefObject<{
    getCurrentTimelinePrompt: () => string;
  } | null>;
}

export function PromptInputWithTimeline({
  className = "",
  currentPrompt,
  onPromptChange,
  onPromptSubmit,
  disabled = false,
  isStreaming = false,
  isVideoPaused = false,
  timelineRef,
}: PromptInputWithTimelineProps) {
  const [mode, setMode] = useState<"text" | "timeline">("text");
  const [isRecording, setIsRecording] = useState(false);

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
    isVideoPaused,
  });

  const handlePromptSubmit = (prompt: string) => {
    if (onPromptSubmit) {
      onPromptSubmit(prompt);
    }
  };

  const handleRecordingToggle = () => {
    const newRecordingState = !isRecording;
    setIsRecording(newRecordingState);

    if (newRecordingState) {
      // Auto-start timeline playback when recording begins
      if (!isPlaying) {
        togglePlayback();
      }
    } else {
      // Pause timeline when recording stops
      if (isPlaying) {
        togglePlayback();
      }
    }
  };

  // Custom play/pause handler that respects recording state
  const handlePlayPause = () => {
    // If recording is active, don't allow manual pause
    if (isRecording && isPlaying) {
      return; // Disabled during recording
    }
    togglePlayback();
  };

  // Expose current timeline prompt to parent
  const getCurrentTimelinePrompt = React.useCallback(() => {
    if (mode === "timeline") {
      const activePrompt = prompts.find(
        prompt =>
          currentTime >= prompt.startTime && currentTime <= prompt.endTime
      );
      return activePrompt ? activePrompt.text : "";
    }
    return "";
  }, [mode, prompts, currentTime]);

  // Expose the function to parent via ref
  React.useImperativeHandle(timelineRef, () => ({
    getCurrentTimelinePrompt,
  }));

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
          onPlayPause={handlePlayPause}
          onTimeChange={updateCurrentTime}
          isRecording={isRecording}
          onRecordingToggle={handleRecordingToggle}
          onPromptSubmit={onPromptSubmit}
          initialPrompt={currentPrompt}
        />
      )}
    </div>
  );
}
