import React, { useState } from "react";
import { PromptTimeline } from "./PromptTimeline";
import { useTimelinePlayback } from "../hooks/useTimelinePlayback";

interface PromptInputWithTimelineProps {
  className?: string;
  currentPrompt: string;
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
  onPromptSubmit,
  disabled = false,
  isStreaming = false,
  isVideoPaused = false,
  timelineRef,
}: PromptInputWithTimelineProps) {
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
    const activePrompt = prompts.find(
      prompt => currentTime >= prompt.startTime && currentTime <= prompt.endTime
    );
    return activePrompt ? activePrompt.text : "";
  }, [prompts, currentTime]);

  // Expose the function to parent via ref
  React.useImperativeHandle(timelineRef, () => ({
    getCurrentTimelinePrompt,
  }));

  return (
    <div className={`space-y-3 ${className}`}>
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
    </div>
  );
}
