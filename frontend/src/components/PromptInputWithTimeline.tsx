import React, { useState } from "react";
import { PromptInput } from "./PromptInput";
import { PromptTimeline } from "./PromptTimeline";
import { Card, CardContent } from "./ui/card";
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
  const [showTimeline, setShowTimeline] = useState(false);
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
    if (showTimeline) {
      const activePrompt = prompts.find(
        prompt =>
          currentTime >= prompt.startTime && currentTime <= prompt.endTime
      );
      return activePrompt ? activePrompt.text : "";
    }
    return "";
  }, [showTimeline, prompts, currentTime]);

  // Expose the function to parent via ref
  React.useImperativeHandle(timelineRef, () => ({
    getCurrentTimelinePrompt,
  }));

  return (
    <div className={`space-y-3 ${className}`}>
      {/* Left-side checkbox to show/hide timeline */}
      <Card className="bg-transparent border-none shadow-none">
        <CardContent className="p-0">
          <label className="flex items-center gap-2 cursor-pointer select-none">
            <input
              type="checkbox"
              className="h-4 w-4"
              checked={showTimeline}
              onChange={e => setShowTimeline(e.target.checked)}
              disabled={disabled}
            />
            <span className="text-sm text-foreground">Show timeline</span>
          </label>
        </CardContent>
      </Card>

      {/* Content based on checkbox */}
      {!showTimeline ? (
        <PromptInput
          prompts={[{ text: currentPrompt, weight: 100 }]}
          onPromptsChange={prompts => {
            if (prompts.length > 0 && onPromptChange) {
              onPromptChange(prompts[0].text);
            }
          }}
          onPromptsSubmit={prompts => {
            if (prompts.length > 0 && onPromptSubmit) {
              onPromptSubmit(prompts[0].text);
            }
          }}
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
