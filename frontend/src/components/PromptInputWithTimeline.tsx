import React, { useState } from "react";
import { PromptTimeline, type TimelinePrompt } from "./PromptTimeline";
import { useTimelinePlayback } from "../hooks/useTimelinePlayback";
import type { PromptItem } from "../lib/api";

interface PromptInputWithTimelineProps {
  className?: string;
  currentPrompt: string;
  currentPromptItems?: PromptItem[];
  onPromptSubmit?: (prompt: string) => void;
  disabled?: boolean;
  isStreaming?: boolean;
  isVideoPaused?: boolean;
  timelineRef?: React.RefObject<{
    getCurrentTimelinePrompt: () => string;
  } | null>;
  selectedPrompt?: TimelinePrompt | null;
  onPromptEdit?: (prompt: TimelinePrompt | null) => void;
  onRecordingStateChange?: (isRecording: boolean) => void;
  onRecordingPromptSubmit?: (prompts: PromptItem[]) => void;
}

export function PromptInputWithTimeline({
  className = "",
  currentPrompt,
  currentPromptItems = [],
  onPromptSubmit,
  disabled = false,
  isStreaming = false,
  isVideoPaused = false,
  timelineRef,
  selectedPrompt: _selectedPrompt = null,
  onPromptEdit,
  onRecordingStateChange,
  onRecordingPromptSubmit,
}: PromptInputWithTimelineProps) {
  const [isRecording, setIsRecording] = useState(false);
  const [selectedPromptId, setSelectedPromptId] = useState<string | null>(null);

  // Generate random colors for prompt boxes, ensuring adjacent boxes have different colors
  const generateRandomColor = (excludeColors: string[] = []) => {
    const colors = [
      "#FF6B6B",
      "#4ECDC4",
      "#45B7D1",
      "#96CEB4",
      "#FFEAA7",
      "#DDA0DD",
      "#98D8C8",
      "#F7DC6F",
      "#BB8FCE",
      "#85C1E9",
      "#F8C471",
      "#82E0AA",
      "#F1948A",
      "#85C1E9",
      "#D7BDE2",
    ];

    // Filter out excluded colors
    const availableColors = colors.filter(
      color => !excludeColors.includes(color)
    );

    // If no colors available, return a random one
    if (availableColors.length === 0) {
      return colors[Math.floor(Math.random() * colors.length)];
    }

    return availableColors[Math.floor(Math.random() * availableColors.length)];
  };
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

  const buildLivePromptFromCurrent = (
    start: number,
    end: number
  ): TimelinePrompt => {
    if (currentPromptItems && currentPromptItems.length > 0) {
      return {
        id: `live-${Date.now()}`,
        text: currentPromptItems.map(p => p.text).join(", "),
        startTime: start,
        endTime: end,
        isLive: true,
        prompts: currentPromptItems.map(p => ({
          text: p.text,
          weight: p.weight,
        })),
      };
    }
    return {
      id: `live-${Date.now()}`,
      text: currentPrompt || "Recording...",
      startTime: start,
      endTime: end,
      isLive: true,
    };
  };

  const handleRecordingToggle = () => {
    const newRecordingState = !isRecording;
    setIsRecording(newRecordingState);

    // Notify parent component of recording state change
    if (onRecordingStateChange) {
      onRecordingStateChange(newRecordingState);
    }

    if (newRecordingState) {
      // Deselect any currently selected timeline prompt so panel input is enabled
      setSelectedPromptId(null);
      onPromptEdit?.(null);

      // Start recording - implement smart recording behavior
      const sortedPrompts = [...prompts].sort(
        (a, b) => a.startTime - b.startTime
      );
      const lastPrompt = sortedPrompts[sortedPrompts.length - 1];
      const isAtEnd = !lastPrompt || currentTime >= lastPrompt.endTime;
      const isAtBeginning = currentTime <= 0;

      if (isAtBeginning) {
        // Remove all blocks and start fresh
        setPrompts([]);
        const livePrompt = buildLivePromptFromCurrent(0, 0);
        setPrompts([livePrompt]);
      } else if (isAtEnd) {
        // Start from the end of the last block
        const start = lastPrompt ? lastPrompt.endTime : currentTime;
        const livePrompt = buildLivePromptFromCurrent(start, start);
        setPrompts(prevPrompts => [...prevPrompts, livePrompt]);
      } else {
        // In the middle - remove all blocks after current time and make current block live
        const filteredPrompts = sortedPrompts.filter(
          p => p.endTime <= currentTime
        );

        // Find the block that contains the current time
        const currentBlock = sortedPrompts.find(
          p => currentTime >= p.startTime && currentTime <= p.endTime
        );

        if (currentBlock) {
          // Make the current block live
          const livePrompt: TimelinePrompt = {
            ...currentBlock,
            endTime: currentTime,
            isLive: true,
          };
          setPrompts([
            ...filteredPrompts.filter(p => p.id !== currentBlock.id),
            livePrompt,
          ]);
        } else {
          // No current block, create new live block from current blend
          const livePrompt = buildLivePromptFromCurrent(
            currentTime,
            currentTime
          );
          setPrompts([...filteredPrompts, livePrompt]);
        }
      }

      // Auto-start timeline playback when recording begins
      if (!isPlaying) {
        togglePlayback();
      }
    } else {
      // Stop recording - complete the current live box
      setPrompts(prevPrompts =>
        prevPrompts.map(p =>
          p.isLive
            ? {
                ...p,
                endTime: currentTime,
                isLive: false,
                color: generateRandomColor(),
              }
            : p
        )
      );

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

  // Handle prompt selection
  const handlePromptSelect = React.useCallback((promptId: string | null) => {
    setSelectedPromptId(promptId);
  }, []);

  // Handle prompt editing
  const handlePromptEdit = React.useCallback(
    (prompt: TimelinePrompt) => {
      if (onPromptEdit) {
        onPromptEdit(prompt);
      }
    },
    [onPromptEdit]
  );

  // Handle recording prompt submission
  const handleRecordingPromptSubmit = React.useCallback(
    (promptItems: PromptItem[]) => {
      if (!promptItems.length || !promptItems.some(p => p.text.trim())) return;

      // Complete the current live box and start a new one
      setPrompts(prevPrompts => {
        const updatedPrompts = prevPrompts.map(p =>
          p.isLive
            ? {
                ...p,
                endTime: currentTime,
                isLive: false,
                color: generateRandomColor(),
              }
            : p
        );

        // Create new live box with blend information
        const newLivePrompt: TimelinePrompt = {
          id: `live-${Date.now()}`,
          text: promptItems.map(p => p.text).join(", "), // Combined text for display
          startTime: currentTime,
          endTime: currentTime, // Will be updated as time progresses
          isLive: true,
          prompts: promptItems.map(p => ({ text: p.text, weight: p.weight })), // Store blend info
        };

        return [...updatedPrompts, newLivePrompt];
      });

      // Do NOT call onPromptSubmit here; it would reset blends to a single prompt
    },
    [currentTime]
  );

  // Handle prompt updates from the editor
  const handlePromptUpdate = React.useCallback(
    (updatedPrompt: TimelinePrompt) => {
      setPrompts(prevPrompts =>
        prevPrompts.map(p => (p.id === updatedPrompt.id ? updatedPrompt : p))
      );
    },
    []
  );

  // Update live box end time as current time progresses
  React.useEffect(() => {
    if (isRecording) {
      setPrompts(prevPrompts =>
        prevPrompts.map(p => (p.isLive ? { ...p, endTime: currentTime } : p))
      );
    }
  }, [currentTime, isRecording]);

  // Expose the recording prompt submit function to parent
  React.useImperativeHandle(timelineRef, () => ({
    getCurrentTimelinePrompt,
    submitRecordingPrompt: handleRecordingPromptSubmit,
    updatePrompt: handlePromptUpdate,
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
        selectedPromptId={selectedPromptId}
        onPromptSelect={handlePromptSelect}
        onPromptEdit={handlePromptEdit}
        onRecordingPromptSubmit={onRecordingPromptSubmit}
      />
    </div>
  );
}
