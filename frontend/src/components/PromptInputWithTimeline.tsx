import React, { useState, useCallback, useEffect } from "react";
import { PromptTimeline, type TimelinePrompt } from "./PromptTimeline";
import { useTimelinePlayback } from "../hooks/useTimelinePlayback";
import type { PromptItem } from "../lib/api";
import type { SettingsState } from "../types";

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
  onLiveStateChange?: (isLive: boolean) => void;
  onLivePromptSubmit?: (prompts: PromptItem[]) => void;
  onDisconnect?: () => void;
  onStartStream?: () => void;
  onVideoPlayPauseToggle?: () => void;
  isCollapsed?: boolean;
  onCollapseToggle?: (collapsed: boolean) => void;
  externalSelectedPromptId?: string | null; // New prop to control selection externally
  // New props for settings export/import
  settings?: SettingsState;
  onSettingsImport?: (settings: Partial<SettingsState>) => void;
  // New prop to expose play/pause handler
  onPlayPauseRef?: React.RefObject<(() => Promise<void>) | null>;
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
  onLiveStateChange,
  onLivePromptSubmit,
  onDisconnect,
  onStartStream,
  onVideoPlayPauseToggle,
  isCollapsed = false,
  onCollapseToggle,
  externalSelectedPromptId = null,
  settings,
  onSettingsImport,
  onPlayPauseRef,
}: PromptInputWithTimelineProps) {
  const [isLive, setIsLive] = useState(false);
  const [selectedPromptId, setSelectedPromptId] = useState<string | null>(null);
  const [hasStartedPlayback, setHasStartedPlayback] = useState(false);
  const [scrollToTimeFn, setScrollToTimeFn] = useState<
    ((time: number) => void) | null
  >(null);

  // Sync external selected prompt ID with internal state
  useEffect(() => {
    if (externalSelectedPromptId !== undefined) {
      setSelectedPromptId(externalSelectedPromptId);
    }
  }, [externalSelectedPromptId]);

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
    resetPlayback,
    startPlayback,
    pausePlayback,
  } = useTimelinePlayback({
    onPromptChange: onPromptSubmit,
    isStreaming,
    isVideoPaused,
  });

  // Compute actual playing state - timeline is playing AND video is not paused
  const isActuallyPlaying = isPlaying && !isVideoPaused;

  // Enhanced rewind handler that completes current box, resets to beginning, and preserves play/pause state
  const handleRewind = useCallback(() => {
    // Complete current live box if live
    if (isLive) {
      setIsLive(false);
      if (onLiveStateChange) {
        onLiveStateChange(false);
      }

      // Complete the current live box
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
    }

    // Reset time to 0
    updateCurrentTime(0);

    // Find the first prompt in the timeline
    const sortedPrompts = [...prompts].sort(
      (a, b) => a.startTime - b.startTime
    );
    const firstPrompt = sortedPrompts.find(p => !p.isLive);

    if (firstPrompt) {
      // Update the current prompt text box to show the first prompt
      if (onPromptSubmit) {
        onPromptSubmit(firstPrompt.text);
      }
    }

    // Preserve play/pause state: if was playing, start playing from beginning; if paused, stay paused
    if (isActuallyPlaying) {
      // If currently playing, restart playback from the beginning
      pausePlayback(); // Pause first
      // Use a longer timeout to ensure currentTime state has been updated
      setTimeout(() => {
        // Double-check that currentTime is 0 before starting playback
        if (currentTime === 0) {
          startPlayback();
        } else {
          // If currentTime is not 0 yet, force it to 0 and then start playback
          updateCurrentTime(0);
          setTimeout(() => startPlayback(), 10);
        }
      }, 10);
    }
    // If currently paused, do nothing - stay paused at the beginning
  }, [
    prompts,
    updateCurrentTime,
    onPromptSubmit,
    isLive,
    onLiveStateChange,
    currentTime,
    isActuallyPlaying,
    pausePlayback,
    startPlayback,
    setPrompts,
  ]);

  // Enhanced disconnect handler that stops stream, pauses timeline, and rewinds playhead
  const handleEnhancedDisconnect = useCallback(() => {
    // 1. Stop the stream
    if (onDisconnect) {
      onDisconnect();
    }

    // 2. Pause the timeline (same as clicking Pause button)
    if (isActuallyPlaying) {
      togglePlayback();
    }

    // 3. Rewind the playhead to the beginning (same as clicking Rewind)
    // Complete current live box if live
    if (isLive) {
      setIsLive(false);
      if (onLiveStateChange) {
        onLiveStateChange(false);
      }

      // Complete the current live box
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
    }

    // Reset time to 0
    updateCurrentTime(0);

    // Find the first prompt in the timeline
    const sortedPrompts = [...prompts].sort(
      (a, b) => a.startTime - b.startTime
    );
    const firstPrompt = sortedPrompts.find(p => !p.isLive);

    if (firstPrompt) {
      // Update the current prompt text box to show the first prompt
      if (onPromptSubmit) {
        onPromptSubmit(firstPrompt.text);
      }
    }

    // Note: Disconnect always leaves the timeline paused at the beginning
    // (no additional play/pause logic needed since we already paused above)
  }, [
    onDisconnect,
    isActuallyPlaying,
    togglePlayback,
    isLive,
    onLiveStateChange,
    setPrompts,
    currentTime,
    updateCurrentTime,
    prompts,
    onPromptSubmit,
  ]);

  // Reset hasStartedPlayback when stream stops
  React.useEffect(() => {
    if (!isStreaming) {
      setHasStartedPlayback(false);
    }
  }, [isStreaming]);

  const buildLivePromptFromCurrent = useCallback(
    (start: number, end: number): TimelinePrompt => {
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
        text: currentPrompt || "Live...",
        startTime: start,
        endTime: end,
        isLive: true,
      };
    },
    [currentPromptItems, currentPrompt]
  );

  // Custom play/pause handler that implements new behavior
  const handlePlayPause = useCallback(async () => {
    // If not streaming, start the stream first (first time only)
    if (!isStreaming && onStartStream) {
      await onStartStream();
      // Wait a bit for the stream to initialize
      await new Promise(resolve => setTimeout(resolve, 1000));
    }

    // Check if we're at the end of the timeline
    const sortedPrompts = [...prompts].sort(
      (a, b) => a.startTime - b.startTime
    );
    const lastPrompt = sortedPrompts[sortedPrompts.length - 1];
    const isAtEnd = !lastPrompt || currentTime >= lastPrompt.endTime;

    if (!isActuallyPlaying) {
      // Starting playback

      // Deselect any selected prompt when starting playback
      if (selectedPromptId) {
        setSelectedPromptId(null);
        if (onPromptEdit) {
          onPromptEdit(null);
        }
      }

      if (isAtEnd) {
        // At the end - start live mode with current prompt
        setIsLive(true);
        if (onLiveStateChange) {
          onLiveStateChange(true);
        }

        // Instead of creating a new live box, extend the last prompt as live
        // This matches the automatic behavior in useTimelinePlayback
        const sortedNonLivePrompts = [...prompts]
          .filter(p => !p.isLive)
          .sort((a, b) => a.endTime - b.endTime);

        if (sortedNonLivePrompts.length === 0) {
          // No existing prompts, create a new live box
          if (!isStreaming && onStartStream) {
            await onStartStream();
            // Wait a bit for the stream to initialize
            await new Promise(resolve => setTimeout(resolve, 1000));
            const livePrompt = buildLivePromptFromCurrent(
              currentTime,
              currentTime
            );
            setPrompts(prevPrompts => [...prevPrompts, livePrompt]);
          }
        }

        // Start timeline playback immediately after creating/extending the live box
        // Use setTimeout to ensure the state update has been applied
        setTimeout(() => {
          togglePlayback();
          // Resume video if paused
          if (isVideoPaused && onVideoPlayPauseToggle) {
            onVideoPlayPauseToggle();
          }
        }, 0);
      } else {
        // Not at the end - resume both timeline and video
        togglePlayback();
        if (isVideoPaused && onVideoPlayPauseToggle) {
          onVideoPlayPauseToggle();
        }
      }

      // Track that we've started playback
      if (!hasStartedPlayback) {
        setHasStartedPlayback(true);
      }
    } else {
      // Pausing playback - pause both timeline and video
      togglePlayback();
      if (!isVideoPaused && onVideoPlayPauseToggle) {
        onVideoPlayPauseToggle();
      }
    }
  }, [
    isStreaming,
    onStartStream,
    prompts,
    currentTime,
    isActuallyPlaying,
    selectedPromptId,
    onPromptEdit,
    onLiveStateChange,
    setPrompts,
    togglePlayback,
    isVideoPaused,
    onVideoPlayPauseToggle,
    hasStartedPlayback,
    buildLivePromptFromCurrent,
  ]);

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
    (prompt: TimelinePrompt | null) => {
      if (onPromptEdit) {
        onPromptEdit(prompt);
      }
    },
    [onPromptEdit]
  );

  // Handle live prompt submission
  const handleLivePromptSubmit = React.useCallback(
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

        // Determine where to place the new prompt
        // If we're paused in the middle (not playing and not at the end), append to the end
        const sortedPrompts = [...updatedPrompts].sort(
          (a, b) => a.endTime - b.endTime
        );
        const lastPrompt = sortedPrompts[sortedPrompts.length - 1];
        const maxEndTime = lastPrompt ? lastPrompt.endTime : 0;
        const isAtEnd = currentTime >= maxEndTime;
        const isPausedInMiddle = !isActuallyPlaying && !isAtEnd;

        // Use end of timeline if paused in middle, otherwise use current time
        const startTime = isPausedInMiddle ? maxEndTime : currentTime;

        // Create new live box with blend information
        const newLivePrompt: TimelinePrompt = {
          id: `live-${Date.now()}`,
          text: promptItems.map(p => p.text).join(", "), // Combined text for display
          startTime: startTime,
          endTime: startTime, // Will be updated as time progresses
          isLive: true,
          prompts: promptItems.map(p => ({ text: p.text, weight: p.weight })), // Store blend info
        };

        return [...updatedPrompts, newLivePrompt];
      });

      // Set live state to true when creating a new live prompt box
      // This ensures that after rewind, when users submit prompts, they can continue adding more
      setIsLive(true);
      if (onLiveStateChange) {
        onLiveStateChange(true);
      }

      // Scroll timeline to show the new live prompt if it's not visible
      if (scrollToTimeFn) {
        scrollToTimeFn(currentTime);
      }

      // Do NOT call onPromptSubmit here; it would reset blends to a single prompt
    },
    [
      currentTime,
      setPrompts,
      isActuallyPlaying,
      onLiveStateChange,
      scrollToTimeFn,
    ]
  );

  // Handle prompt updates from the editor
  const handlePromptUpdate = React.useCallback(
    (updatedPrompt: TimelinePrompt) => {
      setPrompts(prevPrompts =>
        prevPrompts.map(p => (p.id === updatedPrompt.id ? updatedPrompt : p))
      );
    },
    [setPrompts]
  );

  // Note: Live box end time updates are handled by useTimelinePlayback hook
  // to avoid conflicts and ensure proper synchronization

  // Expose the live prompt submit function to parent
  React.useImperativeHandle(timelineRef, () => ({
    getCurrentTimelinePrompt,
    submitLivePrompt: handleLivePromptSubmit,
    updatePrompt: handlePromptUpdate,
    clearTimeline: () => setPrompts([]),
    resetPlayhead: resetPlayback,
    getPrompts: () => prompts,
    getCurrentTime: () => currentTime,
    getIsPlaying: () => isPlaying,
  }));

  // Expose play/pause handler to parent
  React.useEffect(() => {
    if (onPlayPauseRef) {
      onPlayPauseRef.current = handlePlayPause;
    }
  }, [handlePlayPause, onPlayPauseRef]);

  return (
    <div className={`space-y-3 ${className}`}>
      <PromptTimeline
        prompts={prompts}
        onPromptsChange={setPrompts}
        disabled={disabled}
        isPlaying={isActuallyPlaying}
        currentTime={currentTime}
        onPlayPause={handlePlayPause}
        onTimeChange={handleRewind}
        onDisconnect={handleEnhancedDisconnect}
        onPromptSubmit={onPromptSubmit}
        initialPrompt={currentPrompt}
        selectedPromptId={selectedPromptId}
        onPromptSelect={handlePromptSelect}
        onPromptEdit={handlePromptEdit}
        onLivePromptSubmit={onLivePromptSubmit}
        isCollapsed={isCollapsed}
        onCollapseToggle={onCollapseToggle}
        settings={settings}
        onSettingsImport={onSettingsImport}
        onScrollToTime={scrollFn => setScrollToTimeFn(() => scrollFn)}
      />
    </div>
  );
}
