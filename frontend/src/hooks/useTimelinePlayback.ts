import { useState, useRef, useCallback, useEffect } from "react";

import type { TimelinePrompt } from "../components/PromptTimeline";
import type { PromptItem } from "../lib/api";
import { submitTimelinePrompt } from "../utils/timelinePromptSubmission";

interface UseTimelinePlaybackOptions {
  onPromptChange?: (prompt: string) => void;
  onPromptItemsChange?: (
    prompts: PromptItem[],
    transitionSteps?: number,
    temporalInterpolationMethod?: "linear" | "slerp"
  ) => void;
  isStreaming?: boolean;
  isVideoPaused?: boolean;
  onPromptsChange?: (prompts: TimelinePrompt[]) => void;
  onCurrentTimeChange?: (currentTime: number) => void;
  onPlayingChange?: (isPlaying: boolean) => void;
}

// Animation frame update function
const createUpdateTimeFunction = (
  startTimeRef: React.MutableRefObject<number>,
  setCurrentTime: (time: number) => void,
  setPrompts: (updater: (prev: TimelinePrompt[]) => TimelinePrompt[]) => void,
  promptsRef: React.MutableRefObject<TimelinePrompt[]>,
  lastAppliedPromptIdRef: React.MutableRefObject<string | null>,
  lastAppliedPromptTextRef: React.MutableRefObject<string | null>,
  optionsRef: React.MutableRefObject<UseTimelinePlaybackOptions | undefined>,
  animationFrameRef: React.MutableRefObject<number | undefined>
) => {
  return () => {
    const now = performance.now();
    const elapsed = (now - startTimeRef.current) / 1000;

    setCurrentTime(elapsed);

    // Update live blocks to extend as time progresses
    const currentPrompts = promptsRef.current;
    const livePrompt = currentPrompts.find(p => p.isLive);
    if (livePrompt) {
      setPrompts(prevPrompts =>
        prevPrompts.map(p =>
          p.id === livePrompt.id ? { ...p, endTime: elapsed } : p
        )
      );
    }

    // Find active prompt and apply it only if it changed
    const activePrompt = currentPrompts.find(
      prompt => elapsed >= prompt.startTime && elapsed <= prompt.endTime
    );

    // Only send update if the active prompt block OR its text has changed
    if (
      activePrompt &&
      (activePrompt.id !== lastAppliedPromptIdRef.current ||
        activePrompt.text !== lastAppliedPromptTextRef.current)
    ) {
      submitTimelinePrompt(activePrompt, {
        onPromptSubmit: optionsRef.current?.onPromptChange,
        onPromptItemsSubmit: optionsRef.current?.onPromptItemsChange,
      });
      lastAppliedPromptIdRef.current = activePrompt.id;
      lastAppliedPromptTextRef.current = activePrompt.text;
    } else if (!activePrompt && lastAppliedPromptIdRef.current !== null) {
      // No active prompt, reset the last applied prompt
      lastAppliedPromptIdRef.current = null;
      lastAppliedPromptTextRef.current = null;
    }

    // Check if we've reached the end of all prompts
    const hasLiveBlock = currentPrompts.some(p => p.isLive);

    // If we've reached the end and there are prompts, extend the last box as live
    if (currentPrompts.length > 0 && !hasLiveBlock) {
      const lastPrompt = currentPrompts[currentPrompts.length - 1];

      if (elapsed >= lastPrompt.endTime) {
        // Make the last prompt live
        setPrompts(prevPrompts =>
          prevPrompts.map(p =>
            p.id === lastPrompt.id
              ? { ...p, isLive: true, endTime: elapsed }
              : p
          )
        );

        // Notify parent that live mode has started
        submitTimelinePrompt(lastPrompt, {
          onPromptSubmit: optionsRef.current?.onPromptChange,
          onPromptItemsSubmit: optionsRef.current?.onPromptItemsChange,
        });

        // Continue playing instead of stopping
        animationFrameRef.current = requestAnimationFrame(
          createUpdateTimeFunction(
            startTimeRef,
            setCurrentTime,
            setPrompts,
            promptsRef,
            lastAppliedPromptIdRef,
            lastAppliedPromptTextRef,
            optionsRef,
            animationFrameRef
          )
        );
        return;
      }
    }

    // Continue animation frame loop
    animationFrameRef.current = requestAnimationFrame(
      createUpdateTimeFunction(
        startTimeRef,
        setCurrentTime,
        setPrompts,
        promptsRef,
        lastAppliedPromptIdRef,
        lastAppliedPromptTextRef,
        optionsRef,
        animationFrameRef
      )
    );
  };
};

export function useTimelinePlayback(options?: UseTimelinePlaybackOptions) {
  const [prompts, setPrompts] = useState<TimelinePrompt[]>([]);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);

  const animationFrameRef = useRef<number | undefined>(undefined);
  const startTimeRef = useRef<number>(0);
  const lastAppliedPromptIdRef = useRef<string | null>(null);
  const lastAppliedPromptTextRef = useRef<string | null>(null);
  const optionsRef = useRef(options);
  const promptsRef = useRef(prompts);

  // Update refs when dependencies change
  useEffect(() => {
    optionsRef.current = options;
  }, [options]);

  useEffect(() => {
    promptsRef.current = prompts;
  }, [prompts]);

  // Emit events when state changes
  useEffect(() => {
    options?.onPromptsChange?.(prompts);
  }, [prompts, options]);

  useEffect(() => {
    options?.onCurrentTimeChange?.(currentTime);
  }, [currentTime, options]);

  useEffect(() => {
    options?.onPlayingChange?.(isPlaying);
  }, [isPlaying, options]);

  // Pause playback
  const pausePlayback = useCallback(() => {
    setIsPlaying(false);
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
    }
  }, []);

  // Start playback
  const startPlayback = useCallback(() => {
    setIsPlaying(true);
    startTimeRef.current = performance.now() - currentTime * 1000;

    const updateTime = createUpdateTimeFunction(
      startTimeRef,
      setCurrentTime,
      setPrompts,
      promptsRef,
      lastAppliedPromptIdRef,
      lastAppliedPromptTextRef,
      optionsRef,
      animationFrameRef
    );

    animationFrameRef.current = requestAnimationFrame(updateTime);
  }, [currentTime]);

  // Toggle playback
  const togglePlayback = useCallback(() => {
    if (isPlaying) {
      pausePlayback();
    } else {
      startPlayback();
    }
  }, [isPlaying, startPlayback, pausePlayback]);

  // Reset playback
  const resetPlayback = useCallback(() => {
    pausePlayback();
    setCurrentTime(0);
  }, [pausePlayback]);

  // Update current time
  const updateCurrentTime = useCallback((time: number) => {
    setCurrentTime(time);
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, []);

  return {
    prompts,
    setPrompts,
    isPlaying,
    currentTime,
    updateCurrentTime,
    togglePlayback,
    resetPlayback,
    startPlayback,
    pausePlayback,
  };
}
