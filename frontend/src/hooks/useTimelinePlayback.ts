import { useState, useRef, useCallback, useEffect } from "react";
import type { TimelinePrompt } from "../components/PromptTimeline";

interface UseTimelinePlaybackOptions {
  onPromptChange?: (prompt: string) => void;
  isStreaming?: boolean;
}

export function useTimelinePlayback(options?: UseTimelinePlaybackOptions) {
  const [prompts, setPrompts] = useState<TimelinePrompt[]>([]);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);

  const animationFrameRef = useRef<number | undefined>(undefined);
  const lastTimeRef = useRef<number>(0);
  const startTimeRef = useRef<number>(0);
  const optionsRef = useRef(options);

  // Update options ref when options change
  useEffect(() => {
    optionsRef.current = options;
  }, [options]);

  const startPlayback = useCallback(() => {
    console.log("Starting playback, current time:", currentTime);
    setIsPlaying(true);
    startTimeRef.current = performance.now() - currentTime * 1000;
    lastTimeRef.current = performance.now();

    const updateTime = () => {
      const now = performance.now();
      const elapsed = (now - startTimeRef.current) / 1000;

      setCurrentTime(elapsed);

      // Find active prompt and apply it
      const activePrompt = prompts.find(
        prompt => elapsed >= prompt.startTime && elapsed <= prompt.endTime
      );

      if (activePrompt && optionsRef.current?.onPromptChange) {
        optionsRef.current.onPromptChange(activePrompt.text);
      }

      // Continue animation frame loop
      animationFrameRef.current = requestAnimationFrame(updateTime);
    };

    animationFrameRef.current = requestAnimationFrame(updateTime);
  }, [currentTime, prompts]);

  const pausePlayback = useCallback(() => {
    setIsPlaying(false);
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
    }
  }, []);

  const togglePlayback = useCallback(() => {
    if (isPlaying) {
      pausePlayback();
    } else {
      startPlayback();
    }
  }, [isPlaying, startPlayback, pausePlayback]);

  const resetPlayback = useCallback(() => {
    pausePlayback();
    setCurrentTime(0);
  }, [pausePlayback]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, []);

  const updateCurrentTime = useCallback((time: number) => {
    setCurrentTime(time);
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
