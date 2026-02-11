import React, { useRef, useCallback, useEffect } from "react";
import type { TimelinePrompt } from "../components/PromptTimeline";

const MIN_DURATION_SECONDS = 0.5;

interface UseTimelineDragResizeParams {
  pixelsPerSecond: number;
  visibleStartTime: number;
  setVisibleStartTime: (time: number) => void;
  prompts: TimelinePrompt[];
  onPromptsChange: (prompts: TimelinePrompt[]) => void;
  isPlaying: boolean;
  timelineRef: React.RefObject<HTMLDivElement | null>;
}

export function useTimelineDragResize({
  pixelsPerSecond,
  visibleStartTime,
  setVisibleStartTime,
  prompts,
  onPromptsChange,
  isPlaying,
}: UseTimelineDragResizeParams) {
  // Resize state
  const resizeStateRef = useRef<{
    promptId: string;
    edge: "left" | "right";
    startClientX: number;
    startPrompt: TimelinePrompt;
    prevPrompt?: TimelinePrompt;
    nextPrompt?: TimelinePrompt;
  } | null>(null);

  // Drag-to-pan state
  const isDraggingRef = useRef(false);
  const dragStartXRef = useRef(0);
  const dragStartVisibleStartRef = useRef(0);

  const beginResize = useCallback(
    (
      e: React.MouseEvent,
      prompt: TimelinePrompt,
      edge: "left" | "right",
      prevPrompt?: TimelinePrompt,
      nextPrompt?: TimelinePrompt
    ) => {
      e.stopPropagation();
      if (isPlaying || prompt.isLive) return;
      resizeStateRef.current = {
        promptId: prompt.id,
        edge,
        startClientX: e.clientX,
        startPrompt: { ...prompt },
        prevPrompt: prevPrompt ? { ...prevPrompt } : undefined,
        nextPrompt: nextPrompt ? { ...nextPrompt } : undefined,
      };
      document.body.style.cursor = "col-resize";
    },
    [isPlaying]
  );

  const handleTimelineMouseDown = useCallback(
    (e: React.MouseEvent) => {
      isDraggingRef.current = true;
      dragStartXRef.current = e.clientX;
      dragStartVisibleStartRef.current = visibleStartTime;
      document.body.style.cursor = "grabbing";
    },
    [visibleStartTime]
  );

  // Global listeners to update panning and finish drag
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      // Resize has priority over panning
      if (resizeStateRef.current) {
        const state = resizeStateRef.current;
        const deltaX = e.clientX - state.startClientX;
        const deltaSeconds = deltaX / pixelsPerSecond;

        const index = prompts.findIndex(p => p.id === state.promptId);
        if (index === -1) return;

        const current = { ...state.startPrompt };
        const prev = index > 0 ? prompts[index - 1] : null;
        const next = index < prompts.length - 1 ? prompts[index + 1] : null;

        if (state.edge === "left") {
          let newStart = current.startTime + deltaSeconds;
          const leftBound = prev ? prev.startTime + MIN_DURATION_SECONDS : 0;
          const rightBound = current.endTime - MIN_DURATION_SECONDS;
          newStart = Math.max(leftBound, Math.min(newStart, rightBound));

          current.startTime = newStart;
          if (prev) {
            prev.endTime = newStart;
          }
        } else {
          let newEnd = current.endTime + deltaSeconds;
          const leftBound = current.startTime + MIN_DURATION_SECONDS;
          const rightBound = next
            ? next.endTime - MIN_DURATION_SECONDS
            : Number.POSITIVE_INFINITY;
          newEnd = Math.max(leftBound, Math.min(newEnd, rightBound));

          current.endTime = newEnd;
          if (next) {
            next.startTime = newEnd;
          }
        }

        const updated = prompts.map((p, i) => {
          if (i === index) return current;
          if (state.edge === "left" && prev && i === index - 1) return prev;
          if (state.edge === "right" && next && i === index + 1) return next;
          return p;
        });
        onPromptsChange(updated);
        return;
      }

      if (!isDraggingRef.current) return;
      const deltaX = e.clientX - dragStartXRef.current;
      const deltaSeconds = -deltaX / pixelsPerSecond;
      const nextStart = Math.max(
        0,
        dragStartVisibleStartRef.current + deltaSeconds
      );
      setVisibleStartTime(nextStart);
    };

    const handleMouseUp = () => {
      if (resizeStateRef.current) {
        resizeStateRef.current = null;
        document.body.style.cursor = "";
        return;
      }
      if (isDraggingRef.current) {
        isDraggingRef.current = false;
        document.body.style.cursor = "";
      }
    };

    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mouseup", handleMouseUp);
    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handleMouseUp);
    };
  }, [pixelsPerSecond, prompts, onPromptsChange, setVisibleStartTime]);

  return {
    beginResize,
    handleTimelineMouseDown,
    isDraggingRef,
  };
}
