import React, {
  useState,
  useRef,
  useEffect,
  useCallback,
  useMemo,
} from "react";

import { Button } from "./ui/button";
import { Card, CardContent } from "./ui/card";
import { ZoomIn, ZoomOut } from "lucide-react";
import { TimelineHeader } from "./timeline/TimelineHeader";
import { useTimelineImportExport } from "../hooks/useTimelineImportExport";
import { useTimelineDragResize } from "../hooks/useTimelineDragResize";

import type { PromptItem } from "../lib/api";
import type { SettingsState } from "../types";
import { generateRandomColor } from "../utils/promptColors";

// Timeline constants
const BASE_PIXELS_PER_SECOND = 20;
const MAX_ZOOM_LEVEL = 4;
const MIN_ZOOM_LEVEL = 0.25;
const DEFAULT_VISIBLE_END_TIME = 20;

// Utility functions
const timeToPosition = (
  time: number,
  visibleStartTime: number,
  pixelsPerSecond: number
): number => {
  return (time - visibleStartTime) * pixelsPerSecond;
};

const formatTime = (seconds: number): string => {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
};

const getAdjacentColors = (
  prompts: TimelinePrompt[],
  currentIndex: number
): string[] => {
  const adjacentColors: string[] = [];
  if (currentIndex > 0 && prompts[currentIndex - 1].color) {
    adjacentColors.push(prompts[currentIndex - 1].color!);
  }
  if (currentIndex < prompts.length - 1 && prompts[currentIndex + 1].color) {
    adjacentColors.push(prompts[currentIndex + 1].color!);
  }
  return adjacentColors;
};

const calculatePromptPosition = (
  prompt: TimelinePrompt,
  index: number,
  visiblePrompts: TimelinePrompt[],
  timeToPositionFn: (time: number) => number
): number => {
  let leftPosition = Math.max(0, timeToPositionFn(prompt.startTime));

  if (index > 0) {
    const previousPrompt = visiblePrompts[index - 1];
    const previousEndPosition = Math.max(
      0,
      timeToPositionFn(previousPrompt.endTime)
    );
    leftPosition = Math.max(leftPosition, previousEndPosition);
  }

  return leftPosition;
};

const getPromptBoxStyle = (
  prompt: TimelinePrompt,
  leftPosition: number,
  timelineWidth: number,
  timeToPositionFn: (time: number) => number,
  isSelected: boolean,
  isLivePrompt: boolean,
  boxColor: string
) => {
  return {
    left: leftPosition,
    top: "8px",
    bottom: "8px",
    width: Math.min(
      timelineWidth - leftPosition,
      timeToPositionFn(prompt.endTime) - leftPosition
    ),
    backgroundColor: isLivePrompt ? "#6B7280" : boxColor,
    borderColor: isLivePrompt ? "#9CA3AF" : boxColor,
    opacity: isSelected ? 1.0 : 0.7,
  };
};

export interface TimelinePrompt {
  id: string;
  text: string;
  startTime: number;
  endTime: number;
  prompts?: Array<{ text: string; weight: number }>;
  color?: string;
  isLive?: boolean;
  transitionSteps?: number;
  temporalInterpolationMethod?: "linear" | "slerp";
}

const TIMELINE_RESET_STATE = {
  timelineWidth: 800,
  visibleStartTime: 0,
  visibleEndTime: 20,
  zoomLevel: 1,
};

interface PromptTimelineProps {
  className?: string;
  prompts: TimelinePrompt[];
  onPromptsChange: (prompts: TimelinePrompt[]) => void;
  disabled?: boolean;
  isPlaying?: boolean;
  currentTime?: number;
  onPlayPause?: () => void;
  onTimeChange?: (time: number) => void;
  onReset?: () => void;
  onClear?: () => void;
  onPromptSubmit?: (prompt: string) => void;
  initialPrompt?: string;
  selectedPromptId?: string | null;
  onPromptSelect?: (promptId: string | null) => void;
  onPromptEdit?: (prompt: TimelinePrompt | null) => void;
  onLivePromptSubmit?: (prompts: PromptItem[]) => void;
  isCollapsed?: boolean;
  onCollapseToggle?: (collapsed: boolean) => void;
  settings?: SettingsState;
  onSettingsImport?: (settings: Partial<SettingsState>) => void;
  onScrollToTime?: (scrollFn: (time: number) => void) => void;
  isStreaming?: boolean;
  isLoading?: boolean;
  videoScaleMode?: "fit" | "native";
  onVideoScaleModeToggle?: () => void;
  isDownloading?: boolean;
  onSaveGeneration?: () => void;
  isRecording?: boolean;
  onRecordingToggle?: () => void;
}

export function PromptTimeline({
  className = "",
  prompts,
  onPromptsChange,
  disabled = false,
  isPlaying = false,
  currentTime = 0,
  onPlayPause,
  onTimeChange,
  onReset,
  onClear,
  onPromptSubmit,
  initialPrompt: _initialPrompt,
  selectedPromptId = null,
  onPromptSelect,
  onPromptEdit,
  onLivePromptSubmit: _onLivePromptSubmit,
  isCollapsed = false,
  onCollapseToggle,
  settings,
  onSettingsImport,
  onScrollToTime,
  isStreaming = false,
  isLoading = false,
  videoScaleMode = "fit",
  onVideoScaleModeToggle,
  isDownloading = false,
  onSaveGeneration,
  isRecording = false,
  onRecordingToggle,
}: PromptTimelineProps) {
  const timelineRef = useRef<HTMLDivElement>(null);
  const [timelineWidth, setTimelineWidth] = useState(800);
  const [visibleStartTime, setVisibleStartTime] = useState(0);
  const [visibleEndTime, setVisibleEndTime] = useState(
    DEFAULT_VISIBLE_END_TIME
  );
  const [zoomLevel, setZoomLevel] = useState(1);

  const isLive = useMemo(() => prompts.some(p => p.isLive), [prompts]);

  const visiblePrompts = useMemo(() => {
    return prompts.filter(
      prompt =>
        prompt.startTime !== prompt.endTime &&
        prompt.endTime >= visibleStartTime &&
        prompt.startTime <= visibleEndTime
    );
  }, [prompts, visibleStartTime, visibleEndTime]);

  const pixelsPerSecond = useMemo(
    () => BASE_PIXELS_PER_SECOND * zoomLevel,
    [zoomLevel]
  );
  const visibleTimeRange = useMemo(
    () => timelineWidth / pixelsPerSecond,
    [timelineWidth, pixelsPerSecond]
  );

  // Scroll timeline to show a specific time
  const scrollToTime = useCallback(
    (time: number) => {
      const targetVisibleStartTime = Math.max(0, time - visibleTimeRange * 0.5);
      setVisibleStartTime(targetVisibleStartTime);
    },
    [visibleTimeRange]
  );

  useEffect(() => {
    if (onScrollToTime) {
      onScrollToTime(scrollToTime);
    }
  }, [onScrollToTime, scrollToTime]);

  const resetTimelineUI = useCallback(() => {
    setTimelineWidth(TIMELINE_RESET_STATE.timelineWidth);
    setVisibleStartTime(TIMELINE_RESET_STATE.visibleStartTime);
    setVisibleEndTime(TIMELINE_RESET_STATE.visibleEndTime);
    setZoomLevel(TIMELINE_RESET_STATE.zoomLevel);
  }, []);

  useEffect(() => {
    setVisibleEndTime(visibleStartTime + visibleTimeRange);
  }, [visibleStartTime, visibleTimeRange]);

  // Hook: drag-to-pan + resize
  const { beginResize, handleTimelineMouseDown, isDraggingRef } =
    useTimelineDragResize({
      pixelsPerSecond,
      visibleStartTime,
      setVisibleStartTime,
      prompts,
      onPromptsChange,
      isPlaying,
      timelineRef,
    });

  // Auto-scroll timeline during playback
  useEffect(() => {
    if (isDraggingRef.current || !isPlaying) return;

    if (currentTime > visibleEndTime - visibleTimeRange * 0.2) {
      setVisibleStartTime(Math.max(0, currentTime - visibleTimeRange * 0.8));
    } else if (currentTime < visibleStartTime + visibleTimeRange * 0.2) {
      setVisibleStartTime(Math.max(0, currentTime - visibleTimeRange * 0.2));
    }
  }, [
    isPlaying,
    currentTime,
    visibleEndTime,
    visibleStartTime,
    visibleTimeRange,
    isDraggingRef,
  ]);

  // Update timeline width on mount/resize
  useEffect(() => {
    const updateWidth = () => {
      if (timelineRef.current) {
        setTimelineWidth(timelineRef.current.offsetWidth);
      }
    };

    updateWidth();
    window.addEventListener("resize", updateWidth);
    return () => window.removeEventListener("resize", updateWidth);
  }, []);

  // Memoized calculations
  const timeToPositionMemo = useCallback(
    (time: number) => timeToPosition(time, visibleStartTime, pixelsPerSecond),
    [visibleStartTime, pixelsPerSecond]
  );

  const currentTimePosition = useMemo(() => {
    return Math.max(
      0,
      Math.min(
        timelineWidth,
        (currentTime - visibleStartTime) * pixelsPerSecond
      )
    );
  }, [currentTime, timelineWidth, visibleStartTime, pixelsPerSecond]);

  const timeMarkers = useMemo(() => {
    const targetPixelGap = 100;
    const idealInterval = targetPixelGap / pixelsPerSecond;

    const niceIntervals = [1, 2, 5, 10, 15, 30, 60];
    let interval = niceIntervals[niceIntervals.length - 1];
    for (const nice of niceIntervals) {
      if (nice >= idealInterval) {
        interval = nice;
        break;
      }
    }

    const startTime = Math.floor(visibleStartTime / interval) * interval;
    const endTime = Math.ceil(visibleEndTime / interval) * interval;

    const markers = [];
    for (let time = startTime; time <= endTime; time += interval) {
      const position = (time - visibleStartTime) * pixelsPerSecond;
      markers.push({ time, position });
    }
    return markers;
  }, [visibleEndTime, visibleStartTime, pixelsPerSecond]);

  // Prompt interaction handlers
  const handlePromptClick = useCallback(
    (e: React.MouseEvent, prompt: TimelinePrompt) => {
      e.stopPropagation();
      if (prompt.isLive) return;
      if (isPlaying && selectedPromptId !== prompt.id) return;

      const isCurrentlySelected = selectedPromptId === prompt.id;

      if (onPromptSelect) {
        onPromptSelect(isCurrentlySelected ? null : prompt.id);
      }
      if (onPromptEdit) {
        onPromptEdit(isCurrentlySelected ? null : prompt);
      }
    },
    [selectedPromptId, onPromptSelect, onPromptEdit, isPlaying]
  );

  const handleTimelineClick = useCallback(
    (_e: React.MouseEvent) => {
      if (selectedPromptId && onPromptSelect) {
        onPromptSelect(null);
      }
      if (selectedPromptId && onPromptEdit) {
        onPromptEdit(null);
      }
    },
    [selectedPromptId, onPromptSelect, onPromptEdit]
  );

  // Hook: import/export
  const {
    showExportDialog,
    setShowExportDialog,
    handleExport,
    handleImport,
    handleSaveTimeline,
  } = useTimelineImportExport({
    prompts,
    settings,
    onPromptsChange,
    onSettingsImport,
    onTimeChange,
    onPromptSubmit,
    resetTimelineUI,
    setVisibleStartTime,
    setVisibleEndTime,
  });

  // Zoom functions
  const zoomIn = useCallback(() => {
    setZoomLevel(prev => Math.min(prev * 2, MAX_ZOOM_LEVEL));
  }, []);

  const zoomOut = useCallback(() => {
    setZoomLevel(prev => Math.max(prev / 2, MIN_ZOOM_LEVEL));
  }, []);

  return (
    <Card className={`${className}`}>
      <CardContent className={`p-4 ${isCollapsed ? "py-2" : ""}`}>
        <TimelineHeader
          isPlaying={isPlaying}
          isCollapsed={isCollapsed}
          disabled={disabled}
          isLoading={isLoading}
          isDownloading={isDownloading}
          isStreaming={isStreaming}
          isRecording={isRecording}
          videoScaleMode={videoScaleMode}
          showExportDialog={showExportDialog}
          onPlayPause={onPlayPause}
          onReset={onReset}
          onRecordingToggle={onRecordingToggle}
          onClear={onClear}
          onVideoScaleModeToggle={onVideoScaleModeToggle}
          onExport={handleExport}
          onCloseExportDialog={() => setShowExportDialog(false)}
          onSaveGeneration={onSaveGeneration}
          onSaveTimeline={handleSaveTimeline}
          onImport={handleImport}
          onCollapseToggle={onCollapseToggle}
        />

        {/* Timeline */}
        {!isCollapsed && (
          <div className="relative overflow-hidden w-full" ref={timelineRef}>
            {/* Time markers */}
            <div className="relative mb-1 w-full" style={{ height: "30px" }}>
              {timeMarkers.map(({ time, position }) => (
                <div
                  key={time}
                  className="absolute top-0 flex items-center justify-center"
                  style={{
                    left: time === 0 ? position + 10 : position,
                    transform: "translateX(-50%)",
                  }}
                >
                  <span className="text-gray-400 text-xs">
                    {formatTime(time)}
                  </span>
                </div>
              ))}
            </div>

            {/* Timeline track */}
            <div
              className="relative bg-muted rounded-lg border overflow-hidden cursor-grab w-full"
              style={{ height: "80px" }}
              onClick={handleTimelineClick}
              onMouseDown={handleTimelineMouseDown}
            >
              {/* Zoom controls */}
              <div className="absolute bottom-2 right-2 flex items-center gap-1 z-50">
                <Button
                  onClick={zoomOut}
                  disabled={zoomLevel <= MIN_ZOOM_LEVEL}
                  size="sm"
                  variant="outline"
                  className="text-xs px-2 h-6"
                  title="Zoom Out"
                >
                  <ZoomOut className="h-3 w-3" />
                </Button>
                <span className="text-xs text-muted-foreground px-1">
                  {zoomLevel}x
                </span>
                <Button
                  onClick={zoomIn}
                  disabled={zoomLevel >= MAX_ZOOM_LEVEL}
                  size="sm"
                  variant="outline"
                  className="text-xs px-2 h-6"
                  title="Zoom In"
                >
                  <ZoomIn className="h-3 w-3" />
                </Button>
              </div>

              {/* Current time cursor */}
              <div
                className="absolute top-0 bottom-0 w-1 bg-red-500 z-30 shadow-lg"
                style={{
                  left: currentTimePosition,
                  display: "block",
                }}
              />

              {/* Prompt blocks */}
              {visiblePrompts.map((prompt, index) => {
                const isSelected = selectedPromptId === prompt.id;
                const isActive =
                  currentTime >= prompt.startTime &&
                  currentTime <= prompt.endTime;
                const isLiveActive = isLive && currentTime >= prompt.startTime;
                const isLivePrompt = prompt.isLive;

                let boxColor = prompt.color;
                if (!boxColor) {
                  const adjacentColors = getAdjacentColors(
                    visiblePrompts,
                    index
                  );
                  boxColor = generateRandomColor(adjacentColors);

                  const updatedPrompt = { ...prompt, color: boxColor };
                  const updatedPrompts = prompts.map(p =>
                    p.id === prompt.id ? updatedPrompt : p
                  );
                  onPromptsChange(updatedPrompts);
                }

                const leftPosition = calculatePromptPosition(
                  prompt,
                  index,
                  visiblePrompts,
                  timeToPositionMemo
                );

                const prevPrompt =
                  index > 0 ? visiblePrompts[index - 1] : undefined;
                const nextPrompt =
                  index < visiblePrompts.length - 1
                    ? visiblePrompts[index + 1]
                    : undefined;

                const isEditable = !isPlaying || isSelected || isLivePrompt;

                return (
                  <div
                    key={prompt.id}
                    className={`absolute border rounded px-2 py-1 transition-colors ${
                      isEditable ? "cursor-pointer" : "cursor-default"
                    } ${
                      isSelected
                        ? "shadow-lg border-blue-500"
                        : isActive
                          ? "border-green-500"
                          : isLiveActive
                            ? "border-red-500"
                            : ""
                    }`}
                    style={getPromptBoxStyle(
                      prompt,
                      leftPosition,
                      timelineWidth,
                      timeToPositionMemo,
                      isSelected,
                      isLivePrompt || false,
                      boxColor
                    )}
                    onClick={e => handlePromptClick(e, prompt)}
                  >
                    {/* Resize handles */}
                    {!isPlaying && !isLivePrompt && (
                      <>
                        <div
                          className="absolute top-0 bottom-0 w-2 -left-1 z-40"
                          style={{ cursor: "col-resize" }}
                          onMouseDown={e =>
                            beginResize(
                              e,
                              prompt,
                              "left",
                              prevPrompt,
                              nextPrompt
                            )
                          }
                        />
                        <div
                          className="absolute top-0 bottom-0 w-2 -right-1 z-40"
                          style={{ cursor: "col-resize" }}
                          onMouseDown={e =>
                            beginResize(
                              e,
                              prompt,
                              "right",
                              prevPrompt,
                              nextPrompt
                            )
                          }
                        />
                      </>
                    )}
                    <div className="flex flex-col justify-center h-full">
                      <div className="flex-1 flex flex-col justify-center">
                        {prompt.prompts && prompt.prompts.length > 1 ? (
                          prompt.prompts.map((promptItem, idx) => (
                            <div
                              key={idx}
                              className="text-xs text-white font-medium truncate"
                            >
                              {promptItem.text} ({promptItem.weight}%)
                            </div>
                          ))
                        ) : (
                          <span className="text-xs text-white font-medium truncate">
                            {prompt.text}
                          </span>
                        )}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
