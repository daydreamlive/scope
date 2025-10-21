import React, { useState, useRef, useEffect, useCallback } from "react";
import { Button } from "./ui/button";
import { Card, CardContent } from "./ui/card";
import {
  Play,
  Pause,
  Plus,
  Download,
  Upload,
  ZoomIn,
  ZoomOut,
  Square,
  RotateCcw,
} from "lucide-react";
import type { PromptItem } from "../lib/api";

export interface TimelinePrompt {
  id: string;
  text: string;
  startTime: number; // in seconds
  endTime: number; // in seconds
  prompts?: Array<{ text: string; weight: number }>; // For prompt blending
  color?: string; // Random color for the box
  isLive?: boolean; // Whether this is a live recording box
}

interface PromptTimelineProps {
  className?: string;
  prompts: TimelinePrompt[];
  onPromptsChange: (prompts: TimelinePrompt[]) => void;
  disabled?: boolean;
  isPlaying?: boolean;
  currentTime?: number; // in seconds
  onPlayPause?: () => void;
  onTimeChange?: (time: number) => void;
  isRecording?: boolean;
  onRecordingToggle?: () => void;
  onPromptSubmit?: (prompt: string) => void;
  initialPrompt?: string;
  selectedPromptId?: string | null;
  onPromptSelect?: (promptId: string | null) => void;
  onPromptEdit?: (prompt: TimelinePrompt) => void;
  onRecordingPromptSubmit?: (prompts: PromptItem[]) => void;
  isRecordDisabled?: boolean;
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
  isRecording = false,
  onRecordingToggle,
  onPromptSubmit: _onPromptSubmit,
  initialPrompt: _initialPrompt,
  selectedPromptId = null,
  onPromptSelect,
  onPromptEdit,
  onRecordingPromptSubmit,
  isRecordDisabled = false,
}: PromptTimelineProps) {
  const timelineRef = useRef<HTMLDivElement>(null);
  const [timelineWidth, setTimelineWidth] = useState(800);
  const [visibleStartTime, setVisibleStartTime] = useState(0);
  const [visibleEndTime, setVisibleEndTime] = useState(20); // Changed from 40 to 20
  const [zoomLevel, setZoomLevel] = useState(1); // 1 = 20s, 2 = 10s, 0.5 = 40s
  const basePixelsPerSecond = 20; // Base pixels per second

  // Generate random colors for prompt boxes, ensuring adjacent boxes have different colors
  const generateRandomColor = useCallback((excludeColors: string[] = []) => {
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
  }, []);
  const pixelsPerSecond = basePixelsPerSecond * zoomLevel; // Scaled pixels per second

  // Calculate visible time range based on zoom level and timeline width
  const visibleTimeRange = timelineWidth / pixelsPerSecond;

  // Update visible end time when zoom level or timeline width changes
  useEffect(() => {
    setVisibleEndTime(visibleStartTime + visibleTimeRange);
  }, [visibleStartTime, visibleTimeRange]);

  // Auto-scroll timeline during recording to follow the red line
  useEffect(() => {
    if (isRecording && currentTime > visibleEndTime - visibleTimeRange * 0.2) {
      // When the red line gets close to the right edge, scroll forward
      setVisibleStartTime(currentTime - visibleTimeRange * 0.8);
    } else if (
      isRecording &&
      currentTime < visibleStartTime + visibleTimeRange * 0.2
    ) {
      // When the red line gets close to the left edge, scroll backward
      setVisibleStartTime(Math.max(0, currentTime - visibleTimeRange * 0.2));
    }
  }, [
    isRecording,
    currentTime,
    visibleEndTime,
    visibleStartTime,
    visibleTimeRange,
  ]);

  // Update timeline width when component mounts or resizes
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

  // Resize state
  const resizeStateRef = useRef<{
    promptId: string;
    edge: "left" | "right";
    startClientX: number;
    startPrompt: TimelinePrompt;
    prevPrompt?: TimelinePrompt;
    nextPrompt?: TimelinePrompt;
  } | null>(null);
  const MIN_DURATION_SECONDS = 0.5;

  const beginResize = useCallback(
    (
      e: React.MouseEvent,
      prompt: TimelinePrompt,
      edge: "left" | "right",
      prevPrompt?: TimelinePrompt,
      nextPrompt?: TimelinePrompt
    ) => {
      e.stopPropagation();
      if (isRecording || prompt.isLive) return;
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
    [isRecording]
  );

  const timeToPosition = useCallback(
    (time: number) => {
      return (time - visibleStartTime) * pixelsPerSecond;
    },
    [visibleStartTime, pixelsPerSecond]
  );

  const handlePromptClick = useCallback(
    (e: React.MouseEvent, prompt: TimelinePrompt) => {
      e.stopPropagation();
      if (isRecording) return; // Don't allow selection during recording

      if (onPromptSelect) {
        onPromptSelect(prompt.id);
      }
      if (onPromptEdit) {
        onPromptEdit(prompt);
      }
    },
    [isRecording, onPromptSelect, onPromptEdit]
  );

  // Disable click-to-seek; clicks should not affect playhead
  const handleTimelineClick = useCallback((_e: React.MouseEvent) => {
    // Intentionally no-op to satisfy requirement #4
  }, []);

  const handleExport = useCallback(() => {
    // Filter out id, text, isLive, and color from prompts for export
    const exportPrompts = prompts.map(prompt => {
      const { id, text, isLive, color, ...exportPrompt } = prompt;
      // Suppress unused variable warnings for intentionally excluded fields
      void id;
      void text;
      void isLive;
      void color;
      return exportPrompt;
    });

    const timelineData = {
      prompts: exportPrompts,
      version: "1.0",
      exportedAt: new Date().toISOString(),
    };

    const dataStr = JSON.stringify(timelineData, null, 2);
    const dataBlob = new Blob([dataStr], { type: "application/json" });
    const url = URL.createObjectURL(dataBlob);

    const link = document.createElement("a");
    link.href = url;
    link.download = `timeline-${new Date().toISOString().split("T")[0]}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }, [prompts]);

  const handleImport = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      if (!file) return;

      const reader = new FileReader();
      reader.onload = e => {
        try {
          const content = e.target?.result as string;
          const timelineData = JSON.parse(content);

          if (timelineData.prompts && Array.isArray(timelineData.prompts)) {
            // Assign default values for id, text, isLive, and color when importing
            const importedPrompts = timelineData.prompts.map(
              (prompt: Partial<TimelinePrompt>, index: number) => ({
                ...prompt,
                id: prompt.id || `imported-${Date.now()}-${index}`,
                text:
                  prompt.text ||
                  (prompt.prompts && prompt.prompts.length > 0
                    ? prompt.prompts
                        .map((p: { text: string; weight: number }) => p.text)
                        .join(", ")
                    : ""),
                isLive: prompt.isLive || false,
                color: prompt.color || generateRandomColor(),
              })
            );
            onPromptsChange(importedPrompts);
          } else {
            alert("Invalid timeline file format");
          }
        } catch (error) {
          alert("Error reading timeline file");
          console.error("Import error:", error);
        }
      };
      reader.readAsText(file);

      // Reset the input so the same file can be selected again
      event.target.value = "";
    },
    [onPromptsChange, generateRandomColor]
  );

  // Drag-to-pan state
  const isDraggingRef = useRef(false);
  const dragStartXRef = useRef(0);
  const dragStartVisibleStartRef = useRef(0);

  // Handle mouse down on the track to begin panning
  const handleTimelineMouseDown = useCallback(
    (e: React.MouseEvent) => {
      if (!timelineRef.current) return;
      isDraggingRef.current = true;
      dragStartXRef.current = e.clientX;
      dragStartVisibleStartRef.current = visibleStartTime;
      // Change cursor to grabbing while dragging
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

        const sorted = [...prompts].sort((a, b) => a.startTime - b.startTime);
        const index = sorted.findIndex(p => p.id === state.promptId);
        if (index === -1) return;

        const current = { ...state.startPrompt };
        const prev = state.prevPrompt;
        const next = state.nextPrompt;

        if (state.edge === "left") {
          let newStart = current.startTime + deltaSeconds;
          const leftBound = prev ? prev.startTime + MIN_DURATION_SECONDS : 0;
          const rightBound = current.endTime - MIN_DURATION_SECONDS;
          newStart = Math.max(leftBound, Math.min(newStart, rightBound));

          current.startTime = newStart;
          if (prev) {
            // Keep adjacency
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
            // Keep adjacency
            next.startTime = newEnd;
          }
        }

        const updated = sorted.map((p, i) => {
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
  }, [pixelsPerSecond, prompts, onPromptsChange]);

  const addPrompt = useCallback(() => {
    // Find the latest end time among existing prompts
    const latestEndTime =
      prompts.length > 0 ? Math.max(...prompts.map(p => p.endTime)) : 0;

    // Exclude the last prompt's color to avoid adjacent duplicates
    const lastPrompt =
      prompts.length > 0
        ? [...prompts].sort((a, b) => a.startTime - b.startTime)[
            prompts.length - 1
          ]
        : undefined;
    const excludeColor = lastPrompt?.color ? [lastPrompt.color] : [];

    const newPrompt: TimelinePrompt = {
      id: Date.now().toString(),
      text: "New prompt",
      startTime: latestEndTime, // Start immediately after the last prompt
      endTime: latestEndTime + 10, // 10 second duration
      color: generateRandomColor(excludeColor),
    };
    onPromptsChange([...prompts, newPrompt]);
  }, [prompts, onPromptsChange, generateRandomColor]);

  // Zoom functions
  const zoomIn = useCallback(() => {
    setZoomLevel(prev => Math.min(prev * 2, 4)); // Max zoom 4x
  }, []);

  const zoomOut = useCallback(() => {
    setZoomLevel(prev => Math.max(prev / 2, 0.25)); // Min zoom 0.25x
  }, []);

  // Handle recording prompt submission from external source
  const handleExternalRecordingPromptSubmit = useCallback(
    (promptItems: PromptItem[]) => {
      if (!promptItems.length || !promptItems.some(p => p.text.trim())) return;

      // Complete the current live box and start a new one
      const lastNonLive = [...prompts].filter(p => !p.isLive).slice(-1)[0];
      const updatedPrompts = prompts.map(p => {
        if (!p.isLive) return p;
        const exclude = lastNonLive?.color ? [lastNonLive.color] : [];
        return {
          ...p,
          endTime: currentTime,
          isLive: false,
          color: generateRandomColor(exclude),
        };
      });

      // Create new live box with blend information
      const newLivePrompt: TimelinePrompt = {
        id: `live-${Date.now()}`,
        text: promptItems.map(p => p.text).join(", "), // Combined text for display
        startTime: currentTime,
        endTime: currentTime, // Will be updated as time progresses
        isLive: true,
        prompts: promptItems.map(p => ({ text: p.text, weight: p.weight })), // Store blend info
      };

      onPromptsChange([...updatedPrompts, newLivePrompt]);

      // Also call the external prompt submit handler if provided
      if (_onPromptSubmit) {
        _onPromptSubmit(promptItems[0].text); // Send first prompt for backward compatibility
      }
    },
    [
      prompts,
      currentTime,
      onPromptsChange,
      _onPromptSubmit,
      generateRandomColor,
    ]
  );

  // Expose the function to parent via ref or callback
  React.useEffect(() => {
    if (onRecordingPromptSubmit) {
      // This is a bit of a hack, but we need to expose the function
      // In a real implementation, you might want to use a ref or context
      (
        window as unknown as Record<string, unknown>
      ).handleRecordingPromptSubmit = handleExternalRecordingPromptSubmit;
    }
  }, [handleExternalRecordingPromptSubmit, onRecordingPromptSubmit]);

  return (
    <Card className={`${className}`}>
      <CardContent className="p-4">
        {/* Timeline Header */}
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <Button
              onClick={onPlayPause}
              disabled={disabled || isRecording}
              size="sm"
              variant="outline"
            >
              {isPlaying ? (
                <Pause className="h-4 w-4" />
              ) : (
                <Play className="h-4 w-4" />
              )}
            </Button>
            <Button
              onClick={() => onTimeChange?.(0)}
              disabled={disabled || isRecording}
              size="sm"
              variant="outline"
              title="Reset to beginning"
            >
              <RotateCcw className="h-4 w-4" />
            </Button>
            <Button
              onClick={onRecordingToggle}
              disabled={disabled || isRecordDisabled}
              size="sm"
              variant={isRecording ? "destructive" : "outline"}
              className={isRecording ? "animate-pulse" : ""}
            >
              {isRecording ? (
                <Square className="h-4 w-4" />
              ) : (
                <div className="h-4 w-4 rounded-full bg-red-500" />
              )}
              {isRecording ? "Stop" : "Record"}
            </Button>
            <Button
              onClick={addPrompt}
              disabled={disabled || isRecording}
              size="sm"
              variant="outline"
            >
              <Plus className="h-4 w-4 mr-1" />
              Add Prompt
            </Button>
            {/* Removed live timer text, scroll arrows, and range label */}
            <div className="flex items-center gap-1 ml-2">
              <Button
                onClick={zoomOut}
                disabled={zoomLevel <= 0.25}
                size="sm"
                variant="outline"
                className="text-xs px-2"
                title="Zoom Out"
              >
                <ZoomOut className="h-3 w-3" />
              </Button>
              <span className="text-xs text-muted-foreground px-1">
                {zoomLevel}x
              </span>
              <Button
                onClick={zoomIn}
                disabled={zoomLevel >= 4}
                size="sm"
                variant="outline"
                className="text-xs px-2"
                title="Zoom In"
              >
                <ZoomIn className="h-3 w-3" />
              </Button>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Button
              onClick={handleExport}
              disabled={disabled || prompts.length === 0 || isRecording}
              size="sm"
              variant="outline"
            >
              <Download className="h-4 w-4 mr-1" />
              Export
            </Button>
            <div className="relative">
              <input
                type="file"
                accept=".json"
                onChange={handleImport}
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                disabled={disabled || isRecording}
              />
              <Button
                size="sm"
                variant="outline"
                disabled={disabled || isRecording}
              >
                <Upload className="h-4 w-4 mr-1" />
                Import
              </Button>
            </div>
          </div>
        </div>

        {/* Timeline */}
        <div className="relative overflow-hidden w-full" ref={timelineRef}>
          {/* Time markers */}
          <div className="relative mb-1 w-full" style={{ height: "30px" }}>
            {Array.from(
              {
                length: Math.ceil((visibleEndTime - visibleStartTime) / 10) + 1,
              },
              (_, i) => {
                const time = Math.round(visibleStartTime + i * 10); // Round to integer
                const position = timeToPosition(time);
                return (
                  <div
                    key={i}
                    className="absolute top-0 flex items-center justify-center"
                    style={{
                      left: time === 0 ? position + 10 : position,
                      transform: "translateX(-50%)",
                    }}
                  >
                    <span className="text-gray-400 text-xs">{time}s</span>
                  </div>
                );
              }
            )}
          </div>

          {/* Timeline track */}
          <div
            className="relative bg-muted rounded-lg border overflow-hidden cursor-grab w-full"
            style={{ height: "120px" }} // Increased height for vertical blend display
            onClick={handleTimelineClick}
            onMouseDown={handleTimelineMouseDown}
          >
            {/* Current time cursor */}
            <div
              className="absolute top-0 bottom-0 w-1 bg-red-500 z-30 shadow-lg"
              style={{
                left: Math.max(
                  0,
                  Math.min(timelineWidth, timeToPosition(currentTime))
                ),
                display: "block",
              }}
            />
            {/* Removed debug time overlay */}

            {/* Prompt blocks */}
            {prompts
              .filter(
                prompt =>
                  prompt.endTime >= visibleStartTime &&
                  prompt.startTime <= visibleEndTime
              )
              .sort((a, b) => a.startTime - b.startTime) // Sort by start time
              .map((prompt, index, sortedPrompts) => {
                const isSelected = selectedPromptId === prompt.id;
                const isActive =
                  currentTime >= prompt.startTime &&
                  currentTime <= prompt.endTime;
                const isRecordingActive =
                  isRecording && currentTime >= prompt.startTime;
                const isLive = prompt.isLive;

                // Use the prompt's color, only generate if it doesn't exist
                let boxColor = prompt.color;
                if (!boxColor) {
                  // Only generate color if the prompt doesn't have one
                  const adjacentColors: string[] = [];
                  if (index > 0 && sortedPrompts[index - 1].color) {
                    adjacentColors.push(sortedPrompts[index - 1].color!);
                  }
                  if (
                    index < sortedPrompts.length - 1 &&
                    sortedPrompts[index + 1].color
                  ) {
                    adjacentColors.push(sortedPrompts[index + 1].color!);
                  }
                  boxColor = generateRandomColor(adjacentColors);

                  // Update the prompt with the new color to persist it
                  const updatedPrompt = { ...prompt, color: boxColor };
                  const updatedPrompts = prompts.map(p =>
                    p.id === prompt.id ? updatedPrompt : p
                  );
                  onPromptsChange(updatedPrompts);
                }

                // Calculate position - boxes should be adjacent with no gaps
                let leftPosition = Math.max(
                  0,
                  timeToPosition(prompt.startTime)
                );

                // If this is not the first prompt, position it right after the previous one
                if (index > 0) {
                  const previousPrompt = sortedPrompts[index - 1];
                  const previousEndPosition = Math.max(
                    0,
                    timeToPosition(previousPrompt.endTime)
                  );
                  leftPosition = Math.max(leftPosition, previousEndPosition);
                }

                const prevPrompt =
                  index > 0 ? sortedPrompts[index - 1] : undefined;
                const nextPrompt =
                  index < sortedPrompts.length - 1
                    ? sortedPrompts[index + 1]
                    : undefined;

                return (
                  <div
                    key={prompt.id}
                    className={`absolute border rounded px-2 py-1 transition-colors cursor-pointer ${
                      isSelected
                        ? "shadow-lg border-blue-500"
                        : isActive
                          ? "border-green-500"
                          : isRecordingActive
                            ? "border-red-500"
                            : ""
                    }`}
                    style={{
                      left: leftPosition,
                      top: "8px", // Position from top
                      bottom: "8px", // Position from bottom
                      width: Math.min(
                        timelineWidth - leftPosition,
                        timeToPosition(prompt.endTime) - leftPosition
                      ),
                      backgroundColor: isLive ? "#6B7280" : boxColor, // Grey for live boxes, random color for completed
                      borderColor: isLive ? "#9CA3AF" : boxColor,
                    }}
                    onClick={e => handlePromptClick(e, prompt)}
                  >
                    {/* Resize handles */}
                    {!isRecording && !isLive && (
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
                          // Display blend prompts vertically
                          prompt.prompts.map((promptItem, idx) => (
                            <div
                              key={idx}
                              className="text-xs text-white font-medium truncate"
                            >
                              {promptItem.text} ({promptItem.weight}%)
                            </div>
                          ))
                        ) : (
                          // Single prompt display
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
      </CardContent>
    </Card>
  );
}
