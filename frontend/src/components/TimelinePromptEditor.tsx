import { useState, useEffect, useRef, useMemo } from "react";

import { Button } from "./ui/button";
import { Plus } from "lucide-react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./ui/select";

import type { TimelinePrompt } from "./PromptTimeline";
import { usePromptManager } from "../hooks/usePromptManager";
import { PromptField } from "./shared/PromptField";
import { WeightSlider } from "./shared/WeightSlider";
import { TemporalTransitionControls } from "./shared/TemporalTransitionControls";

interface TimelinePromptEditorProps {
  className?: string;
  prompt: TimelinePrompt | null;
  onPromptUpdate?: (prompt: TimelinePrompt) => void;
  disabled?: boolean;
  interpolationMethod?: "linear" | "slerp";
  onInterpolationMethodChange?: (method: "linear" | "slerp") => void;
  promptIndex?: number;
  defaultTemporalInterpolationMethod?: "linear" | "slerp" | null;
  defaultSpatialInterpolationMethod?: "linear" | "slerp" | null;
}

const MAX_PROMPTS = 4;
const DEFAULT_WEIGHT = 100;

export function TimelinePromptEditor({
  className = "",
  prompt,
  onPromptUpdate,
  disabled = false,
  interpolationMethod = "linear",
  onInterpolationMethodChange,
  promptIndex,
  defaultTemporalInterpolationMethod,
  defaultSpatialInterpolationMethod,
}: TimelinePromptEditorProps) {
  // Derive support from null check - null means feature not supported
  const supportsTemporalInterpolation =
    defaultTemporalInterpolationMethod !== null;
  const supportsSpatialInterpolation =
    defaultSpatialInterpolationMethod !== null;
  const [focusedIndex, setFocusedIndex] = useState<number | null>(null);
  const lastSyncedPromptIdRef = useRef<string | null>(null);
  const isInternalUpdateRef = useRef(false);

  // Derive prompts from a TimelinePrompt
  const getPromptsFromTimelinePrompt = (
    timelinePrompt: TimelinePrompt | null
  ) => {
    if (!timelinePrompt) return [];
    if (timelinePrompt.prompts && timelinePrompt.prompts.length > 0) {
      return timelinePrompt.prompts;
    }
    return [{ text: timelinePrompt.text, weight: DEFAULT_WEIGHT }];
  };

  // Initialize prompts from prompt prop (memoized to avoid recalculation)
  const initialPrompts = useMemo(
    () => getPromptsFromTimelinePrompt(prompt),
    [prompt]
  );

  // Use shared prompt management hook (uncontrolled mode)
  const {
    prompts,
    handlePromptTextChange,
    handleWeightChange,
    handleAddPrompt,
    handleRemovePrompt,
    normalizedWeights,
    setPrompts,
  } = usePromptManager({
    initialPrompts: initialPrompts,
    maxPrompts: MAX_PROMPTS,
    defaultWeight: DEFAULT_WEIGHT,
    onPromptsChange: newPrompts => {
      if (prompt) {
        isInternalUpdateRef.current = true;
        const updatedPrompt = {
          ...prompt,
          text: newPrompts.length === 1 ? newPrompts[0].text : "",
          prompts: newPrompts.length > 1 ? newPrompts : undefined,
        };
        onPromptUpdate?.(updatedPrompt);
        // Reset flag in next tick
        requestAnimationFrame(() => {
          isInternalUpdateRef.current = false;
        });
      }
    },
  });

  // Sync prompts when prompt prop changes (external update only)
  useEffect(() => {
    // Skip if this is an internal update from user interaction
    if (isInternalUpdateRef.current) {
      return;
    }

    if (!prompt) {
      // Only clear if prompts aren't already empty
      if (prompts.length > 0) {
        lastSyncedPromptIdRef.current = null;
        setPrompts([]);
      }
      return;
    }

    // Skip if we've already synced this prompt
    if (lastSyncedPromptIdRef.current === prompt.id) {
      return;
    }

    const expectedPrompts = getPromptsFromTimelinePrompt(prompt);

    // Compare prompts to avoid unnecessary updates
    const promptsMatch =
      prompts.length === expectedPrompts.length &&
      prompts.every(
        (p, i) =>
          p.text === expectedPrompts[i].text &&
          Math.abs(p.weight - expectedPrompts[i].weight) < 0.001
      );

    if (!promptsMatch) {
      lastSyncedPromptIdRef.current = prompt.id;
      setPrompts(expectedPrompts);
    } else {
      // Mark as synced even if they match
      lastSyncedPromptIdRef.current = prompt.id;
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [prompt?.id]);

  // Check if this is the first block (can't transition from nothing)
  const isFirstBlock = promptIndex === 0;

  // Automatically switch to linear interpolation when there are more than 2 prompts
  // SLERP only works with exactly 2 prompts
  useEffect(() => {
    if (prompts.length > 2 && interpolationMethod === "slerp") {
      onInterpolationMethodChange?.("linear");
    }
  }, [prompts.length, interpolationMethod, onInterpolationMethodChange]);

  const isSinglePrompt = prompts.length === 1;

  const handleTransitionStepsChange = (steps: number) => {
    if (prompt) {
      const updatedPrompt = {
        ...prompt,
        transitionSteps: steps,
      };
      onPromptUpdate?.(updatedPrompt);
    }
  };

  const handleTemporalInterpolationMethodChange = (
    method: "linear" | "slerp"
  ) => {
    if (prompt) {
      const updatedPrompt = {
        ...prompt,
        temporalInterpolationMethod: method,
      };
      onPromptUpdate?.(updatedPrompt);
    }
  };

  // Render transition settings
  const renderTransitionSettings = () => {
    const effectiveTransitionSteps = isFirstBlock
      ? 0
      : (prompt?.transitionSteps ?? 0);
    const effectiveTemporalMethod =
      prompt?.temporalInterpolationMethod ?? "slerp";

    return (
      <TemporalTransitionControls
        transitionSteps={effectiveTransitionSteps}
        onTransitionStepsChange={handleTransitionStepsChange}
        temporalInterpolationMethod={effectiveTemporalMethod}
        onTemporalInterpolationMethodChange={
          handleTemporalInterpolationMethodChange
        }
        disabled={disabled || isFirstBlock}
        showHeader={false}
        showDisabledMessage={false}
        maxSteps={16}
        className="space-y-2"
      />
    );
  };
  // Render single prompt mode
  const renderSinglePrompt = () => {
    return (
      <div className={`space-y-3 ${className}`}>
        <div className="flex items-start bg-card border border-border rounded-lg px-4 py-3 gap-3">
          <PromptField
            prompt={prompts[0]}
            index={0}
            placeholder="Edit prompt..."
            showRemove={false}
            focusedIndex={focusedIndex}
            onTextChange={handlePromptTextChange}
            onFocus={setFocusedIndex}
            onBlur={() => setFocusedIndex(null)}
            onRemove={handleRemovePrompt}
            disabled={disabled}
          />
        </div>

        <div className="space-y-2">
          {supportsTemporalInterpolation && renderTransitionSettings()}

          {supportsSpatialInterpolation && prompts.length < 4 && (
            <div className="flex items-center justify-end gap-2">
              <Button
                onMouseDown={e => {
                  e.preventDefault();
                  handleAddPrompt();
                }}
                disabled={disabled}
                size="sm"
                variant="ghost"
                className="rounded-full w-8 h-8 p-0"
              >
                <Plus className="h-4 w-4" />
              </Button>
            </div>
          )}
        </div>
      </div>
    );
  };

  // Render multiple prompts mode
  const renderMultiplePrompts = () => {
    return (
      <div className={`space-y-3 ${className}`}>
        {prompts.map((promptItem, index) => {
          return (
            <div key={index} className="space-y-2">
              <div className="flex items-start bg-card border border-border rounded-lg px-4 py-3 gap-3">
                <PromptField
                  prompt={promptItem}
                  index={index}
                  placeholder={`Prompt ${index + 1}`}
                  showRemove={true}
                  focusedIndex={focusedIndex}
                  onTextChange={handlePromptTextChange}
                  onFocus={setFocusedIndex}
                  onBlur={() => setFocusedIndex(null)}
                  onRemove={handleRemovePrompt}
                  disabled={disabled}
                />
              </div>

              <WeightSlider
                value={normalizedWeights[index]}
                onValueChange={value => handleWeightChange(index, value)}
                disabled={disabled}
              />
            </div>
          );
        })}

        <div className="space-y-2">
          {/* Spatial Blend - only for multiple prompts */}
          {supportsSpatialInterpolation && prompts.length >= 2 && (
            <div className="flex items-center justify-between gap-2">
              <span className="text-xs text-muted-foreground">
                Spatial Blend:
              </span>
              <Select
                value={interpolationMethod}
                onValueChange={value =>
                  onInterpolationMethodChange?.(value as "linear" | "slerp")
                }
                disabled={disabled}
              >
                <SelectTrigger className="w-24 h-6 text-xs">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="linear">Linear</SelectItem>
                  <SelectItem value="slerp" disabled={prompts.length > 2}>
                    Slerp
                  </SelectItem>
                </SelectContent>
              </Select>
            </div>
          )}

          {supportsTemporalInterpolation && renderTransitionSettings()}

          {/* Add button - Bottom row */}
          {supportsSpatialInterpolation && prompts.length < 4 && (
            <div className="flex items-center justify-end gap-2">
              <Button
                onMouseDown={e => {
                  e.preventDefault();
                  handleAddPrompt();
                }}
                disabled={disabled}
                size="sm"
                variant="ghost"
                className="rounded-full w-8 h-8 p-0"
              >
                <Plus className="h-4 w-4" />
              </Button>
            </div>
          )}
        </div>
      </div>
    );
  };

  // Render component based on state
  if (!prompt) {
    return (
      <div className={`text-center text-muted-foreground py-8 ${className}`}>
        Click on a prompt box in the timeline to edit it
      </div>
    );
  }

  return isSinglePrompt ? renderSinglePrompt() : renderMultiplePrompts();
}
