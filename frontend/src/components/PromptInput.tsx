import { useState, useEffect } from "react";
import { Button } from "./ui/button";
import { ArrowUp, Plus } from "lucide-react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./ui/select";
import type { PromptItem, PromptTransition } from "../lib/api";
import type { TimelinePrompt } from "./PromptTimeline";
import { usePromptManager } from "../hooks/usePromptManager";
import { PromptField } from "./shared/PromptField";
import { WeightSlider } from "./shared/WeightSlider";
import { TemporalTransitionControls } from "./shared/TemporalTransitionControls";
import { MIDIMappable } from "./MIDIMappable";
import { useMIDI } from "../contexts/MIDIContext";
import { Toggle } from "./ui/toggle";

interface PromptInputProps {
  className?: string;
  prompts: PromptItem[];
  onPromptsChange?: (prompts: PromptItem[]) => void;
  onPromptsSubmit?: (prompts: PromptItem[]) => void;
  onTransitionSubmit?: (transition: PromptTransition) => void;
  disabled?: boolean;
  interpolationMethod?: "linear" | "slerp";
  onInterpolationMethodChange?: (method: "linear" | "slerp") => void;
  temporalInterpolationMethod?: "linear" | "slerp";
  onTemporalInterpolationMethodChange?: (method: "linear" | "slerp") => void;
  isLive?: boolean;
  onLivePromptSubmit?: (prompts: PromptItem[]) => void;
  isStreaming?: boolean;
  transitionSteps?: number;
  onTransitionStepsChange?: (steps: number) => void;
  timelinePrompts?: TimelinePrompt[];
  defaultTemporalInterpolationMethod?: "linear" | "slerp" | null;
  defaultSpatialInterpolationMethod?: "linear" | "slerp" | null;
}

export function PromptInput({
  className = "",
  prompts,
  onPromptsChange,
  onPromptsSubmit,
  onTransitionSubmit,
  disabled = false,
  interpolationMethod = "linear",
  onInterpolationMethodChange,
  temporalInterpolationMethod = "slerp",
  onTemporalInterpolationMethodChange,
  isLive: _isLive = false,
  onLivePromptSubmit,
  isStreaming = false,
  transitionSteps = 4,
  onTransitionStepsChange,
  timelinePrompts = [],
  defaultTemporalInterpolationMethod,
  defaultSpatialInterpolationMethod,
}: PromptInputProps) {
  // Derive support from null check - null means feature not supported
  const supportsTemporalInterpolation =
    defaultTemporalInterpolationMethod !== null;
  const supportsSpatialInterpolation =
    defaultSpatialInterpolationMethod !== null;
  const [isProcessing, setIsProcessing] = useState(false);
  const [focusedIndex, setFocusedIndex] = useState<number | null>(null);
  const { midiEnabled } = useMIDI();

  // Use shared prompt management hook
  // This is a controlled component, so we pass prompts directly
  const {
    prompts: managedPrompts,
    setPrompts,
    handlePromptTextChange,
    handleWeightChange,
    handleAddPrompt,
    handleRemovePrompt,
    normalizedWeights,
  } = usePromptManager({
    prompts: prompts, // Controlled component
    maxPrompts: 4,
    defaultWeight: 100,
    onPromptsChange: onPromptsChange,
  });

  // Automatically switch to linear interpolation when there are more than 2 prompts
  // SLERP only works with exactly 2 prompts
  // TODO: When toasts are added to the project, show a warning toast when auto-switching
  // from slerp to linear (e.g., "Switched to linear interpolation: Slerp requires exactly 2 prompts")
  useEffect(() => {
    if (managedPrompts.length > 2 && interpolationMethod === "slerp") {
      onInterpolationMethodChange?.("linear");
    }
  }, [managedPrompts.length, interpolationMethod, onInterpolationMethodChange]);

  const isSoloPrompt = (index: number) =>
    managedPrompts.every((prompt, promptIndex) => {
      if (promptIndex === index) return prompt.weight > 99;
      return prompt.weight < 1;
    });

  const handlePromptSolo = (index: number) => {
    const soloPrompts = managedPrompts.map((prompt, promptIndex) => ({
      ...prompt,
      weight: promptIndex === index ? 100 : 0,
    }));

    setPrompts(soloPrompts);

    if (isStreaming) {
      onLivePromptSubmit?.(soloPrompts);
    }
  };

  const renderPromptSelectButton = (index: number) => {
    if (!midiEnabled) return undefined;

    const isSolo = isSoloPrompt(index);

    return (
      <MIDIMappable
        actionId={`switch_prompt_${index}`}
        mappingType="trigger"
        className="inline-flex h-5 w-5 shrink-0 items-center border border-gray-800 justify-center overflow-hidden rounded-sm leading-none"
        mappingModeClassName="p-2 rounded-md"
        overlayClassName="rounded-md"
      >
        <div className="h-full w-full">
          <Toggle
            pressed={isSolo}
            variant="default"
            className="shrink-0 cursor-pointer rounded-sm  bg-transparent text-[10px] font-semibold uppercase leading-none text-muted-foreground hover:bg-transparent hover:text-foreground data-[state=on]:border-blue-500 data-[state=on]:bg-blue-500 data-[state=on]:text-white data-[state=on]:hover:bg-blue-500"
            onPressedChange={pressed => {
              if (pressed) {
                handlePromptSolo(index);
              }
            }}
            disabled={disabled}
            aria-label={`Solo prompt ${index + 1}`}
            title={`Solo prompt ${index + 1}`}
          >
            S
          </Toggle>
        </div>
      </MIDIMappable>
    );
  };

  type SubmitStrategy = "transition" | "live" | "normal";

  const determineSubmitStrategy = (): SubmitStrategy => {
    if (isStreaming && transitionSteps > 0 && onTransitionSubmit) {
      return "transition";
    }
    if (onLivePromptSubmit) {
      return "live";
    }
    return "normal";
  };

  const handleSubmit = () => {
    const validPrompts = managedPrompts.filter(p => p.text.trim());
    if (!validPrompts.length) return;

    setIsProcessing(true);

    const strategy = determineSubmitStrategy();

    switch (strategy) {
      case "transition":
        // Smooth transition over multiple frames
        onTransitionSubmit?.({
          target_prompts: validPrompts,
          num_steps: transitionSteps,
          temporal_interpolation_method: temporalInterpolationMethod,
        });
        break;

      case "live":
        // Submit to timeline in live mode
        onLivePromptSubmit?.(validPrompts);
        break;

      case "normal":
        // Normal immediate update
        onPromptsSubmit?.(validPrompts);
        break;
    }

    setTimeout(() => {
      setIsProcessing(false);
    }, 1000);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
      // Unfocus the textarea after submission
      if (document.activeElement instanceof HTMLElement) {
        document.activeElement.blur();
      }
    }
  };

  const isSinglePrompt = managedPrompts.length === 1;

  // Single prompt mode: simple pill UI
  if (isSinglePrompt) {
    return (
      <div className={`space-y-3 ${className}`}>
        <div className="flex items-center bg-card border border-border rounded-lg px-4 py-3 gap-3">
          <PromptField
            prompt={managedPrompts[0]}
            index={0}
            placeholder="blooming flowers"
            showRemove={false}
            focusedIndex={focusedIndex}
            onTextChange={handlePromptTextChange}
            onFocus={setFocusedIndex}
            onBlur={() => setFocusedIndex(null)}
            onRemove={handleRemovePrompt}
            onKeyDown={handleKeyDown}
            disabled={disabled}
          />
        </div>

        <div className="space-y-2">
          {supportsTemporalInterpolation && (
            <TemporalTransitionControls
              transitionSteps={transitionSteps}
              onTransitionStepsChange={steps =>
                onTransitionStepsChange?.(steps)
              }
              temporalInterpolationMethod={temporalInterpolationMethod}
              onTemporalInterpolationMethodChange={method =>
                onTemporalInterpolationMethodChange?.(method)
              }
              disabled={
                disabled || !isStreaming || timelinePrompts.length === 0
              }
              className="space-y-2"
            />
          )}

          {/* Add/Submit buttons - Bottom row */}
          <div className="flex items-center justify-end gap-2">
            {supportsSpatialInterpolation && managedPrompts.length < 4 && (
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
            )}
            <Button
              onMouseDown={e => {
                e.preventDefault();
                handleSubmit();
              }}
              disabled={
                disabled ||
                !managedPrompts.some(p => p.text.trim()) ||
                isProcessing
              }
              size="sm"
              className="rounded-full w-8 h-8 p-0 bg-black hover:bg-gray-800 text-white disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isProcessing ? "..." : <ArrowUp className="h-4 w-4" />}
            </Button>
          </div>
        </div>
      </div>
    );
  }

  // Multiple prompts mode: show weights and controls
  return (
    <div className={`space-y-3 ${className}`}>
      {managedPrompts.map((prompt, index) => {
        return (
          <div key={index} className="space-y-2">
            <div className="flex items-center bg-card border border-border rounded-lg px-4 py-3 gap-3">
              <PromptField
                prompt={prompt}
                index={index}
                placeholder={`Prompt ${index + 1}`}
                showRemove={true}
                focusedIndex={focusedIndex}
                onTextChange={handlePromptTextChange}
                onFocus={setFocusedIndex}
                onBlur={() => setFocusedIndex(null)}
                onRemove={handleRemovePrompt}
                onKeyDown={handleKeyDown}
                disabled={disabled}
              />
            </div>

            <div className="flex items-center gap-2">
              <MIDIMappable
                parameterId="prompt_weight"
                arrayIndex={index}
                mappingType="continuous"
                range={{ min: 0, max: 100 }}
                className="flex-1"
              >
                <WeightSlider
                  value={normalizedWeights[index]}
                  onValueChange={value => handleWeightChange(index, value)}
                  disabled={disabled}
                />
              </MIDIMappable>
              {midiEnabled &&
                managedPrompts.length > 1 &&
                renderPromptSelectButton(index)}
            </div>
          </div>
        );
      })}

      <div className="space-y-2">
        {/* Spatial Blend - only for multiple prompts */}
        {supportsSpatialInterpolation && managedPrompts.length >= 2 && (
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
                <SelectItem value="slerp" disabled={managedPrompts.length > 2}>
                  Slerp
                </SelectItem>
              </SelectContent>
            </Select>
          </div>
        )}

        {supportsTemporalInterpolation && (
          <TemporalTransitionControls
            transitionSteps={transitionSteps}
            onTransitionStepsChange={steps => onTransitionStepsChange?.(steps)}
            temporalInterpolationMethod={temporalInterpolationMethod}
            onTemporalInterpolationMethodChange={method =>
              onTemporalInterpolationMethodChange?.(method)
            }
            disabled={disabled || !isStreaming || timelinePrompts.length === 0}
            className="space-y-2"
          />
        )}

        {/* Add/Submit buttons - Bottom row */}
        <div className="flex items-center justify-end gap-2">
          {supportsSpatialInterpolation && managedPrompts.length < 4 && (
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
          )}
          <Button
            onMouseDown={e => {
              e.preventDefault();
              handleSubmit();
            }}
            disabled={
              disabled ||
              !managedPrompts.some(p => p.text.trim()) ||
              isProcessing
            }
            size="sm"
            className="rounded-full w-8 h-8 p-0 bg-black hover:bg-gray-800 text-white disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isProcessing ? "..." : <ArrowUp className="h-4 w-4" />}
          </Button>
        </div>
      </div>
    </div>
  );
}
