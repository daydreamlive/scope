import { useState, useEffect } from "react";

import { Textarea } from "./ui/textarea";
import { Button } from "./ui/button";
import { Slider } from "./ui/slider";
import { Plus, X } from "lucide-react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./ui/select";

import type { TimelinePrompt } from "./PromptTimeline";
import {
  redistributeWeightsOnAdd,
  redistributeWeightsOnRemove,
} from "../utils/promptWeights";

interface TimelinePromptEditorProps {
  className?: string;
  prompt: TimelinePrompt | null;
  onPromptUpdate?: (prompt: TimelinePrompt) => void;
  disabled?: boolean;
  interpolationMethod?: "linear" | "slerp";
  onInterpolationMethodChange?: (method: "linear" | "slerp") => void;
  promptIndex?: number;
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
}: TimelinePromptEditorProps) {
  const [editingPrompt, setEditingPrompt] = useState<TimelinePrompt | null>(
    null
  );
  const [prompts, setPrompts] = useState<
    Array<{ text: string; weight: number }>
  >([]);
  const [focusedIndex, setFocusedIndex] = useState<number | null>(null);

  // Check if this is the first block (can't transition from nothing)
  const isFirstBlock = promptIndex === 0;

  // Automatically switch to linear interpolation when there are more than 2 prompts
  // SLERP only works with exactly 2 prompts
  useEffect(() => {
    if (prompts.length > 2 && interpolationMethod === "slerp") {
      onInterpolationMethodChange?.("linear");
    }
  }, [prompts.length, interpolationMethod, onInterpolationMethodChange]);

  // Initialize editing prompt when prompt changes
  useEffect(() => {
    if (prompt) {
      // Ensure first block always has transitionSteps=0
      const normalizedPrompt =
        isFirstBlock && prompt.transitionSteps !== 0
          ? { ...prompt, transitionSteps: 0 }
          : prompt;

      setEditingPrompt(normalizedPrompt);
      if (normalizedPrompt.prompts && normalizedPrompt.prompts.length > 0) {
        setPrompts(normalizedPrompt.prompts);
      } else {
        setPrompts([{ text: normalizedPrompt.text, weight: DEFAULT_WEIGHT }]);
      }

      // Update parent if we normalized the first block
      if (
        isFirstBlock &&
        prompt.transitionSteps !== 0 &&
        normalizedPrompt !== prompt
      ) {
        onPromptUpdate?.(normalizedPrompt);
      }
    } else {
      setEditingPrompt(null);
      setPrompts([]);
    }
  }, [prompt, promptIndex, onPromptUpdate]);

  // Update prompt text
  const handlePromptTextChange = (index: number, text: string) => {
    const newPrompts = [...prompts];
    newPrompts[index] = { ...newPrompts[index], text };
    setPrompts(newPrompts);

    if (editingPrompt) {
      const updatedPrompt = {
        ...editingPrompt,
        text: newPrompts.length === 1 ? newPrompts[0].text : "",
        prompts: newPrompts.length > 1 ? newPrompts : undefined,
      };
      setEditingPrompt(updatedPrompt);
      onPromptUpdate?.(updatedPrompt);
    }
  };

  // Update prompt weight
  const handleWeightChange = (index: number, normalizedWeight: number) => {
    const newPrompts = [...prompts];

    // Calculate the remaining weight to distribute among other prompts
    const remainingWeight = 100 - normalizedWeight;

    // Get the sum of other prompts' current weights (excluding the changed one)
    const otherWeightsSum = prompts.reduce(
      (sum, p, i) => (i === index ? sum : sum + p.weight),
      0
    );

    // Update the changed prompt's weight
    newPrompts[index] = { ...newPrompts[index], weight: normalizedWeight };

    // Redistribute remaining weight proportionally to other prompts
    if (otherWeightsSum > 0) {
      newPrompts.forEach((_, i) => {
        if (i !== index) {
          const proportion = prompts[i].weight / otherWeightsSum;
          newPrompts[i] = {
            ...newPrompts[i],
            weight: remainingWeight * proportion,
          };
        }
      });
    } else {
      // If all other weights are 0, distribute evenly
      const evenWeight = remainingWeight / (prompts.length - 1);
      newPrompts.forEach((_, i) => {
        if (i !== index) {
          newPrompts[i] = { ...newPrompts[i], weight: evenWeight };
        }
      });
    }

    setPrompts(newPrompts);

    if (editingPrompt) {
      const updatedPrompt = {
        ...editingPrompt,
        prompts: newPrompts,
      };
      setEditingPrompt(updatedPrompt);
      onPromptUpdate?.(updatedPrompt);
    }
  };

  // Add new prompt
  const handleAddPrompt = () => {
    if (prompts.length < MAX_PROMPTS) {
      const newPrompts = redistributeWeightsOnAdd(prompts);
      setPrompts(newPrompts);

      if (editingPrompt) {
        const updatedPrompt = {
          ...editingPrompt,
          prompts: newPrompts,
        };
        setEditingPrompt(updatedPrompt);
        onPromptUpdate?.(updatedPrompt);
      }
    }
  };

  // Remove prompt
  const handleRemovePrompt = (index: number) => {
    if (prompts.length > 1) {
      const redistributed = redistributeWeightsOnRemove(prompts, index);
      setPrompts(redistributed);

      if (editingPrompt) {
        const updatedPrompt = {
          ...editingPrompt,
          text: redistributed.length === 1 ? redistributed[0].text : "",
          prompts: redistributed.length > 1 ? redistributed : undefined,
        };
        setEditingPrompt(updatedPrompt);
        onPromptUpdate?.(updatedPrompt);
      }
    }
  };

  const handleTransitionStepsChange = (steps: number) => {
    if (editingPrompt) {
      const updatedPrompt = {
        ...editingPrompt,
        transitionSteps: steps,
      };
      setEditingPrompt(updatedPrompt);
      onPromptUpdate?.(updatedPrompt);
    }
  };

  const handleTemporalInterpolationMethodChange = (
    method: "linear" | "slerp"
  ) => {
    if (editingPrompt) {
      const updatedPrompt = {
        ...editingPrompt,
        temporalInterpolationMethod: method,
      };
      setEditingPrompt(updatedPrompt);
      onPromptUpdate?.(updatedPrompt);
    }
  };

  // Calculate normalized weights for display
  const totalWeight = prompts.reduce((sum, p) => sum + p.weight, 0);
  const normalizedWeights = prompts.map(p =>
    totalWeight > 0 ? (p.weight / totalWeight) * 100 : 0
  );

  const isSinglePrompt = prompts.length === 1;

  // Render prompt field with ellipsis when collapsed
  const renderPromptField = (
    index: number,
    placeholder: string,
    showRemove: boolean
  ) => {
    const isFocused = focusedIndex === index;
    const promptItem = prompts[index];

    return (
      <>
        <Textarea
          placeholder={placeholder}
          value={promptItem.text}
          onChange={e => handlePromptTextChange(index, e.target.value)}
          onFocus={() => setFocusedIndex(index)}
          onBlur={() => setFocusedIndex(null)}
          disabled={disabled}
          rows={isFocused ? 3 : 1}
          className={`flex-1 resize-none bg-transparent border-0 text-card-foreground placeholder:text-muted-foreground focus-visible:ring-0 focus-visible:ring-offset-0 p-0 disabled:opacity-50 disabled:cursor-not-allowed ${
            isFocused
              ? "min-h-[80px]"
              : "min-h-[24px] overflow-hidden whitespace-nowrap text-ellipsis"
          }`}
        />
        {showRemove && (
          <Button
            onClick={() => handleRemovePrompt(index)}
            disabled={disabled}
            size="sm"
            variant="ghost"
            className="rounded-full w-8 h-8 p-0"
          >
            <X className="h-4 w-4" />
          </Button>
        )}
      </>
    );
  };

  const renderTransitionSettings = () => {
    const effectiveTransitionSteps = isFirstBlock
      ? 0
      : (editingPrompt?.transitionSteps ?? 0);
    const effectiveTemporalMethod =
      editingPrompt?.temporalInterpolationMethod ?? "slerp";

    return (
      <div className="space-y-3 pt-3 border-t border-border">
        <div className="text-xs font-medium text-muted-foreground">
          Temporal Transition Settings
        </div>

        <div className="flex items-center gap-3">
          <span className="text-xs text-muted-foreground w-16">Steps:</span>
          <Slider
            value={[effectiveTransitionSteps]}
            onValueChange={([value]) => handleTransitionStepsChange(value)}
            min={0}
            max={10}
            step={1}
            disabled={disabled || isFirstBlock}
            className="flex-1"
          />
          <span className="text-xs text-muted-foreground w-8 text-right">
            {effectiveTransitionSteps}
          </span>
        </div>

        <div className="flex items-center gap-3">
          <span className="text-xs text-muted-foreground w-16">Method:</span>
          <Select
            value={effectiveTemporalMethod}
            onValueChange={value =>
              handleTemporalInterpolationMethodChange(
                value as "linear" | "slerp"
              )
            }
            disabled={disabled || isFirstBlock}
          >
            <SelectTrigger className="flex-1 h-7 text-xs">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="linear">Linear</SelectItem>
              <SelectItem value="slerp">Slerp</SelectItem>
            </SelectContent>
          </Select>
        </div>

        {isFirstBlock && (
          <div className="text-xs text-muted-foreground italic">
            First block cannot have transitions
          </div>
        )}
      </div>
    );
  };

  // Render single prompt mode
  const renderSinglePrompt = () => {
    return (
      <div className={`space-y-3 ${className}`}>
        <div className="flex items-start bg-card border border-border rounded-lg px-4 py-3 gap-3">
          {renderPromptField(0, "Edit prompt...", false)}
        </div>

        {prompts.length < 4 && (
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

        {renderTransitionSettings()}
      </div>
    );
  };

  // Render multiple prompts mode
  const renderMultiplePrompts = () => {
    return (
      <div className={`space-y-3 ${className}`}>
        {prompts.map((_, index) => {
          return (
            <div key={index} className="space-y-2">
              <div className="flex items-start bg-card border border-border rounded-lg px-4 py-3 gap-3">
                {renderPromptField(index, `Prompt ${index + 1}`, true)}
              </div>

              <div className="flex items-center gap-3">
                <span className="text-xs text-muted-foreground w-12">
                  Weight:
                </span>
                <Slider
                  value={[normalizedWeights[index]]}
                  onValueChange={([value]) => handleWeightChange(index, value)}
                  min={0}
                  max={100}
                  step={1}
                  disabled={disabled}
                  className="flex-1"
                />
                <span className="text-xs text-muted-foreground w-12 text-right">
                  {normalizedWeights[index].toFixed(0)}%
                </span>
              </div>
            </div>
          );
        })}

        <div className="flex items-center justify-between gap-2">
          {prompts.length >= 2 ? (
            <div className="flex items-center gap-2">
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
                <SelectTrigger className="w-24 h-7 text-xs">
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
          ) : (
            <div />
          )}

          {prompts.length < 4 && (
            <div className="flex items-center gap-2">
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

        {renderTransitionSettings()}
      </div>
    );
  };

  // Render component based on state
  if (!editingPrompt) {
    return (
      <div className={`text-center text-muted-foreground py-8 ${className}`}>
        Click on a prompt box in the timeline to edit it
      </div>
    );
  }

  return isSinglePrompt ? renderSinglePrompt() : renderMultiplePrompts();
}
