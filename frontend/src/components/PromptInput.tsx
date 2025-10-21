import { useState, useEffect } from "react";
import { Input } from "./ui/input";
import { Textarea } from "./ui/textarea";
import { Button } from "./ui/button";
import { Slider } from "./ui/slider";
import { ArrowUp, Plus, X } from "lucide-react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./ui/select";
import type { PromptItem } from "../lib/api";

interface PromptInputProps {
  className?: string;
  prompts: PromptItem[];
  onPromptsChange?: (prompts: PromptItem[]) => void;
  onPromptsSubmit?: (prompts: PromptItem[]) => void;
  disabled?: boolean;
  interpolationMethod?: "linear" | "slerp";
  onInterpolationMethodChange?: (method: "linear" | "slerp") => void;
  isRecording?: boolean;
  onRecordingPromptSubmit?: (prompts: PromptItem[]) => void;
  showTimeline?: boolean;
  onAddToTimeline?: (prompts: PromptItem[]) => void;
}

export function PromptInput({
  className = "",
  prompts,
  onPromptsChange,
  onPromptsSubmit,
  disabled = false,
  interpolationMethod = "linear",
  onInterpolationMethodChange,
  isRecording = false,
  onRecordingPromptSubmit,
  showTimeline = false,
  onAddToTimeline,
}: PromptInputProps) {
  const [isProcessing, setIsProcessing] = useState(false);
  const [focusedIndex, setFocusedIndex] = useState<number | null>(null);
  const [localPrompts, setLocalPrompts] = useState<PromptItem[]>([]);

  // Sync local prompts with props
  useEffect(() => {
    setLocalPrompts(prompts);
  }, [prompts]);

  // Automatically switch to linear interpolation when there are more than 2 prompts
  // SLERP only works with exactly 2 prompts
  // TODO: When toasts are added to the project, show a warning toast when auto-switching
  // from slerp to linear (e.g., "Switched to linear interpolation: Slerp requires exactly 2 prompts")
  useEffect(() => {
    if (localPrompts.length > 2 && interpolationMethod === "slerp") {
      onInterpolationMethodChange?.("linear");
    }
  }, [localPrompts.length, interpolationMethod, onInterpolationMethodChange]);

  const handlePromptTextChange = (index: number, text: string) => {
    const newPrompts = [...localPrompts];
    newPrompts[index] = { ...newPrompts[index], text };
    setLocalPrompts(newPrompts);
    onPromptsChange?.(newPrompts);
  };

  const handleWeightChange = (index: number, weight: number) => {
    const newPrompts = [...localPrompts];
    newPrompts[index] = { ...newPrompts[index], weight };
    setLocalPrompts(newPrompts);
    onPromptsChange?.(newPrompts);
  };

  const handleAddPrompt = () => {
    if (localPrompts.length < 4) {
      const newPrompts = [...localPrompts, { text: "", weight: 100 }];
      setLocalPrompts(newPrompts);
      onPromptsChange?.(newPrompts);
    }
  };

  const handleRemovePrompt = (index: number) => {
    if (localPrompts.length > 1) {
      const newPrompts = localPrompts.filter((_, i) => i !== index);
      setLocalPrompts(newPrompts);
      onPromptsChange?.(newPrompts);
    }
  };

  const handleSubmit = () => {
    const validPrompts = localPrompts.filter(p => p.text.trim());
    if (!validPrompts.length) return;

    setIsProcessing(true);

    if (isRecording && onRecordingPromptSubmit) {
      // During recording, submit the full prompt blend to the timeline
      onRecordingPromptSubmit(validPrompts);
      // Don't clear prompts during recording - keep the blend for next submission
      setTimeout(() => {
        setIsProcessing(false);
      }, 1000);
    } else if (showTimeline && onAddToTimeline) {
      // When timeline is shown and not recording, add prompts to timeline
      onAddToTimeline(validPrompts);
      // Keep the prompts unchanged after adding to timeline
      // Reset processing state immediately to keep input enabled
      setIsProcessing(false);
      // Force focus back to input field
      setTimeout(() => {
        setFocusedIndex(0);
      }, 0);
    } else {
      // Normal mode, submit all prompts
      onPromptsSubmit?.(validPrompts);
      // In normal mode, we can clear the prompts after submission
      // But we'll let the parent component handle this
      setTimeout(() => {
        setIsProcessing(false);
      }, 1000);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  // Calculate normalized weights for display
  const totalWeight = localPrompts.reduce((sum, p) => sum + p.weight, 0);
  const normalizedWeights = localPrompts.map(p =>
    totalWeight > 0 ? (p.weight / totalWeight) * 100 : 0
  );

  const isSinglePrompt = localPrompts.length === 1;

  // Render a single prompt field with expandable textarea
  const renderPromptField = (
    index: number,
    placeholder: string,
    showRemove: boolean
  ) => {
    const isFocused = focusedIndex === index;
    const prompt = localPrompts[index];

    return (
      <>
        {isFocused ? (
          <Textarea
            placeholder={placeholder}
            value={prompt.text}
            onChange={e => handlePromptTextChange(index, e.target.value)}
            onKeyDown={handleKeyDown}
            onFocus={() => setFocusedIndex(index)}
            onBlur={() => setFocusedIndex(null)}
            disabled={disabled}
            autoFocus
            className="flex-1 min-h-[80px] resize-none bg-transparent border-0 text-card-foreground placeholder:text-muted-foreground focus-visible:ring-0 focus-visible:ring-offset-0 p-0 disabled:opacity-50 disabled:cursor-not-allowed"
          />
        ) : (
          <Input
            placeholder={placeholder}
            value={prompt.text}
            onChange={e => handlePromptTextChange(index, e.target.value)}
            onKeyDown={handleKeyDown}
            onFocus={() => setFocusedIndex(index)}
            disabled={disabled}
            className="flex-1 bg-transparent border-0 text-card-foreground placeholder:text-muted-foreground focus-visible:ring-0 focus-visible:ring-offset-0 p-0 disabled:opacity-50 disabled:cursor-not-allowed"
          />
        )}
        <Button
          onClick={handleSubmit}
          disabled={
            disabled || !localPrompts.some(p => p.text.trim()) || isProcessing
          }
          size="sm"
          className="rounded-full w-8 h-8 p-0 bg-black hover:bg-gray-800 text-white disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isProcessing ? "..." : <ArrowUp className="h-4 w-4" />}
        </Button>
        {index === localPrompts.length - 1 && localPrompts.length < 4 && (
          <Button
            onClick={handleAddPrompt}
            disabled={disabled}
            size="sm"
            variant="ghost"
            className="rounded-full w-8 h-8 p-0"
          >
            <Plus className="h-4 w-4" />
          </Button>
        )}
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

  // Single prompt mode: simple pill UI
  if (isSinglePrompt) {
    const isFocused = focusedIndex === 0;
    return (
      <div
        className={`flex items-start bg-card border border-border px-4 py-3 gap-3 transition-all ${
          isFocused ? "rounded-lg" : "rounded-full"
        } ${className}`}
      >
        {renderPromptField(0, "blooming flowers", false)}
      </div>
    );
  }

  // Multiple prompts mode: show weights and controls
  return (
    <div className={`space-y-3 ${className}`}>
      {localPrompts.map((prompt, index) => {
        const isFocused = focusedIndex === index;
        return (
          <div key={index} className="space-y-2">
            <div
              className={`flex items-start bg-card border border-border px-4 py-3 gap-3 transition-all ${
                isFocused ? "rounded-lg" : "rounded-full"
              }`}
            >
              {renderPromptField(index, `Prompt ${index + 1}`, true)}
            </div>

            <div className="flex items-center gap-3 px-4">
              <span className="text-xs text-muted-foreground w-12">
                Weight:
              </span>
              <Slider
                value={[prompt.weight]}
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

      {localPrompts.length >= 2 && (
        <div className="flex items-center gap-2 px-4">
          <span className="text-xs text-muted-foreground">Blend:</span>
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
              <SelectItem value="slerp" disabled={localPrompts.length > 2}>
                Slerp
              </SelectItem>
            </SelectContent>
          </Select>
        </div>
      )}
    </div>
  );
}
