import { useState, useEffect } from "react";
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
  isLive?: boolean;
  onLivePromptSubmit?: (prompts: PromptItem[]) => void;
}

export function PromptInput({
  className = "",
  prompts,
  onPromptsChange,
  onPromptsSubmit,
  disabled = false,
  interpolationMethod = "linear",
  onInterpolationMethodChange,
  isLive = false,
  onLivePromptSubmit,
}: PromptInputProps) {
  const [isProcessing, setIsProcessing] = useState(false);
  const [focusedIndex, setFocusedIndex] = useState<number | null>(null);

  // Automatically switch to linear interpolation when there are more than 2 prompts
  // SLERP only works with exactly 2 prompts
  // TODO: When toasts are added to the project, show a warning toast when auto-switching
  // from slerp to linear (e.g., "Switched to linear interpolation: Slerp requires exactly 2 prompts")
  useEffect(() => {
    if (prompts.length > 2 && interpolationMethod === "slerp") {
      onInterpolationMethodChange?.("linear");
    }
  }, [prompts.length, interpolationMethod, onInterpolationMethodChange]);

  const handlePromptTextChange = (index: number, text: string) => {
    const newPrompts = [...prompts];
    newPrompts[index] = { ...newPrompts[index], text };
    onPromptsChange?.(newPrompts);
  };

  const handleWeightChange = (index: number, weight: number) => {
    const newPrompts = [...prompts];
    newPrompts[index] = { ...newPrompts[index], weight };
    onPromptsChange?.(newPrompts);
  };

  const handleAddPrompt = () => {
    if (prompts.length < 4) {
      onPromptsChange?.([...prompts, { text: "", weight: 100 }]);
    }
  };

  const handleRemovePrompt = (index: number) => {
    if (prompts.length > 1) {
      const newPrompts = prompts.filter((_, i) => i !== index);
      onPromptsChange?.(newPrompts);
    }
  };

  const handleSubmit = () => {
    const validPrompts = prompts.filter(p => p.text.trim());
    if (!validPrompts.length) return;

    setIsProcessing(true);
    onPromptsSubmit?.(validPrompts);

    // Always call onLivePromptSubmit if available, regardless of isLive state
    // The timeline component will handle live state management internally
    if (onLivePromptSubmit) {
      // Submit the full prompt blend to the timeline
      onLivePromptSubmit(validPrompts);
      // Don't clear prompts during live mode - keep the blend for next submission
    } else if (!isLive) {
      // Normal mode, submit all prompts (fallback for when onLivePromptSubmit is not available)
      onPromptsSubmit?.(validPrompts);
      // In normal mode, we can clear the prompts after submission
      // But we'll let the parent component handle this
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

  // Calculate normalized weights for display
  const totalWeight = prompts.reduce((sum, p) => sum + p.weight, 0);
  const normalizedWeights = prompts.map(p =>
    totalWeight > 0 ? (p.weight / totalWeight) * 100 : 0
  );

  const isSinglePrompt = prompts.length === 1;

  // Render a single prompt field with expandable textarea
  const renderPromptField = (
    index: number,
    placeholder: string,
    showRemove: boolean
  ) => {
    const isFocused = focusedIndex === index;
    const prompt = prompts[index];

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
            rows={3}
            className="flex-1 resize-none bg-transparent border-0 text-card-foreground placeholder:text-muted-foreground focus-visible:ring-0 focus-visible:ring-offset-0 p-0 disabled:opacity-50 disabled:cursor-not-allowed min-h-[80px]"
            autoFocus
          />
        ) : (
          <div
            onClick={() => !disabled && setFocusedIndex(index)}
            className={`flex-1 min-h-[24px] cursor-text overflow-hidden whitespace-nowrap text-ellipsis text-base md:text-sm ${
              disabled ? "opacity-50 cursor-not-allowed" : ""
            } ${prompt.text ? "text-card-foreground" : "text-muted-foreground"}`}
          >
            {prompt.text || placeholder}
          </div>
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
    return (
      <div className={`space-y-3 ${className}`}>
        <div className="flex items-start bg-card border border-border rounded-lg px-4 py-3 gap-3">
          {renderPromptField(0, "blooming flowers", false)}
        </div>

        <div className="flex items-center justify-end gap-2">
          {prompts.length < 4 && (
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
              disabled || !prompts.some(p => p.text.trim()) || isProcessing
            }
            size="sm"
            className="rounded-full w-8 h-8 p-0 bg-black hover:bg-gray-800 text-white disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isProcessing ? "..." : <ArrowUp className="h-4 w-4" />}
          </Button>
        </div>
      </div>
    );
  }

  // Multiple prompts mode: show weights and controls
  return (
    <div className={`space-y-3 ${className}`}>
      {prompts.map((prompt, index) => {
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

      <div className="flex items-center justify-between gap-2">
        {prompts.length >= 2 ? (
          <div className="flex items-center gap-2">
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
                <SelectItem value="slerp" disabled={prompts.length > 2}>
                  Slerp
                </SelectItem>
              </SelectContent>
            </Select>
          </div>
        ) : (
          <div />
        )}

        <div className="flex items-center gap-2">
          {prompts.length < 4 && (
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
              disabled || !prompts.some(p => p.text.trim()) || isProcessing
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
