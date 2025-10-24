import React, { useState, useEffect } from "react";

import { Input } from "./ui/input";
import { Textarea } from "./ui/textarea";
import { Button } from "./ui/button";
import { Slider } from "./ui/slider";
import { Plus, X } from "lucide-react";

import type { TimelinePrompt } from "./PromptTimeline";

interface TimelinePromptEditorProps {
  className?: string;
  prompt: TimelinePrompt | null;
  onPromptUpdate?: (prompt: TimelinePrompt) => void;
  onPromptSubmit?: (prompt: TimelinePrompt) => void;
  disabled?: boolean;
}

const MAX_PROMPTS = 4;
const DEFAULT_WEIGHT = 100;

export function TimelinePromptEditor({
  className = "",
  prompt,
  onPromptUpdate,
  onPromptSubmit,
  disabled = false,
}: TimelinePromptEditorProps) {
  const [editingPrompt, setEditingPrompt] = useState<TimelinePrompt | null>(
    null
  );
  const [prompts, setPrompts] = useState<
    Array<{ text: string; weight: number }>
  >([]);
  const [focusedIndex, setFocusedIndex] = useState<number | null>(null);

  // Initialize editing prompt when prompt changes
  useEffect(() => {
    if (prompt) {
      setEditingPrompt(prompt);
      if (prompt.prompts && prompt.prompts.length > 0) {
        setPrompts(prompt.prompts);
      } else {
        setPrompts([{ text: prompt.text, weight: DEFAULT_WEIGHT }]);
      }
    } else {
      setEditingPrompt(null);
      setPrompts([]);
    }
  }, [prompt]);

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
  const handleWeightChange = (index: number, weight: number) => {
    const newPrompts = [...prompts];
    newPrompts[index] = { ...newPrompts[index], weight };
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
      const newPrompts = [...prompts, { text: "", weight: DEFAULT_WEIGHT }];
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
      const newPrompts = prompts.filter((_, i) => i !== index);
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
    }
  };

  // Submit prompt
  const handleSubmit = () => {
    if (!editingPrompt) return;

    const validPrompts = prompts.filter(p => p.text.trim());
    if (!validPrompts.length) return;

    const finalPrompt = {
      ...editingPrompt,
      text: validPrompts.length === 1 ? validPrompts[0].text : "",
      prompts: validPrompts.length > 1 ? validPrompts : undefined,
    };

    onPromptSubmit?.(finalPrompt);
  };

  // Handle keyboard events
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  // Calculate normalized weights for display
  const totalWeight = prompts.reduce((sum, p) => sum + p.weight, 0);
  const normalizedWeights = prompts.map(p =>
    totalWeight > 0 ? (p.weight / totalWeight) * 100 : 0
  );

  const isSinglePrompt = prompts.length === 1;

  // Render prompt field (input or textarea)
  const renderPromptField = (
    index: number,
    placeholder: string,
    showRemove: boolean
  ) => {
    const isFocused = focusedIndex === index;
    const promptItem = prompts[index];

    return (
      <>
        {isFocused ? (
          <Textarea
            placeholder={placeholder}
            value={promptItem.text}
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
            value={promptItem.text}
            onChange={e => handlePromptTextChange(index, e.target.value)}
            onKeyDown={handleKeyDown}
            onFocus={() => setFocusedIndex(index)}
            disabled={disabled}
            className="flex-1 bg-transparent border-0 text-card-foreground placeholder:text-muted-foreground focus-visible:ring-0 focus-visible:ring-offset-0 p-0 disabled:opacity-50 disabled:cursor-not-allowed"
          />
        )}
        {index === prompts.length - 1 && prompts.length < MAX_PROMPTS && (
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

  // Render single prompt mode
  const renderSinglePrompt = () => {
    const isFocused = focusedIndex === 0;
    return (
      <div
        className={`flex items-start bg-card border border-border px-4 py-3 gap-3 transition-all ${
          isFocused ? "rounded-lg" : "rounded-full"
        } ${className}`}
      >
        {renderPromptField(0, "Edit prompt...", false)}
      </div>
    );
  };

  // Render multiple prompts mode
  const renderMultiplePrompts = () => {
    return (
      <div className={`space-y-3 ${className}`}>
        <div className="text-sm font-medium text-muted-foreground">
          Editing Timeline Prompt (Blend Mode)
        </div>
        {prompts.map((promptItem, index) => {
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
                  value={[promptItem.weight]}
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
