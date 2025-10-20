import React, { useState, useEffect } from "react";
import { Input } from "./ui/input";
import { Textarea } from "./ui/textarea";
import { Button } from "./ui/button";
import { Slider } from "./ui/slider";
import { ArrowUp, Plus, X } from "lucide-react";
import type { TimelinePrompt } from "./PromptTimeline";

interface TimelinePromptEditorProps {
  className?: string;
  prompt: TimelinePrompt | null;
  onPromptUpdate?: (prompt: TimelinePrompt) => void;
  onPromptSubmit?: (prompt: TimelinePrompt) => void;
  disabled?: boolean;
}

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
        setPrompts([{ text: prompt.text, weight: 100 }]);
      }
    } else {
      setEditingPrompt(null);
      setPrompts([]);
    }
  }, [prompt]);

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

  const handleAddPrompt = () => {
    if (prompts.length < 4) {
      const newPrompts = [...prompts, { text: "", weight: 100 }];
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

  // Render a single prompt field with expandable textarea
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
        <Button
          onClick={handleSubmit}
          disabled={
            disabled || !prompts.some(p => p.text.trim()) || !editingPrompt
          }
          size="sm"
          className="rounded-full w-8 h-8 p-0 bg-black hover:bg-gray-800 text-white disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <ArrowUp className="h-4 w-4" />
        </Button>
        {index === prompts.length - 1 && prompts.length < 4 && (
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

  if (!editingPrompt) {
    return (
      <div className={`text-center text-muted-foreground py-8 ${className}`}>
        Click on a prompt box in the timeline to edit it
      </div>
    );
  }

  // Single prompt mode: simple pill UI
  if (isSinglePrompt) {
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
  }

  // Multiple prompts mode: show weights and controls
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
}
