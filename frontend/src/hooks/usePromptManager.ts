import { useState, useCallback, useMemo, useEffect } from "react";
import type { PromptItem } from "../lib/api";

export interface UsePromptManagerOptions {
  initialPrompts?: PromptItem[];
  prompts?: PromptItem[]; // For controlled components
  maxPrompts?: number;
  defaultWeight?: number;
  onPromptsChange?: (prompts: PromptItem[]) => void;
}

export interface UsePromptManagerReturn {
  prompts: PromptItem[];
  setPrompts: (
    prompts: PromptItem[] | ((prev: PromptItem[]) => PromptItem[])
  ) => void;
  handlePromptTextChange: (index: number, text: string) => void;
  handleWeightChange: (index: number, normalizedWeight: number) => void;
  handleAddPrompt: () => void;
  handleRemovePrompt: (index: number) => void;
  normalizedWeights: number[];
  totalWeight: number;
}

/**
 * Shared hook for managing prompt items (text and weights)
 * Used by both PromptInput and TimelinePromptEditor components
 *
 * Supports both controlled and uncontrolled modes:
 * - Controlled: pass `prompts` prop
 * - Uncontrolled: pass `initialPrompts` prop
 */
export function usePromptManager(
  options: UsePromptManagerOptions = {}
): UsePromptManagerReturn {
  const {
    initialPrompts = [],
    prompts: controlledPrompts,
    maxPrompts = 4,
    defaultWeight = 100,
    onPromptsChange,
  } = options;

  const isControlled = controlledPrompts !== undefined;
  const [internalPrompts, setInternalPromptsState] = useState<PromptItem[]>(
    controlledPrompts || initialPrompts
  );

  // Sync internal state with controlled prompts
  useEffect(() => {
    if (isControlled && controlledPrompts !== undefined) {
      setInternalPromptsState(controlledPrompts);
    }
  }, [isControlled, controlledPrompts]);

  const prompts = isControlled ? controlledPrompts : internalPrompts;

  const setPrompts = useCallback(
    (newPrompts: PromptItem[] | ((prev: PromptItem[]) => PromptItem[])) => {
      const updated =
        typeof newPrompts === "function" ? newPrompts(prompts) : newPrompts;

      if (!isControlled) {
        setInternalPromptsState(updated);
      }
      onPromptsChange?.(updated);
    },
    [prompts, isControlled, onPromptsChange]
  );

  const handlePromptTextChange = useCallback(
    (index: number, text: string) => {
      const newPrompts = [...prompts];
      newPrompts[index] = { ...newPrompts[index], text };
      setPrompts(newPrompts);
    },
    [prompts, setPrompts]
  );

  const handleWeightChange = useCallback(
    (index: number, normalizedWeight: number) => {
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
    },
    [prompts, setPrompts]
  );

  const handleAddPrompt = useCallback(() => {
    if (prompts.length < maxPrompts) {
      setPrompts([...prompts, { text: "", weight: defaultWeight }]);
    }
  }, [prompts, setPrompts, maxPrompts, defaultWeight]);

  const handleRemovePrompt = useCallback(
    (index: number) => {
      if (prompts.length > 1) {
        const newPrompts = prompts.filter((_, i) => i !== index);
        setPrompts(newPrompts);
      }
    },
    [prompts, setPrompts]
  );

  // Calculate normalized weights for display
  const totalWeight = useMemo(
    () => prompts.reduce((sum, p) => sum + p.weight, 0),
    [prompts]
  );

  const normalizedWeights = useMemo(
    () =>
      prompts.map(p => (totalWeight > 0 ? (p.weight / totalWeight) * 100 : 0)),
    [prompts, totalWeight]
  );

  return {
    prompts,
    setPrompts,
    handlePromptTextChange,
    handleWeightChange,
    handleAddPrompt,
    handleRemovePrompt,
    normalizedWeights,
    totalWeight,
  };
}
