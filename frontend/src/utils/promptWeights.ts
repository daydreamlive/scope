interface PromptWithWeight {
  text: string;
  weight: number;
}

/**
 * Redistributes weights equally when adding a new prompt.
 */
export function redistributeWeightsOnAdd<T extends PromptWithWeight>(
  existingPrompts: T[],
  newPromptData: Partial<T> = {}
): T[] {
  const newPromptCount = existingPrompts.length + 1;
  const equalWeight = 100 / newPromptCount;

  const redistributedPrompts = existingPrompts.map(p => ({
    ...p,
    weight: equalWeight,
  }));

  return [
    ...redistributedPrompts,
    { text: "", weight: equalWeight, ...newPromptData } as T,
  ];
}

/**
 * Redistributes weights proportionally when removing a prompt.
 */
export function redistributeWeightsOnRemove<T extends PromptWithWeight>(
  prompts: T[],
  indexToRemove: number
): T[] {
  const remainingPrompts = prompts.filter((_, i) => i !== indexToRemove);
  const totalWeight = remainingPrompts.reduce((sum, p) => sum + p.weight, 0);

  return remainingPrompts.map(p => ({
    ...p,
    weight:
      totalWeight > 0
        ? (p.weight / totalWeight) * 100
        : 100 / remainingPrompts.length,
  }));
}
