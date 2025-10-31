import type { PromptItem } from "../lib/api";
import type { TimelinePrompt } from "../components/PromptTimeline";

/**
 * Callback interfaces for submitting timeline prompts.
 */
export interface PromptSubmissionCallbacks {
  onPromptSubmit?: (prompt: string) => void;
  onPromptItemsSubmit?: (
    prompts: PromptItem[],
    transitionSteps?: number,
    temporalInterpolationMethod?: "linear" | "slerp"
  ) => void;
}

/**
 * Submits a timeline prompt through the appropriate callback.
 *
 * @param prompt - The timeline prompt to submit
 * @param callbacks - Object containing onPromptSubmit and/or onPromptItemsSubmit callbacks
 */
export function submitTimelinePrompt(
  prompt: TimelinePrompt,
  callbacks: PromptSubmissionCallbacks
): void {
  if (
    prompt.prompts &&
    prompt.prompts.length > 0 &&
    callbacks.onPromptItemsSubmit
  ) {
    const promptItems: PromptItem[] = prompt.prompts.map(p => ({
      text: p.text,
      weight: p.weight,
    }));
    callbacks.onPromptItemsSubmit(
      promptItems,
      prompt.transitionSteps,
      prompt.temporalInterpolationMethod
    );
  } else if (callbacks.onPromptItemsSubmit) {
    const promptItems: PromptItem[] = [{ text: prompt.text, weight: 100 }];
    callbacks.onPromptItemsSubmit(
      promptItems,
      prompt.transitionSteps,
      prompt.temporalInterpolationMethod
    );
  } else if (callbacks.onPromptSubmit) {
    callbacks.onPromptSubmit(prompt.text);
  }
}
