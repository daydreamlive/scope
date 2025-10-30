import type { PromptItem } from "../lib/api";
import type { TimelinePrompt } from "../components/PromptTimeline";

/**
 * Callback interfaces for submitting timeline prompts.
 * Supports both simple text and multi-prompt blends with transition settings.
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
 * Submits a timeline prompt through the appropriate callback with fallback logic.
 * Handles multi-prompt blends, simple prompts with transition settings, and text-only fallback.
 *
 * @param prompt - The timeline prompt to submit
 * @param callbacks - Object containing onPromptSubmit and/or onPromptItemsSubmit callbacks
 */
export function submitTimelinePrompt(
  prompt: TimelinePrompt,
  callbacks: PromptSubmissionCallbacks
): void {
  // If the prompt has blend data, send it as PromptItems
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
    // Simple prompt - send as single PromptItem with transition settings
    const promptItems: PromptItem[] = [{ text: prompt.text, weight: 100 }];
    callbacks.onPromptItemsSubmit(
      promptItems,
      prompt.transitionSteps,
      prompt.temporalInterpolationMethod
    );
  } else if (callbacks.onPromptSubmit) {
    // Fallback to simple text if onPromptItemsSubmit not available
    callbacks.onPromptSubmit(prompt.text);
  }
}
