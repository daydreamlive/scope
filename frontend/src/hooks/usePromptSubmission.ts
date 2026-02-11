import { useCallback } from "react";
import type { PromptItem } from "../lib/api";
import { useAppStore } from "../stores";
import { useShallow } from "zustand/react/shallow";
import { useStreamContext } from "../contexts/StreamContext";

/**
 * Handles prompt submission logic — both single-prompt and multi-prompt (blended)
 * submissions with transition support.
 */
export function usePromptSubmission() {
  const {
    interpolationMethod,
    transitionSteps,
    temporalInterpolationMethod,
    denoisingSteps,
    setPromptItems,
    setTransitionSteps,
    setTemporalInterpolationMethod,
  } = useAppStore(
    useShallow(s => ({
      interpolationMethod: s.interpolationMethod,
      transitionSteps: s.transitionSteps,
      temporalInterpolationMethod: s.temporalInterpolationMethod,
      denoisingSteps: s.settings.denoisingSteps,
      setPromptItems: s.setPromptItems,
      setTransitionSteps: s.setTransitionSteps,
      setTemporalInterpolationMethod: s.setTemporalInterpolationMethod,
    }))
  );

  const { sendParameterUpdate, isStreaming } = useStreamContext();

  const onPromptSubmit = useCallback(
    (text: string) => {
      const prompts: PromptItem[] = [{ text, weight: 100 }];
      setPromptItems(prompts);

      if (isStreaming && transitionSteps > 0) {
        sendParameterUpdate({
          transition: {
            target_prompts: prompts,
            num_steps: transitionSteps,
            temporal_interpolation_method: temporalInterpolationMethod,
          },
        });
      } else {
        sendParameterUpdate({
          prompts,
          prompt_interpolation_method: interpolationMethod,
          denoising_step_list: denoisingSteps || [700, 500],
        });
      }
    },
    [
      isStreaming,
      transitionSteps,
      temporalInterpolationMethod,
      interpolationMethod,
      denoisingSteps,
      sendParameterUpdate,
      setPromptItems,
    ]
  );

  const onPromptItemsSubmit = useCallback(
    (
      prompts: PromptItem[],
      blockTransitionSteps?: number,
      blockTemporalInterpolationMethod?: "linear" | "slerp"
    ) => {
      setPromptItems(prompts);

      const effectiveTransitionSteps = blockTransitionSteps ?? transitionSteps;
      const effectiveTemporalInterpolationMethod =
        blockTemporalInterpolationMethod ?? temporalInterpolationMethod;

      if (blockTransitionSteps !== undefined) {
        setTransitionSteps(blockTransitionSteps);
      }
      if (blockTemporalInterpolationMethod !== undefined) {
        setTemporalInterpolationMethod(blockTemporalInterpolationMethod);
      }

      if (isStreaming && effectiveTransitionSteps > 0) {
        sendParameterUpdate({
          transition: {
            target_prompts: prompts,
            num_steps: effectiveTransitionSteps,
            temporal_interpolation_method: effectiveTemporalInterpolationMethod,
          },
        });
      } else {
        sendParameterUpdate({
          prompts,
          prompt_interpolation_method: interpolationMethod,
          denoising_step_list: denoisingSteps || [700, 500],
        });
      }
    },
    [
      isStreaming,
      transitionSteps,
      temporalInterpolationMethod,
      interpolationMethod,
      denoisingSteps,
      sendParameterUpdate,
      setPromptItems,
      setTransitionSteps,
      setTemporalInterpolationMethod,
    ]
  );

  return { onPromptSubmit, onPromptItemsSubmit };
}
