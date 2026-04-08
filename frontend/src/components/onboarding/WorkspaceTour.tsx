import { useState } from "react";
import { TourPopover } from "./TourPopover";
import { SIMPLE_TOUR_STEPS, TEACHING_TOUR_STEPS } from "./tourSteps";

const LS_KEY = "scope_tour_completed";

interface WorkspaceTourProps {
  onboardingStyle: "teaching" | "simple" | null;
  /** When true, a dialog is open and the tour should hide until it closes. */
  dialogOpen?: boolean;
}

/**
 * Multi-step onboarding tooltip tour.
 * Dismissed state persists in localStorage so returning users don't see it again.
 */
export function WorkspaceTour({
  onboardingStyle,
  dialogOpen = false,
}: WorkspaceTourProps) {
  const [stepIndex, setStepIndex] = useState<number>(() => {
    if (localStorage.getItem(LS_KEY)) return -1;
    return 0;
  });

  // Nothing to show
  if (!onboardingStyle || stepIndex < 0) return null;

  // Hide tour while a dialog is open (e.g. workflow import)
  if (dialogOpen) return null;

  const steps =
    onboardingStyle === "simple" ? SIMPLE_TOUR_STEPS : TEACHING_TOUR_STEPS;

  if (stepIndex >= steps.length) return null;

  const finish = () => {
    setStepIndex(-1);
    localStorage.setItem(LS_KEY, "1");
  };

  const isLast = stepIndex === steps.length - 1;

  return (
    <TourPopover
      step={steps[stepIndex]}
      stepIndex={stepIndex}
      totalSteps={steps.length}
      onNext={() => {
        if (isLast) {
          finish();
        } else {
          setStepIndex(i => i + 1);
        }
      }}
      onSkip={finish}
    />
  );
}
