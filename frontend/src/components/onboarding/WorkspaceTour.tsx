import { useState } from "react";
import { TourPopover } from "./TourPopover";
import { SIMPLE_TOUR_STEPS, TEACHING_TOUR_STEPS } from "./tourSteps";
import type { TourStepDef } from "./tourSteps";

const LS_KEY = "scope_tour_completed";

interface WorkspaceTourProps {
  onboardingStyle: "teaching" | "simple" | null;
  /** When true, a dialog is open and the tour should hide until it closes. */
  dialogOpen?: boolean;
}

/**
 * Two-step onboarding tooltip tour:
 *   Step 0 — points at the Play button (shown after workflow import dialog closes)
 *   Step 1 — points at the Workflows button (shown immediately after step 0 dismissed)
 *
 * Dismissed state persists in localStorage so returning users don't see it again.
 */
export function WorkspaceTour({
  onboardingStyle,
  dialogOpen = false,
}: WorkspaceTourProps) {
  type Phase = "step-0" | "step-1" | "done";
  const [phase, setPhase] = useState<Phase>(() => {
    if (localStorage.getItem(LS_KEY)) return "done";
    return "step-0";
  });

  // Nothing to show
  if (!onboardingStyle || phase === "done") return null;

  // Hide tour while a dialog is open (e.g. workflow import)
  if (dialogOpen) return null;

  const steps: TourStepDef[] =
    onboardingStyle === "simple" ? SIMPLE_TOUR_STEPS : TEACHING_TOUR_STEPS;

  if (phase === "step-0") {
    return (
      <TourPopover
        step={steps[0]}
        stepIndex={0}
        totalSteps={steps.length}
        onNext={() => setPhase("step-1")}
        onSkip={() => {
          setPhase("done");
          localStorage.setItem(LS_KEY, "1");
        }}
      />
    );
  }

  if (phase === "step-1") {
    return (
      <TourPopover
        step={steps[1]}
        stepIndex={1}
        totalSteps={steps.length}
        onNext={() => {
          setPhase("done");
          localStorage.setItem(LS_KEY, "1");
        }}
        onSkip={() => {
          setPhase("done");
          localStorage.setItem(LS_KEY, "1");
        }}
      />
    );
  }

  return null;
}
