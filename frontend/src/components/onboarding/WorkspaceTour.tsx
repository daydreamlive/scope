import { useState, useEffect, useRef } from "react";
import { TourPopover } from "./TourPopover";
import { SIMPLE_TOUR_STEPS, TEACHING_TOUR_STEPS } from "./tourSteps";
import type { TourStepDef } from "./tourSteps";

const LS_KEY = "scope_tour_completed";

interface WorkspaceTourProps {
  onboardingStyle: "teaching" | "simple" | null;
  isStreaming: boolean;
  /** When true, a dialog is open and the tour should hide until it closes. */
  dialogOpen?: boolean;
}

/**
 * Two-step onboarding tooltip tour:
 *   Step 0 — points at the Play button (shown after workflow import dialog closes)
 *   Step 1 — points at the Workflows button (shown after first play)
 *
 * Dismissed state persists in localStorage so returning users don't see it again.
 */
export function WorkspaceTour({
  onboardingStyle,
  isStreaming,
  dialogOpen = false,
}: WorkspaceTourProps) {
  // "waiting" = showing step 0, "played" = waiting for step 1,
  // "showing-workflows" = showing step 1, "done" = finished
  type Phase = "waiting" | "played" | "showing-workflows" | "done";
  const [phase, setPhase] = useState<Phase>(() => {
    if (localStorage.getItem(LS_KEY)) return "done";
    return "waiting";
  });

  // Track whether streaming has ever started in this session
  const hasEverStreamedRef = useRef(false);

  useEffect(() => {
    if (isStreaming && !hasEverStreamedRef.current) {
      hasEverStreamedRef.current = true;
      if (phase === "played") {
        setPhase("showing-workflows");
      }
    }
  }, [isStreaming, phase]);

  // Nothing to show
  if (!onboardingStyle || phase === "done") return null;

  // Hide tour while a dialog is open (e.g. workflow import)
  if (dialogOpen) return null;

  const steps: TourStepDef[] =
    onboardingStyle === "simple" ? SIMPLE_TOUR_STEPS : TEACHING_TOUR_STEPS;

  if (phase === "waiting") {
    return (
      <TourPopover
        step={steps[0]}
        stepIndex={0}
        totalSteps={steps.length}
        onNext={() => setPhase("played")}
        onSkip={() => {
          setPhase("done");
          localStorage.setItem(LS_KEY, "1");
        }}
      />
    );
  }

  if (phase === "showing-workflows") {
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

  // phase === "played" — waiting for first stream, nothing visible
  return null;
}
