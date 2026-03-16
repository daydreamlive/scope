import { useState, useEffect } from "react";
import { useOnboarding } from "../../contexts/OnboardingContext";
import { TourPopover } from "./TourPopover";
import { TOUR_STEPS } from "./tourSteps";

/**
 * Wait for the first tour step's anchor element to appear in the DOM before
 * showing anything. This prevents the tour from rendering on top of a blank
 * canvas while the workflow import is still finishing.
 */
function useAnchorReady(anchor: string | null): boolean {
  const [ready, setReady] = useState(() => {
    if (!anchor) return true; // centered steps are always ready
    return !!document.querySelector(`[data-tour="${anchor}"]`);
  });

  useEffect(() => {
    if (!anchor || ready) return;

    // Check immediately
    if (document.querySelector(`[data-tour="${anchor}"]`)) {
      setReady(true);
      return;
    }

    // Poll every 200ms for up to 5s
    let elapsed = 0;
    const interval = setInterval(() => {
      elapsed += 200;
      if (document.querySelector(`[data-tour="${anchor}"]`)) {
        setReady(true);
        clearInterval(interval);
      } else if (elapsed >= 5000) {
        // Give up waiting — show tour anyway so user isn't stuck
        setReady(true);
        clearInterval(interval);
      }
    }, 200);

    return () => clearInterval(interval);
  }, [anchor, ready]);

  return ready;
}

export function WorkspaceTour() {
  const { state, advanceTour, skipTour } = useOnboarding();
  const step = TOUR_STEPS[state.tourStep];

  // Wait for the first step's anchor to be in the DOM before starting
  const firstAnchor = TOUR_STEPS[0]?.anchor ?? null;
  const anchorReady = useAnchorReady(firstAnchor);

  if (!step || !anchorReady) return null;

  return (
    <TourPopover
      key={state.tourStep} // remount on step change for fresh positioning + animation
      step={step}
      stepIndex={state.tourStep}
      totalSteps={TOUR_STEPS.length}
      onNext={advanceTour}
      onSkip={skipTour}
    />
  );
}
