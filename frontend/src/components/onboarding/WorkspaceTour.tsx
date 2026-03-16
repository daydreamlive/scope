import { useOnboarding } from "../../contexts/OnboardingContext";
import { TourPopover } from "./TourPopover";
import { TOUR_STEPS } from "./tourSteps";

export function WorkspaceTour() {
  const { state, advanceTour, skipTour } = useOnboarding();
  const step = TOUR_STEPS[state.tourStep];

  if (!step) return null;

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
