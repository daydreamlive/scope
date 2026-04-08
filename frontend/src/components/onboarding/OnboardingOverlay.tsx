import { useCallback } from "react";
import { ArrowLeft } from "lucide-react";
import { useOnboarding } from "../../contexts/OnboardingContext";
import { useTelemetry } from "../../contexts/TelemetryContext";
import { persistOnboardingStyle } from "../../lib/onboardingStorage";
import { InferenceModeStep } from "./InferenceModeStep";
import { CloudAuthStep } from "./CloudAuthStep";
import { CloudConnectingStep } from "./CloudConnectingStep";
import { OnboardingStyleStep } from "./OnboardingStyleStep";
import { WorkflowPickerStep } from "./WorkflowPickerStep";
import { TelemetryDisclosure } from "./TelemetryDisclosure";
import { FogOfWarBackground } from "./FogOfWarBackground";
import type { StarterWorkflow } from "./starterWorkflows";

interface OnboardingOverlayProps {
  /**
   * Called when a starter workflow is selected. Receives the full embedded
   * workflow object so the caller can feed it into WorkflowImportDialog.
   * The tour will NOT start until the import dialog completes and
   * StreamPage calls workflowReady().
   */
  onSelectWorkflow: (workflow: StarterWorkflow) => void;
  /** Activate graph mode before the tour starts. */
  onActivateGraphMode: () => void;
  /** Open the existing WorkflowImportDialog as an escape hatch. */
  onOpenImportDialog: () => void;
}

export function OnboardingOverlay({
  onSelectWorkflow,
  onActivateGraphMode,
  onOpenImportDialog,
}: OnboardingOverlayProps) {
  const {
    state,
    selectInferenceMode,
    completeAuth,
    cloudConnected,
    setOnboardingStyle,
    startFromScratch,
    selectWorkflow,
    workflowReady,
    importWorkflowReady,
    goBack,
    telemetryDisclosed,
  } = useOnboarding();
  const { markDisclosed, setEnabled, flushQueue, dropQueue } = useTelemetry();

  const handleTelemetryAccept = useCallback(() => {
    markDisclosed();
    setEnabled(true);
    flushQueue();
    telemetryDisclosed();
  }, [markDisclosed, setEnabled, flushQueue, telemetryDisclosed]);

  const handleTelemetryDecline = useCallback(() => {
    markDisclosed();
    dropQueue();
    telemetryDisclosed();
  }, [markDisclosed, dropQueue, telemetryDisclosed]);

  const handleSelectWorkflow = useCallback(
    (wf: StarterWorkflow) => {
      onActivateGraphMode();
      // Remember which starter was chosen so the chip can suggest a different one
      localStorage.setItem("scope_starter_chosen_id", wf.id);
      // Set the workflow ID before completing so the analytics event captures it
      selectWorkflow(wf.id);
      // Complete onboarding FIRST so the overlay dismisses and the
      // WorkflowImportDialog (which renders at a lower z-index) becomes
      // visible and interactive.
      workflowReady();
      onSelectWorkflow(wf);
    },
    [onActivateGraphMode, onSelectWorkflow, selectWorkflow, workflowReady]
  );

  const handleStartFromScratch = useCallback(() => {
    onActivateGraphMode();
    startFromScratch();
  }, [onActivateGraphMode, startFromScratch]);

  const handleImportWorkflow = useCallback(() => {
    // Dismiss overlay first so the import dialog is visible
    importWorkflowReady();
    onOpenImportDialog();
  }, [onOpenImportDialog, importWorkflowReady]);

  const handleStyleSelect = useCallback(
    (style: "teaching" | "simple") => {
      persistOnboardingStyle(style);
      setOnboardingStyle(style);
    },
    [setOnboardingStyle]
  );

  // Phase-dependent step indicator
  // Local path:  inference(0) → telemetry(1) → style(2) → workflow(3)  = 4 steps
  // Cloud path:  inference(0) → cloud_auth(1) → cloud_connecting(2) → style(3) → workflow(4) = 5 steps
  const isCloud = state.inferenceMode === "cloud";
  const phaseIndex = (() => {
    switch (state.phase) {
      case "inference":
        return 0;
      case "cloud_auth":
        return 1;
      case "cloud_connecting":
        return 2;
      case "telemetry_disclosure":
        return 1;
      case "onboarding_style":
        return isCloud ? 3 : 2;
      case "workflow":
        return isCloud ? 4 : 3;
      default:
        return 0;
    }
  })();
  const totalSteps = isCloud ? 5 : 4;

  return (
    <div className="fixed inset-0 z-[100] bg-background animate-in fade-in-0 duration-300">
      <FogOfWarBackground />

      {/* Blur veil between fog background and content */}
      <div
        className="pointer-events-none absolute inset-0 z-[1] backdrop-blur-2xl"
        aria-hidden="true"
      />

      {/* Content layer — above the fog canvas + blur veil */}
      <div className="relative z-10 flex flex-col items-center justify-center h-full p-6">
        {/* Back button — visible after the first step (hidden during survey
            screens which handle their own internal back navigation) */}
        {state.phase !== "inference" &&
          state.phase !== "loading" &&
          state.phase !== "cloud_connecting" && (
            <button
              onClick={goBack}
              className="absolute top-6 left-6 flex items-center gap-1.5 text-sm text-muted-foreground hover:text-foreground transition-colors"
            >
              <ArrowLeft className="h-4 w-4" />
              Back
            </button>
          )}

        {/* Step indicator */}
        <div className="absolute top-6 left-1/2 -translate-x-1/2 flex items-center gap-2">
          {Array.from({ length: totalSteps }).map((_, i) => (
            <div
              key={i}
              className={`h-1 rounded-full transition-all ${
                i <= phaseIndex
                  ? "w-8 bg-foreground/40"
                  : "w-8 bg-foreground/10"
              }`}
            />
          ))}
        </div>

        {/* Phase content */}
        {state.phase === "loading" && (
          <div className="flex items-center justify-center">
            <div className="h-6 w-6 border-2 border-foreground/20 border-t-foreground/60 rounded-full animate-spin" />
          </div>
        )}

        {state.phase === "inference" && (
          <InferenceModeStep onSelect={selectInferenceMode} />
        )}

        {state.phase === "cloud_auth" && (
          <CloudAuthStep onComplete={completeAuth} />
        )}

        {state.phase === "cloud_connecting" && (
          <CloudConnectingStep onConnected={cloudConnected} onBack={goBack} />
        )}

        {state.phase === "telemetry_disclosure" && (
          <TelemetryDisclosure
            onAccept={handleTelemetryAccept}
            onDecline={handleTelemetryDecline}
          />
        )}

        {state.phase === "onboarding_style" && (
          <OnboardingStyleStep onSelect={handleStyleSelect} />
        )}

        {state.phase === "workflow" && (
          <WorkflowPickerStep
            onSelectWorkflow={handleSelectWorkflow}
            onStartFromScratch={handleStartFromScratch}
            onImportWorkflow={handleImportWorkflow}
          />
        )}
      </div>
    </div>
  );
}
