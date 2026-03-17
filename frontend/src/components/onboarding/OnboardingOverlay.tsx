import { useCallback } from "react";
import { useOnboarding } from "../../contexts/OnboardingContext";
import { InferenceModeStep } from "./InferenceModeStep";
import { CloudAuthStep } from "./CloudAuthStep";
import { WorkflowPickerStep } from "./WorkflowPickerStep";
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
    startFromScratch,
    workflowReady,
    importWorkflowReady,
  } = useOnboarding();

  const handleSelectWorkflow = useCallback(
    (wf: StarterWorkflow) => {
      onActivateGraphMode();
      // Complete onboarding FIRST so the overlay dismisses and the
      // WorkflowImportDialog (which renders at a lower z-index) becomes
      // visible and interactive.
      workflowReady();
      onSelectWorkflow(wf);
    },
    [onActivateGraphMode, onSelectWorkflow, workflowReady]
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

  // Phase-dependent step indicator
  const phaseIndex =
    state.phase === "inference" ? 0 : state.phase === "cloud_auth" ? 1 : 2;
  const totalSteps = state.inferenceMode === "cloud" ? 3 : 2;

  return (
    <div className="fixed inset-0 z-[100] bg-background animate-in fade-in-0 duration-300">
      <FogOfWarBackground />

      {/* Blur veil between fog background and content */}
      <div className="pointer-events-none absolute inset-0 z-[1] backdrop-blur-2xl" aria-hidden="true" />

      {/* Content layer — above the fog canvas + blur veil */}
      <div className="relative z-10 flex flex-col items-center justify-center h-full p-6">
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
        {state.phase === "inference" && (
          <InferenceModeStep onSelect={selectInferenceMode} />
        )}

        {state.phase === "cloud_auth" && (
          <CloudAuthStep onComplete={completeAuth} />
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
