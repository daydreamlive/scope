import { useCallback } from "react";
import { useOnboarding } from "../../contexts/OnboardingContext";
import { InferenceModeStep } from "./InferenceModeStep";
import { CloudAuthStep } from "./CloudAuthStep";
import { WorkflowPickerStep } from "./WorkflowPickerStep";
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
  const { state, selectInferenceMode, completeAuth, startFromScratch } =
    useOnboarding();

  const handleSelectWorkflow = useCallback(
    (wf: StarterWorkflow) => {
      onActivateGraphMode();
      onSelectWorkflow(wf);
      // NOTE: workflowReady() is NOT called here — StreamPage calls it
      // after the WorkflowImportDialog finishes loading the workflow.
    },
    [onActivateGraphMode, onSelectWorkflow]
  );

  const handleStartFromScratch = useCallback(() => {
    onActivateGraphMode();
    startFromScratch();
  }, [onActivateGraphMode, startFromScratch]);

  const handleImportWorkflow = useCallback(() => {
    onOpenImportDialog();
  }, [onOpenImportDialog]);

  // Phase-dependent step indicator
  const phaseIndex =
    state.phase === "inference" ? 0 : state.phase === "cloud_auth" ? 1 : 2;
  const totalSteps = state.inferenceMode === "cloud" ? 3 : 2;

  return (
    <div className="fixed inset-0 z-[100] bg-background flex flex-col items-center justify-center p-6 animate-in fade-in-0 duration-300">
      {/* Step indicator */}
      <div className="absolute top-6 left-1/2 -translate-x-1/2 flex items-center gap-2">
        {Array.from({ length: totalSteps }).map((_, i) => (
          <div
            key={i}
            className={`h-1 rounded-full transition-all ${
              i <= phaseIndex ? "w-8 bg-foreground/40" : "w-8 bg-foreground/10"
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
  );
}
