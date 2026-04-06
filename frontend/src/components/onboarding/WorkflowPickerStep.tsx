import { useState, useEffect } from "react";
import { Play, Camera } from "lucide-react";
import { Button } from "../ui/button";
import { getWorkflowsForStyle, type StarterWorkflow } from "./starterWorkflows";
import { useOnboarding } from "../../contexts/OnboardingContext";
import { getHardwareInfo } from "../../lib/api";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface WorkflowPickerStepProps {
  onSelectWorkflow: (workflow: StarterWorkflow) => void;
  onStartFromScratch: () => void;
  onImportWorkflow: () => void;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function WorkflowPickerStep({
  onSelectWorkflow,
  onStartFromScratch,
  onImportWorkflow,
}: WorkflowPickerStepProps) {
  const { state } = useOnboarding();
  const [hasGpu, setHasGpu] = useState<boolean | null>(null);

  // When running locally, check if the backend has a GPU
  useEffect(() => {
    if (state.inferenceMode !== "local") return;
    let cancelled = false;
    getHardwareInfo()
      .then(info => {
        if (!cancelled) setHasGpu(info.vram_gb != null && info.vram_gb > 0);
      })
      .catch(() => {
        if (!cancelled) setHasGpu(false);
      });
    return () => {
      cancelled = true;
    };
  }, [state.inferenceMode]);

  // Filter workflows based on inference mode and GPU availability:
  // - Cloud: show style-based workflows
  // - Local + GPU: show style-based workflows (same as cloud)
  // - Local + no GPU: show only Camera Preview
  const isLocal = state.inferenceMode === "local";
  const styleWorkflows =
    isLocal && hasGpu === false
      ? []
      : getWorkflowsForStyle(state.onboardingStyle);
  const localWorkflows =
    isLocal && hasGpu === false ? getWorkflowsForStyle("local") : [];
  const workflows = [...styleWorkflows, ...localWorkflows];
  const [selected, setSelected] = useState<string | null>(null);

  const selectedWorkflow = workflows.find(wf => wf.id === selected);

  return (
    <div
      className={`flex flex-col items-center gap-6 w-full mx-auto ${workflows.length > 3 ? "max-w-5xl" : "max-w-3xl"}`}
    >
      <div className="text-center space-y-2">
        <h2 className="text-2xl font-semibold text-foreground">
          Pick a workflow to get started
        </h2>
        <p className="text-sm text-muted-foreground">
          Choose one and you&rsquo;ll be generating in seconds.
        </p>
        <p className="text-xs text-muted-foreground/60">
          Scope outputs live — use it for live streams, interactive art, or recording sessions.
          Route output via Spout/NDI/Syphon to any external app or broadcast tool.
        </p>
      </div>

      {/* Workflow cards */}
      <div
        className={`grid gap-4 w-full ${workflows.length === 1 ? "grid-cols-1 justify-items-center max-w-sm mx-auto" : workflows.length <= 3 ? "grid-cols-1 sm:grid-cols-3" : "grid-cols-1 sm:grid-cols-2 lg:grid-cols-4"}`}
      >
        {workflows.map(wf => {
          const isSelected = selected === wf.id;

          return (
            <button
              key={wf.id}
              onClick={() => setSelected(wf.id)}
              className={`relative flex flex-col rounded-xl border-2 p-5 text-left transition-all cursor-pointer ${
                isSelected
                  ? "border-foreground/30 bg-card ring-2 ring-foreground/10"
                  : "border-border bg-card/50 hover:border-border/80 hover:bg-card"
              }`}
            >
              {/* Thumbnail */}
              <div className="w-full aspect-video rounded-lg mb-3 overflow-hidden bg-black/20">
                {wf.thumbnail ? (
                  <img
                    src={wf.thumbnail}
                    alt={wf.title}
                    className="w-full h-full object-cover"
                    onError={e => {
                      (e.target as HTMLImageElement).style.display = "none";
                    }}
                  />
                ) : (
                  <div className="w-full h-full flex items-center justify-center bg-gradient-to-br from-slate-800 to-slate-900">
                    <Camera className="h-8 w-8 text-slate-500" />
                  </div>
                )}
              </div>

              <div className="flex items-start justify-between gap-2 mb-1">
                <p className="text-sm font-medium text-foreground">
                  {wf.title}
                </p>
              </div>
              <p className="text-[10px] font-medium text-muted-foreground/70 uppercase tracking-wide mb-1">
                {wf.category}
              </p>
              <p className="text-xs text-muted-foreground leading-relaxed">
                {wf.description}
              </p>
            </button>
          );
        })}
      </div>

      {/* Primary action */}
      <Button
        onClick={() => selectedWorkflow && onSelectWorkflow(selectedWorkflow)}
        disabled={!selectedWorkflow}
        className="px-8"
      >
        <Play className="h-4 w-4 mr-2" />
        Get Started
      </Button>

      {/* Secondary CTAs */}
      <div className="flex items-center gap-4">
        <button
          onClick={onStartFromScratch}
          className="text-xs text-muted-foreground hover:text-foreground transition-colors underline underline-offset-2"
        >
          Start from scratch
        </button>
        <span className="text-xs text-muted-foreground/40">|</span>
        <button
          onClick={onImportWorkflow}
          className="text-xs text-muted-foreground hover:text-foreground transition-colors underline underline-offset-2"
        >
          Import a workflow
        </button>
      </div>
    </div>
  );
}
