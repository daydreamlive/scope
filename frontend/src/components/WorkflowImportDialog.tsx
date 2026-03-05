import { useState, useCallback, useRef } from "react";
import {
  CheckCircle2,
  XCircle,
  AlertTriangle,
  Download,
  Upload,
  Loader2,
} from "lucide-react";
import { Button } from "./ui/button";
import { Badge } from "./ui/badge";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "./ui/dialog";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "./ui/alert-dialog";
import { toast } from "sonner";
import type {
  ScopeWorkflow,
  ResolutionItem,
  WorkflowResolutionPlan,
  WorkflowLoRAProvenance,
} from "../lib/workflowApi";
import { resolveWorkflow } from "../lib/api";
import { useLoRAsContext } from "../contexts/LoRAsContext";
import { usePipelinesContext } from "../contexts/PipelinesContext";
import { usePluginsContext } from "../contexts/PluginsContext";
import type { SettingsState } from "../types";
import type { TimelinePrompt } from "./PromptTimeline";
import {
  workflowToSettings,
  workflowTimelineToPrompts,
  workflowToPromptState,
} from "../lib/workflowSettings";
import type { WorkflowPromptState } from "../lib/workflowSettings";
import {
  useLoRADownloads,
  usePluginInstalls,
} from "../hooks/useWorkflowDependencies";
import { DependencyStatusIndicator } from "./DependencyStatusIndicator";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type ImportStep = "select" | "review";

interface WorkflowImportDialogProps {
  open: boolean;
  onClose: () => void;
  onLoad: (
    settings: Partial<SettingsState>,
    timelinePrompts: TimelinePrompt[],
    promptState: WorkflowPromptState | null
  ) => void;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const statusIcon = (status: ResolutionItem["status"]) => {
  switch (status) {
    case "ok":
      return <CheckCircle2 className="h-4 w-4 text-green-500 shrink-0" />;
    case "missing":
      return <XCircle className="h-4 w-4 text-red-500 shrink-0" />;
    case "version_mismatch":
      return <AlertTriangle className="h-4 w-4 text-amber-500 shrink-0" />;
  }
};

const kindLabel = (kind: ResolutionItem["kind"]) => {
  switch (kind) {
    case "pipeline":
      return "Pipeline";
    case "plugin":
      return "Plugin";
    case "lora":
      return "LoRA";
  }
};

function provenanceLabel(prov: WorkflowLoRAProvenance): string {
  if (prov.source === "huggingface" && prov.repo_id) {
    return `HuggingFace: ${prov.repo_id}`;
  }
  if (prov.source === "civitai") {
    return `CivitAI model ${prov.model_id ?? prov.version_id ?? ""}`;
  }
  if (prov.source === "url" && prov.url) {
    return prov.url;
  }
  return prov.source;
}

function findLoRAProvenance(
  workflow: ScopeWorkflow,
  filename: string
): WorkflowLoRAProvenance | null {
  const lora = workflow.pipelines
    .flatMap(p => p.loras)
    .find(l => l.filename === filename);
  if (lora?.provenance && lora.provenance.source !== "local") {
    return lora.provenance;
  }
  return null;
}

function LoRAProvenanceLabel({
  workflow,
  filename,
}: {
  workflow: ScopeWorkflow;
  filename: string;
}) {
  const prov = findLoRAProvenance(workflow, filename);
  if (!prov) return null;
  return (
    <p className="text-[10px] text-muted-foreground mt-0.5">
      {provenanceLabel(prov)}
    </p>
  );
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function WorkflowImportDialog({
  open,
  onClose,
  onLoad,
}: WorkflowImportDialogProps) {
  const [step, setStep] = useState<ImportStep>("select");
  const [workflow, setWorkflow] = useState<ScopeWorkflow | null>(null);
  const [plan, setPlan] = useState<WorkflowResolutionPlan | null>(null);
  const [validating, setValidating] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { refresh: refreshLoRAs } = useLoRAsContext();
  const { refreshPipelines } = usePipelinesContext();
  const { refresh: refreshPlugins } = usePluginsContext();

  // -- Re-resolution callback (used by both LoRA downloads and plugin installs)
  const reResolveWorkflow = useCallback(async () => {
    if (!workflow) return;
    try {
      await Promise.all([refreshPipelines(), refreshLoRAs(), refreshPlugins()]);
      const resolution = await resolveWorkflow(workflow);
      setPlan(resolution);
    } catch (err) {
      console.error("Failed to re-resolve workflow:", err);
    }
  }, [workflow, refreshPipelines, refreshLoRAs, refreshPlugins]);

  const loras = useLoRADownloads(workflow, reResolveWorkflow);

  // -- Confirm dialog state (shared for load & plugin-install confirms) -----
  const [confirmState, setConfirmState] = useState<{
    title: string;
    description: string;
    resolve: (confirmed: boolean) => void;
  } | null>(null);

  const showConfirm = useCallback(
    (title: string, description: string): Promise<boolean> =>
      new Promise(resolve => {
        setConfirmState({ title, description, resolve });
      }),
    []
  );

  const handleConfirmAction = useCallback(() => {
    confirmState?.resolve(true);
    setConfirmState(null);
  }, [confirmState]);

  const handleConfirmCancel = useCallback(() => {
    confirmState?.resolve(false);
    setConfirmState(null);
  }, [confirmState]);

  // -- Plugin install confirm callback --------------------------------------
  const confirmPluginInstall = useCallback(
    (installSpec: string) =>
      showConfirm(
        "Install Plugin",
        `This will install the package "${installSpec}" via pip. Only proceed if you trust the workflow source.`
      ),
    [showConfirm]
  );

  const plugins = usePluginInstalls(
    workflow,
    reResolveWorkflow,
    confirmPluginInstall
  );

  // Reset all state when dialog closes
  const handleClose = useCallback(() => {
    setStep("select");
    setWorkflow(null);
    setPlan(null);
    loras.reset();
    plugins.reset();
    setValidating(false);
    onClose();
  }, [onClose, loras.reset, plugins.reset]);

  // -----------------------------------------------------------------------
  // File selection and validation
  // -----------------------------------------------------------------------

  const handleFileSelect = useCallback(
    async (file: File) => {
      try {
        setValidating(true);
        const text = await file.text();
        let parsed: ScopeWorkflow;
        try {
          parsed = JSON.parse(text);
        } catch {
          toast.error("Invalid JSON file");
          return;
        }

        if (parsed.format !== "scope-workflow") {
          toast.error("Not a Scope workflow file", {
            description: 'Expected format: "scope-workflow"',
          });
          return;
        }

        if (
          !parsed.metadata ||
          typeof parsed.metadata.name !== "string" ||
          !Array.isArray(parsed.pipelines) ||
          parsed.pipelines.length === 0
        ) {
          toast.error("Malformed workflow file", {
            description: "Missing required fields: metadata or pipelines",
          });
          return;
        }

        setWorkflow(parsed);

        const resolution = await resolveWorkflow(parsed);
        setPlan(resolution);

        // Initialize dependency states from resolution items
        loras.initialize(
          resolution.items
            .filter(i => i.kind === "lora" && i.status === "missing")
            .map(i => i.name)
        );
        plugins.initialize(
          resolution.items
            .filter(
              i =>
                i.kind === "plugin" &&
                i.status === "missing" &&
                i.can_auto_resolve
            )
            .map(i => i.name)
        );

        setStep("review");
      } catch (err) {
        console.error("Workflow validation failed:", err);
        toast.error("Validation failed", {
          description: err instanceof Error ? err.message : String(err),
        });
      } finally {
        setValidating(false);
      }
    },
    [loras.initialize, plugins.initialize]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const file = e.dataTransfer.files[0];
      if (file) handleFileSelect(file);
    },
    [handleFileSelect]
  );

  const handleInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) handleFileSelect(file);
      // Reset so the same file can be re-selected
      e.target.value = "";
    },
    [handleFileSelect]
  );

  // -----------------------------------------------------------------------
  // Load workflow into the interface
  // -----------------------------------------------------------------------

  const handleLoad = useCallback(async () => {
    if (!workflow) return;

    const confirmed = await showConfirm(
      "Load Workflow",
      "Loading this workflow will replace your current settings and timeline. Continue?"
    );
    if (!confirmed) return;

    // Fetch fresh LoRA files to avoid stale closure after downloads
    const freshLoraFiles = await refreshLoRAs();
    const importedSettings = workflowToSettings(workflow, freshLoraFiles);
    const timelinePrompts = workflowTimelineToPrompts(workflow.timeline);
    const promptState = workflowToPromptState(workflow);

    onLoad(importedSettings, timelinePrompts, promptState);
    toast.success("Workflow loaded", {
      description: `"${workflow.metadata.name}" loaded into the interface`,
    });
    handleClose();
  }, [workflow, onLoad, handleClose, showConfirm, refreshLoRAs]);

  // -----------------------------------------------------------------------
  // Derived state
  // -----------------------------------------------------------------------

  const missingLoRAs = plan?.items.filter(
    i => i.kind === "lora" && i.status === "missing"
  );
  const downloadableLoRAs = missingLoRAs?.filter(i => i.can_auto_resolve);

  const missingPlugins = plan?.items.filter(
    i => i.kind === "plugin" && i.status === "missing"
  );
  const installablePlugins = missingPlugins?.filter(i => i.can_auto_resolve);

  const hasUnresolvedDeps = plan?.items.some(i => {
    if (i.status !== "missing") return false;
    if (i.kind === "lora") return loras.downloads[i.name] !== "done";
    if (i.kind === "plugin") return plugins.installs[i.name] !== "done";
    return true; // pipeline or other kinds — always unresolved if missing
  });

  // -----------------------------------------------------------------------
  // Render
  // -----------------------------------------------------------------------

  return (
    <Dialog open={open} onOpenChange={isOpen => !isOpen && handleClose()}>
      <DialogContent className="sm:max-w-lg max-h-[80vh] flex flex-col">
        <DialogHeader>
          <DialogTitle>
            {step === "select" && "Import Workflow"}
            {step === "review" && "Review Workflow"}
          </DialogTitle>
          <DialogDescription>
            {step === "select" &&
              "Select a .scope-workflow.json file to import."}
            {step === "review" &&
              "Review dependencies, then load into the interface."}
          </DialogDescription>
        </DialogHeader>

        {/* Step 1: File selection */}
        {step === "select" && (
          <div
            className="flex flex-col items-center justify-center gap-4 py-8 border-2 border-dashed border-muted-foreground/25 rounded-lg cursor-pointer hover:border-muted-foreground/50 transition-colors"
            onDrop={handleDrop}
            onDragOver={e => e.preventDefault()}
            onClick={() => fileInputRef.current?.click()}
          >
            {validating ? (
              <>
                <Loader2 className="h-8 w-8 text-muted-foreground animate-spin" />
                <p className="text-sm text-muted-foreground">Validating...</p>
              </>
            ) : (
              <>
                <Upload className="h-8 w-8 text-muted-foreground" />
                <div className="text-center">
                  <p className="text-sm font-medium">
                    Drop a workflow file here
                  </p>
                  <p className="text-xs text-muted-foreground mt-1">
                    or click to browse
                  </p>
                </div>
              </>
            )}
            <input
              ref={fileInputRef}
              type="file"
              accept=".json"
              className="hidden"
              onChange={handleInputChange}
            />
          </div>
        )}

        {/* Step 2: Review resolution plan */}
        {step === "review" && plan && workflow && (
          <div className="flex flex-col gap-4 overflow-y-auto min-h-0">
            {/* Workflow metadata */}
            <div className="text-sm space-y-1">
              <p className="font-medium">{workflow.metadata.name}</p>
              <p className="text-muted-foreground text-xs">
                Scope v{workflow.metadata.scope_version} &middot;{" "}
                {new Date(workflow.metadata.created_at).toLocaleDateString()}
              </p>
            </div>

            {/* Resolution items */}
            <div className="space-y-2">
              {plan.items.map((item, i) => (
                <div
                  key={`${item.kind}-${item.name}-${i}`}
                  className="flex items-start gap-2 text-sm"
                >
                  {statusIcon(item.status)}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <Badge
                        variant="outline"
                        className="text-[10px] px-1.5 py-0"
                      >
                        {kindLabel(item.kind)}
                      </Badge>
                      <span className="font-medium truncate">{item.name}</span>
                    </div>
                    {item.detail && (
                      <p className="text-xs text-muted-foreground mt-0.5">
                        {item.detail}
                      </p>
                    )}

                    {/* LoRA download button */}
                    {item.kind === "lora" &&
                      item.status === "missing" &&
                      item.can_auto_resolve && (
                        <div className="mt-1">
                          <DependencyStatusIndicator
                            status={loras.downloads[item.name]}
                            activeStatus="downloading"
                            doneLabel="Downloaded"
                            activeLabel="Downloading..."
                            idleLabel="Download"
                            onAction={() => loras.downloadOne(item.name)}
                          />
                          <LoRAProvenanceLabel
                            workflow={workflow}
                            filename={item.name}
                          />
                        </div>
                      )}

                    {/* Plugin install button */}
                    {item.kind === "plugin" &&
                      item.status === "missing" &&
                      item.can_auto_resolve && (
                        <div className="mt-1">
                          <DependencyStatusIndicator
                            status={plugins.installs[item.name]}
                            activeStatus="installing"
                            doneLabel="Installed"
                            activeLabel="Installing..."
                            idleLabel="Install"
                            onAction={() => plugins.installOne(item.name)}
                          />
                        </div>
                      )}
                  </div>
                </div>
              ))}
            </div>

            {/* Download all LoRAs button */}
            {downloadableLoRAs &&
              downloadableLoRAs.length > 1 &&
              missingLoRAs &&
              missingLoRAs.some(l => loras.downloads[l.name] !== "done") && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={loras.downloadAll}
                  disabled={loras.someDownloading}
                >
                  <Download className="h-4 w-4 mr-2" />
                  {loras.someDownloading
                    ? "Downloading..."
                    : `Download All Missing LoRAs (${downloadableLoRAs.filter(l => loras.downloads[l.name] !== "done").length})`}
                </Button>
              )}

            {/* Install all plugins button */}
            {installablePlugins &&
              installablePlugins.length > 1 &&
              missingPlugins &&
              missingPlugins.some(p => plugins.installs[p.name] !== "done") && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={plugins.installAll}
                  disabled={plugins.someInstalling}
                >
                  <Download className="h-4 w-4 mr-2" />
                  {plugins.someInstalling
                    ? "Installing..."
                    : `Install All Missing Plugins (${installablePlugins.filter(p => plugins.installs[p.name] !== "done").length})`}
                </Button>
              )}

            {/* Warnings */}
            {plan.warnings.length > 0 && (
              <div className="space-y-1">
                {plan.warnings.map((w, i) => (
                  <div
                    key={i}
                    className="flex items-start gap-2 text-xs text-amber-500"
                  >
                    <AlertTriangle className="h-3 w-3 mt-0.5 shrink-0" />
                    <span>{w}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        <DialogFooter>
          <Button variant="ghost" onClick={handleClose}>
            Cancel
          </Button>
          {step === "review" && (
            <Button
              onClick={handleLoad}
              disabled={
                loras.someDownloading ||
                plugins.someInstalling ||
                hasUnresolvedDeps
              }
            >
              Load Workflow
            </Button>
          )}
        </DialogFooter>
      </DialogContent>

      {/* Confirmation alert dialog (replaces window.confirm) */}
      <AlertDialog
        open={confirmState !== null}
        onOpenChange={open => {
          if (!open) handleConfirmCancel();
        }}
      >
        <AlertDialogContent className="sm:max-w-md">
          <AlertDialogHeader>
            <AlertDialogTitle>{confirmState?.title}</AlertDialogTitle>
            <AlertDialogDescription>
              {confirmState?.description}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel onClick={handleConfirmCancel}>
              Cancel
            </AlertDialogCancel>
            <AlertDialogAction onClick={handleConfirmAction}>
              Continue
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </Dialog>
  );
}
