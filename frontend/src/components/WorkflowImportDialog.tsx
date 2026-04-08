import { useState, useCallback, useEffect, useRef, useMemo } from "react";
import {
  AlertTriangle,
  Download,
  Upload,
  Loader2,
  ExternalLink,
  Save,
  KeyRound,
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
import type { ScopeWorkflow, WorkflowResolutionPlan } from "../lib/workflowApi";
import { resolveWorkflow, getApiKeys, setApiKey } from "../lib/api";
import type { ApiKeyInfo } from "../lib/api";
import {
  statusIcon,
  kindLabel,
  findLoRAProvenance,
  LoRAProvenanceLabel,
} from "./workflowDialogHelpers";
import { useLoRAsContext } from "../contexts/LoRAsContext";
import { usePipelinesContext } from "../contexts/PipelinesContext";
import { usePluginsContext } from "../contexts/PluginsContext";
import type { SettingsState } from "../types";
import type { TimelinePrompt } from "./PromptTimeline";
import {
  workflowToSettings,
  workflowTimelineToPrompts,
  workflowToPromptState,
  extractFilename,
} from "../lib/workflowSettings";
import type { WorkflowPromptState } from "../lib/workflowSettings";
import {
  useLoRADownloads,
  usePluginInstalls,
} from "../hooks/useWorkflowDependencies";
import { DependencyStatusIndicator } from "./DependencyStatusIndicator";
import { trackEvent } from "../lib/analytics";

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
  /** When set, the dialog calls this instead of onLoad (used for graph-mode import). */
  onLoadToGraph?: (workflow: ScopeWorkflow) => void;
  initialWorkflow?: ScopeWorkflow | null;
  /** When true, API key warnings for LoRA downloads are hidden (cloud handles auth). */
  cloudConnected?: boolean;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function WorkflowImportDialog({
  open,
  onClose,
  onLoad,
  onLoadToGraph,
  initialWorkflow,
  cloudConnected = false,
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
  const { reset: resetLoras, initialize: initializeLoras } = loras;

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
        "Install Node",
        `This will install the package "${installSpec}" via pip. Only proceed if you trust the workflow source.`
      ),
    [showConfirm]
  );

  const plugins = usePluginInstalls(
    workflow,
    reResolveWorkflow,
    confirmPluginInstall
  );
  const { reset: resetPlugins, initialize: initializePlugins } = plugins;

  // -- API key warning state -------------------------------------------------
  const [apiKeys, setApiKeys] = useState<ApiKeyInfo[]>([]);
  const [keyInputValues, setKeyInputValues] = useState<Record<string, string>>(
    {}
  );
  const [savingKeyIds, setSavingKeyIds] = useState<Set<string>>(new Set());

  const fetchApiKeys = useCallback(async () => {
    try {
      const response = await getApiKeys();
      setApiKeys(response.keys);
    } catch {
      // Non-critical — warning just won't show
    }
  }, []);

  // Fetch API keys when entering review step (skip for cloud — cloud handles auth)
  useEffect(() => {
    if (step === "review" && !cloudConnected) {
      fetchApiKeys();
    }
  }, [step, fetchApiKeys, cloudConnected]);

  // Services that need a key: missing LoRAs hosted on a service without a key.
  // On cloud inference the server manages credentials, so skip the warning.
  const missingKeyServices = useMemo(() => {
    if (cloudConnected) return [];
    if (!plan || !workflow || apiKeys.length === 0) return [];
    const neededSources = new Set<string>();
    for (const item of plan.items) {
      if (
        item.kind !== "lora" ||
        item.status !== "missing" ||
        !item.can_auto_resolve
      )
        continue;
      const prov = findLoRAProvenance(workflow, item.name);
      if (prov?.source === "huggingface" || prov?.source === "civitai") {
        neededSources.add(prov.source);
      }
    }
    return apiKeys.filter(k => neededSources.has(k.id) && !k.is_set);
  }, [cloudConnected, plan, workflow, apiKeys]);

  const handleSaveApiKey = useCallback(
    async (keyInfo: ApiKeyInfo) => {
      const value = keyInputValues[keyInfo.id];
      if (!value?.trim()) return;

      setSavingKeyIds(prev => new Set(prev).add(keyInfo.id));
      try {
        const response = await setApiKey(keyInfo.id, value.trim());
        if (response.success) {
          toast.success(`${keyInfo.name} API key saved`);
          setKeyInputValues(prev => {
            const next = { ...prev };
            delete next[keyInfo.id];
            return next;
          });
          await fetchApiKeys();
        }
      } catch (err) {
        toast.error(
          err instanceof Error ? err.message : "Failed to save API key"
        );
      } finally {
        setSavingKeyIds(prev => {
          const next = new Set(prev);
          next.delete(keyInfo.id);
          return next;
        });
      }
    },
    [keyInputValues, fetchApiKeys]
  );

  // Reset all state when dialog closes
  const handleClose = useCallback(() => {
    setStep("select");
    setWorkflow(null);
    setPlan(null);
    resetLoras();
    resetPlugins();
    setApiKeys([]);
    setKeyInputValues({});
    setSavingKeyIds(new Set());
    setValidating(false);
    onClose();
  }, [onClose, resetLoras, resetPlugins]);

  // -----------------------------------------------------------------------
  // Auto-resolve when opened with a preloaded workflow (e.g. from deeplink)
  // -----------------------------------------------------------------------

  useEffect(() => {
    if (!open || !initialWorkflow) return;

    let cancelled = false;
    (async () => {
      try {
        setValidating(true);
        setWorkflow(initialWorkflow);

        const resolution = await resolveWorkflow(initialWorkflow);
        if (cancelled) return;

        // If all dependencies are already resolved, skip the review dialog
        if (
          resolution.items.every(i => i.status === "ok") &&
          resolution.warnings.length === 0
        ) {
          await loadWorkflowDirect(initialWorkflow);
          return;
        }

        setPlan(resolution);

        initializeLoras(
          resolution.items
            .filter(i => i.kind === "lora" && i.status === "missing")
            .map(i => i.name)
        );
        initializePlugins(
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
        if (cancelled) return;
        console.error("Workflow resolution failed:", err);
        toast.error("Failed to resolve workflow", {
          description: err instanceof Error ? err.message : String(err),
        });
        handleClose();
      } finally {
        if (!cancelled) setValidating(false);
      }
    })();

    return () => {
      cancelled = true;
    };
    // Only re-run when the dialog opens with a new initialWorkflow reference
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, initialWorkflow]);

  // -----------------------------------------------------------------------
  // Load workflow into the interface
  // -----------------------------------------------------------------------

  /** Load a workflow directly (no confirmation dialog). */
  const loadWorkflowDirect = useCallback(
    async (wf: ScopeWorkflow) => {
      if (onLoadToGraph) {
        const freshLoraFiles = await refreshLoRAs();
        const patchedWorkflow = {
          ...wf,
          pipelines: wf.pipelines.map(p => ({
            ...p,
            loras: p.loras.map(l => {
              const resolved = freshLoraFiles.find(
                f =>
                  extractFilename(f.path).toLowerCase() ===
                  l.filename.toLowerCase()
              );
              return resolved ? { ...l, filename: resolved.path } : l;
            }),
          })),
        };
        onLoadToGraph(patchedWorkflow);
        trackEvent("workflow_imported", {
          node_count: wf.graph?.nodes?.length ?? wf.pipelines.length,
          source: "file",
          surface: "app_chrome",
        });
        toast.success("Workflow loaded into graph", {
          description: `"${wf.metadata.name}" loaded into the graph editor`,
        });
        handleClose();
        return;
      }

      // Fetch fresh LoRA files to avoid stale closure after downloads
      const freshLoraFiles = await refreshLoRAs();
      const importedSettings = workflowToSettings(wf, freshLoraFiles);
      const timelinePrompts = workflowTimelineToPrompts(wf.timeline);
      const promptState = workflowToPromptState(wf);

      // Persist the workflow's graph to localStorage so the graph editor can
      // pick it up when refreshGraph() is called after import.
      if (wf.graph?.nodes && wf.graph?.edges) {
        try {
          localStorage.setItem("scope:graph:backup", JSON.stringify(wf.graph));
        } catch {
          /* ignore */
        }
      }

      onLoad(importedSettings, timelinePrompts, promptState);
      trackEvent("workflow_imported", {
        node_count: wf.graph?.nodes?.length ?? wf.pipelines.length,
        source: "file",
        surface: "app_chrome",
      });
      toast.success("Workflow loaded", {
        description: `"${wf.metadata.name}" loaded into the interface`,
      });
      handleClose();
    },
    [onLoad, onLoadToGraph, handleClose, refreshLoRAs]
  );

  const handleLoad = useCallback(async () => {
    if (!workflow) return;

    const confirmed = await showConfirm(
      "Load Workflow",
      "Loading this workflow will replace your current settings and timeline. Continue?"
    );
    if (!confirmed) return;

    // Close the dialog immediately so the user doesn't see the review
    // content flash back while the async load is in progress.
    const wf = workflow;
    handleClose();
    await loadWorkflowDirect(wf);
  }, [workflow, showConfirm, loadWorkflowDirect, handleClose]);

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
            description: "Missing required fields: metadata or models",
          });
          return;
        }

        setWorkflow(parsed);

        const resolution = await resolveWorkflow(parsed);

        // If all dependencies are already resolved, skip the review dialog
        if (
          resolution.items.every(i => i.status === "ok") &&
          resolution.warnings.length === 0
        ) {
          await loadWorkflowDirect(parsed);
          return;
        }

        setPlan(resolution);

        // Initialize dependency states from resolution items
        initializeLoras(
          resolution.items
            .filter(i => i.kind === "lora" && i.status === "missing")
            .map(i => i.name)
        );
        initializePlugins(
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
    [initializeLoras, initializePlugins, loadWorkflowDirect]
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

  const hasUnresolvedDeps = plan?.items.some(i => i.status === "missing");

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

            {/* API key warning */}
            {missingKeyServices.length > 0 && (
              <div className="rounded-md border border-amber-500/50 bg-amber-500/10 p-3 space-y-2.5">
                <div className="flex items-start gap-2">
                  <KeyRound className="h-4 w-4 text-amber-500 mt-0.5 shrink-0" />
                  <p className="text-xs text-amber-500">
                    {missingKeyServices.length === 1
                      ? `A ${missingKeyServices[0].name} API key is required to download some LoRAs in this workflow.`
                      : "API keys are required to download some LoRAs in this workflow."}
                  </p>
                </div>
                {missingKeyServices.map(keyInfo => (
                  <div key={keyInfo.id} className="flex items-center gap-2">
                    <span className="text-xs font-medium text-foreground shrink-0 w-24">
                      {keyInfo.name}
                    </span>
                    <input
                      type="password"
                      placeholder="Paste API key..."
                      value={keyInputValues[keyInfo.id] ?? ""}
                      onChange={e =>
                        setKeyInputValues(prev => ({
                          ...prev,
                          [keyInfo.id]: e.target.value,
                        }))
                      }
                      className="flex-1 min-w-0 h-7 rounded-md border border-input bg-background px-2 text-xs placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                    />
                    <Button
                      size="sm"
                      variant="outline"
                      className="h-7 px-2 text-xs gap-1"
                      disabled={
                        !keyInputValues[keyInfo.id]?.trim() ||
                        savingKeyIds.has(keyInfo.id)
                      }
                      onClick={() => handleSaveApiKey(keyInfo)}
                    >
                      {savingKeyIds.has(keyInfo.id) ? (
                        <Loader2 className="h-3 w-3 animate-spin" />
                      ) : (
                        <Save className="h-3 w-3" />
                      )}
                      Save
                    </Button>
                    {keyInfo.key_url && (
                      <a
                        href={keyInfo.key_url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-xs text-amber-500 hover:text-amber-400 shrink-0 flex items-center gap-0.5"
                      >
                        Get key
                        <ExternalLink className="h-3 w-3" />
                      </a>
                    )}
                  </div>
                ))}
              </div>
            )}

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
                    : `Install All Missing Nodes (${installablePlugins.filter(p => plugins.installs[p.name] !== "done").length})`}
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
        onOpenChange={(open: boolean) => {
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
