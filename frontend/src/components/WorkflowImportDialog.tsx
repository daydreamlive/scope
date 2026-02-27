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
import { toast } from "sonner";
import type {
  ScopeWorkflow,
  ResolutionItem,
  WorkflowResolutionPlan,
  WorkflowLoRAProvenance,
  WorkflowLoRA,
  LoRADownloadRequest,
} from "../lib/workflowApi";
import { validateWorkflow, downloadLoRA } from "../lib/workflowApi";
import { installPlugin, restartServer, waitForServer } from "../lib/api";
import type { SettingsState } from "../types";
import type { TimelinePrompt } from "./PromptTimeline";
import {
  workflowToSettings,
  workflowTimelineToPrompts,
  workflowToPromptState,
} from "../lib/workflowSettings";
import type { WorkflowPromptState } from "../lib/workflowSettings";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type ImportStep = "select" | "review";

interface LoRADownloadState {
  [filename: string]: "idle" | "downloading" | "done" | "error";
}

interface PluginInstallState {
  [pluginName: string]: "idle" | "installing" | "done" | "error";
}

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

function buildLoRADownloadRequest(
  lora: WorkflowLoRA
): LoRADownloadRequest | null {
  const prov = lora.provenance;
  if (!prov || prov.source === "local") return null;

  if (prov.source === "huggingface") {
    return {
      source: "huggingface",
      repo_id: prov.repo_id,
      hf_filename: prov.hf_filename,
      expected_sha256: lora.sha256,
    };
  }

  if (prov.source === "civitai") {
    return {
      source: "civitai",
      model_id: prov.model_id,
      version_id: prov.version_id,
      expected_sha256: lora.sha256,
    };
  }

  if (prov.source === "url" && prov.url) {
    return {
      source: "url",
      url: prov.url,
      expected_sha256: lora.sha256,
    };
  }

  return null;
}

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

function findPluginInstallSpec(
  workflow: ScopeWorkflow,
  pluginName: string
): string | null {
  for (const p of workflow.pipelines) {
    if (p.source.plugin_name === pluginName) {
      return p.source.package_spec ?? p.source.plugin_name ?? null;
    }
  }
  return null;
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
  const [loraDownloads, setLoraDownloads] = useState<LoRADownloadState>({});
  const [pluginInstalls, setPluginInstalls] = useState<PluginInstallState>({});
  const [validating, setValidating] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Reset all state when dialog closes
  const handleClose = useCallback(() => {
    setStep("select");
    setWorkflow(null);
    setPlan(null);
    setLoraDownloads({});
    setPluginInstalls({});
    setValidating(false);
    onClose();
  }, [onClose]);

  // -----------------------------------------------------------------------
  // File selection and validation
  // -----------------------------------------------------------------------

  const handleFileSelect = useCallback(async (file: File) => {
    try {
      setValidating(true);
      const text = await file.text();
      let parsed: ScopeWorkflow;
      try {
        parsed = JSON.parse(text);
      } catch {
        toast.error("Invalid JSON file");
        setValidating(false);
        return;
      }

      if (parsed.format !== "scope-workflow") {
        toast.error("Not a Scope workflow file", {
          description: 'Expected format: "scope-workflow"',
        });
        setValidating(false);
        return;
      }

      setWorkflow(parsed);

      const resolution = await validateWorkflow(parsed);
      setPlan(resolution);

      // Initialize LoRA download states
      const initialDownloads: LoRADownloadState = {};
      for (const item of resolution.items) {
        if (item.kind === "lora" && item.status === "missing") {
          initialDownloads[item.name] = "idle";
        }
      }
      setLoraDownloads(initialDownloads);

      // Initialize plugin install states
      const initialPlugins: PluginInstallState = {};
      for (const item of resolution.items) {
        if (
          item.kind === "plugin" &&
          item.status === "missing" &&
          item.can_auto_resolve
        ) {
          initialPlugins[item.name] = "idle";
        }
      }
      setPluginInstalls(initialPlugins);
      setStep("review");
    } catch (err) {
      console.error("Workflow validation failed:", err);
      toast.error("Validation failed", {
        description: err instanceof Error ? err.message : String(err),
      });
    } finally {
      setValidating(false);
    }
  }, []);

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
  // LoRA download
  // -----------------------------------------------------------------------

  const handleDownloadLoRA = useCallback(
    async (filename: string) => {
      if (!workflow) return;

      // Find the LoRA in the workflow
      const lora = workflow.pipelines
        .flatMap(p => p.loras)
        .find(l => l.filename === filename);
      if (!lora) return;

      const req = buildLoRADownloadRequest(lora);
      if (!req) {
        toast.error("No download source available for this LoRA");
        return;
      }

      setLoraDownloads(prev => ({ ...prev, [filename]: "downloading" }));
      try {
        await downloadLoRA(req);
        setLoraDownloads(prev => ({ ...prev, [filename]: "done" }));
        toast.success(`Downloaded ${filename}`);
      } catch (err) {
        setLoraDownloads(prev => ({ ...prev, [filename]: "error" }));
        toast.error(`Failed to download ${filename}`, {
          description: err instanceof Error ? err.message : String(err),
        });
      }
    },
    [workflow]
  );

  const handleDownloadAllLoRAs = useCallback(async () => {
    if (!workflow) return;
    const missing = Object.entries(loraDownloads)
      .filter(([, s]) => s === "idle" || s === "error")
      .map(([name]) => name);

    await Promise.allSettled(missing.map(f => handleDownloadLoRA(f)));
  }, [workflow, loraDownloads, handleDownloadLoRA]);

  // -----------------------------------------------------------------------
  // Plugin install
  // -----------------------------------------------------------------------

  const doRestartServer = useCallback(async () => {
    toast.info("Restarting server to load new plugins...");
    try {
      const oldStartTime = await restartServer();
      await waitForServer(oldStartTime);
      toast.success("Server restarted successfully");
    } catch {
      toast.error(
        "Server did not restart in time. You may need to restart manually."
      );
    }
  }, []);

  const handleInstallPlugin = useCallback(
    async (pluginName: string, { skipRestart = false } = {}) => {
      if (!workflow) return;

      const installSpec = findPluginInstallSpec(workflow, pluginName);
      if (!installSpec) {
        toast.error("No install source available for this plugin");
        return;
      }

      setPluginInstalls(prev => ({ ...prev, [pluginName]: "installing" }));
      try {
        const result = await installPlugin({ package: installSpec });
        if (!result.success) {
          throw new Error(result.message || "Installation failed");
        }
        setPluginInstalls(prev => ({ ...prev, [pluginName]: "done" }));
        toast.success(`Installed ${pluginName}`);
        if (!skipRestart) {
          await doRestartServer();
        }
      } catch (err) {
        setPluginInstalls(prev => ({ ...prev, [pluginName]: "error" }));
        toast.error(`Failed to install ${pluginName}`, {
          description: err instanceof Error ? err.message : String(err),
        });
      }
    },
    [workflow, doRestartServer]
  );

  const handleInstallAllPlugins = useCallback(async () => {
    if (!workflow) return;
    const missing = Object.entries(pluginInstalls)
      .filter(([, s]) => s === "idle" || s === "error")
      .map(([name]) => name);

    await Promise.allSettled(
      missing.map(name => handleInstallPlugin(name, { skipRestart: true }))
    );
    await doRestartServer();
  }, [workflow, pluginInstalls, handleInstallPlugin, doRestartServer]);

  // -----------------------------------------------------------------------
  // Load workflow into the interface (no pipeline loading)
  // -----------------------------------------------------------------------

  const handleLoad = useCallback(() => {
    if (!workflow) return;

    const importedSettings = workflowToSettings(workflow);
    const timelinePrompts = workflowTimelineToPrompts(workflow.timeline);
    const promptState = workflowToPromptState(workflow);

    onLoad(importedSettings, timelinePrompts, promptState);
    toast.success("Workflow loaded", {
      description: `"${workflow.metadata.name}" loaded into the interface`,
    });
    handleClose();
  }, [workflow, onLoad, handleClose]);

  // -----------------------------------------------------------------------
  // Derived state
  // -----------------------------------------------------------------------

  const missingLoRAs = plan?.items.filter(
    i => i.kind === "lora" && i.status === "missing"
  );
  const downloadableLoRAs = missingLoRAs?.filter(i => i.can_auto_resolve);
  const someLoRAsDownloading = Object.values(loraDownloads).some(
    s => s === "downloading"
  );

  const missingPlugins = plan?.items.filter(
    i => i.kind === "plugin" && i.status === "missing"
  );
  const installablePlugins = missingPlugins?.filter(i => i.can_auto_resolve);
  const somePluginsInstalling = Object.values(pluginInstalls).some(
    s => s === "installing"
  );

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
                          {loraDownloads[item.name] === "done" ? (
                            <span className="text-xs text-green-500 flex items-center gap-1">
                              <CheckCircle2 className="h-3 w-3" />
                              Downloaded
                            </span>
                          ) : loraDownloads[item.name] === "downloading" ? (
                            <span className="text-xs text-muted-foreground flex items-center gap-1">
                              <Loader2 className="h-3 w-3 animate-spin" />
                              Downloading...
                            </span>
                          ) : (
                            <Button
                              variant="ghost"
                              size="sm"
                              className="h-6 text-xs px-2"
                              onClick={() => handleDownloadLoRA(item.name)}
                            >
                              <Download className="h-3 w-3 mr-1" />
                              {loraDownloads[item.name] === "error"
                                ? "Retry"
                                : "Download"}
                            </Button>
                          )}
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
                          {pluginInstalls[item.name] === "done" ? (
                            <span className="text-xs text-green-500 flex items-center gap-1">
                              <CheckCircle2 className="h-3 w-3" />
                              Installed
                            </span>
                          ) : pluginInstalls[item.name] === "installing" ? (
                            <span className="text-xs text-muted-foreground flex items-center gap-1">
                              <Loader2 className="h-3 w-3 animate-spin" />
                              Installing...
                            </span>
                          ) : (
                            <Button
                              variant="ghost"
                              size="sm"
                              className="h-6 text-xs px-2"
                              onClick={() => handleInstallPlugin(item.name)}
                            >
                              <Download className="h-3 w-3 mr-1" />
                              {pluginInstalls[item.name] === "error"
                                ? "Retry"
                                : "Install"}
                            </Button>
                          )}
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
              missingLoRAs.some(l => loraDownloads[l.name] !== "done") && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleDownloadAllLoRAs}
                  disabled={someLoRAsDownloading}
                >
                  <Download className="h-4 w-4 mr-2" />
                  {someLoRAsDownloading
                    ? "Downloading..."
                    : `Download All Missing LoRAs (${downloadableLoRAs.filter(l => loraDownloads[l.name] !== "done").length})`}
                </Button>
              )}

            {/* Install all plugins button */}
            {installablePlugins &&
              installablePlugins.length > 1 &&
              missingPlugins &&
              missingPlugins.some(p => pluginInstalls[p.name] !== "done") && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleInstallAllPlugins}
                  disabled={somePluginsInstalling}
                >
                  <Download className="h-4 w-4 mr-2" />
                  {somePluginsInstalling
                    ? "Installing..."
                    : `Install All Missing Plugins (${installablePlugins.filter(p => pluginInstalls[p.name] !== "done").length})`}
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
              disabled={someLoRAsDownloading || somePluginsInstalling}
            >
              Load Workflow
            </Button>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
