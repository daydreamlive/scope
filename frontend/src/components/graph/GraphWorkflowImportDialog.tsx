import { useState, useCallback } from "react";
import { AlertTriangle, Download, Loader2 } from "lucide-react";
import { Button } from "../ui/button";
import { Badge } from "../ui/badge";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "../ui/dialog";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "../ui/alert-dialog";
import { toast } from "sonner";
import type {
  ScopeWorkflow,
  WorkflowResolutionPlan,
} from "../../lib/workflowApi";
import {
  useLoRADownloads,
  usePluginInstalls,
} from "../../hooks/useWorkflowDependencies";
import { DependencyStatusIndicator } from "../DependencyStatusIndicator";
import {
  statusIcon,
  kindLabel,
  LoRAProvenanceLabel,
} from "../workflowDialogHelpers";

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

interface GraphWorkflowImportDialogProps {
  workflow: ScopeWorkflow | null;
  plan: WorkflowResolutionPlan | null;
  resolving: boolean;
  onConfirm: () => void;
  onCancel: () => void;
  onReResolve: () => Promise<void>;
}

export function GraphWorkflowImportDialog({
  workflow,
  plan,
  resolving,
  onConfirm,
  onCancel,
  onReResolve,
}: GraphWorkflowImportDialogProps) {
  const open = workflow !== null;

  // -- Re-resolution callback
  const reResolveWorkflow = useCallback(async () => {
    await onReResolve();
  }, [onReResolve]);

  const loras = useLoRADownloads(workflow, reResolveWorkflow);
  const { reset: resetLoras } = loras;

  // -- Confirm dialog state
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
  const { reset: resetPlugins } = plugins;

  const handleClose = useCallback(() => {
    resetLoras();
    resetPlugins();
    onCancel();
  }, [onCancel, resetLoras, resetPlugins]);

  const handleLoad = useCallback(() => {
    toast.success("Workflow loaded into graph", {
      description: workflow?.metadata?.name
        ? `"${workflow.metadata.name}" loaded`
        : undefined,
    });
    resetLoras();
    resetPlugins();
    onConfirm();
  }, [workflow, onConfirm, resetLoras, resetPlugins]);

  // -- Derived state
  const missingLoRAs = plan?.items.filter(
    i => i.kind === "lora" && i.status === "missing"
  );
  const downloadableLoRAs = missingLoRAs?.filter(i => i.can_auto_resolve);

  const missingPlugins = plan?.items.filter(
    i => i.kind === "plugin" && i.status === "missing"
  );
  const installablePlugins = missingPlugins?.filter(i => i.can_auto_resolve);

  const hasUnresolvedDeps = plan?.items.some(i => i.status === "missing");

  return (
    <Dialog open={open} onOpenChange={isOpen => !isOpen && handleClose()}>
      <DialogContent className="sm:max-w-lg max-h-[80vh] flex flex-col">
        <DialogHeader>
          <DialogTitle>Review Workflow</DialogTitle>
          <DialogDescription>
            Review dependencies before loading into the graph.
          </DialogDescription>
        </DialogHeader>

        {resolving && (
          <div className="flex items-center justify-center gap-2 py-8">
            <Loader2 className="h-6 w-6 text-muted-foreground animate-spin" />
            <p className="text-sm text-muted-foreground">
              Resolving dependencies...
            </p>
          </div>
        )}

        {!resolving && plan && workflow && (
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
          {!resolving && plan && (
            <Button
              onClick={handleLoad}
              disabled={
                loras.someDownloading ||
                plugins.someInstalling ||
                hasUnresolvedDeps
              }
            >
              Load into Graph
            </Button>
          )}
        </DialogFooter>
      </DialogContent>

      {/* Confirmation alert dialog */}
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
