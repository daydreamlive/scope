import { useState } from "react";
import { Download } from "lucide-react";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "./ui/dialog";
import { toast } from "sonner";
import type { SettingsState } from "../types";
import type { TimelinePrompt } from "./PromptTimeline";
import type { WorkflowPromptState } from "../lib/workflowSettings";
import {
  buildExportPipelines,
  getExportMergeMode,
  buildWorkflowTimeline,
  annotateWorkflowRoles,
  annotateWorkflowPromptState,
} from "../lib/workflowSettings";
import { exportWorkflow } from "../lib/workflowApi";

interface WorkflowExportDialogProps {
  open: boolean;
  onClose: () => void;
  settings: SettingsState;
  timelinePrompts: TimelinePrompt[];
  promptState: WorkflowPromptState;
}

export function WorkflowExportDialog({
  open,
  onClose,
  settings,
  timelinePrompts,
  promptState,
}: WorkflowExportDialogProps) {
  const [name, setName] = useState("Untitled Workflow");
  const [exporting, setExporting] = useState(false);

  const handleExport = async () => {
    setExporting(true);
    try {
      const pipelines = buildExportPipelines(settings);
      const loraMergeMode = getExportMergeMode(settings);
      const timeline = buildWorkflowTimeline(timelinePrompts);

      const workflow = await exportWorkflow({
        name,
        pipelines,
        lora_merge_mode: loraMergeMode,
        timeline,
      });

      annotateWorkflowRoles(workflow, settings);
      annotateWorkflowPromptState(workflow, promptState);

      // Download as JSON file
      const blob = new Blob([JSON.stringify(workflow, null, 2)], {
        type: "application/json",
      });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      const safeName = name.replace(/[^a-zA-Z0-9_-]/g, "_").toLowerCase();
      link.download = `${safeName}.scope-workflow.json`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);

      toast.success("Workflow exported", {
        description: `"${name}" saved as .scope-workflow.json`,
      });
      onClose();
    } catch (err) {
      console.error("Workflow export failed:", err);
      toast.error("Export failed", {
        description: err instanceof Error ? err.message : String(err),
      });
    } finally {
      setExporting(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={isOpen => !isOpen && onClose()}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Export Workflow</DialogTitle>
          <DialogDescription>
            Save your current pipeline configuration, settings
            {timelinePrompts.length > 0 ? ", and timeline" : ""} as a shareable
            workflow file.
          </DialogDescription>
        </DialogHeader>

        <div className="flex flex-col gap-4 py-2">
          <div className="flex flex-col gap-2">
            <label
              htmlFor="workflow-name"
              className="text-sm font-medium text-foreground"
            >
              Workflow name
            </label>
            <Input
              id="workflow-name"
              value={name}
              onChange={e => setName(e.target.value)}
              placeholder="My Workflow"
              autoFocus
              onKeyDown={e => {
                if (e.key === "Enter" && name.trim()) {
                  handleExport();
                }
              }}
            />
          </div>

          <div className="text-xs text-muted-foreground space-y-1">
            <p>
              Pipeline:{" "}
              <span className="font-medium">{settings.pipelineId}</span>
            </p>
            {(settings.loras?.length ?? 0) > 0 && (
              <p>LoRAs: {settings.loras!.length} adapter(s) included</p>
            )}
            {timelinePrompts.length > 0 && (
              <p>Timeline: {timelinePrompts.length} prompt(s) included</p>
            )}
          </div>
        </div>

        <DialogFooter>
          <Button variant="ghost" onClick={onClose} disabled={exporting}>
            Cancel
          </Button>
          <Button onClick={handleExport} disabled={exporting || !name.trim()}>
            <Download className="h-4 w-4 mr-2" />
            {exporting ? "Exporting..." : "Export"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
