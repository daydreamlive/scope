import { Download } from "lucide-react";
import { Button } from "./ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "./ui/dialog";
import type { PipelineId } from "../types";
import type { PipelineInfo } from "../hooks/usePipelines";

interface DownloadDialogProps {
  open: boolean;
  pipelines: Record<string, PipelineInfo> | null;
  pipelineId: PipelineId;
  onClose: () => void;
  onDownload: () => void;
}

export function DownloadDialog({
  open,
  pipelines,
  pipelineId,
  onClose,
  onDownload,
}: DownloadDialogProps) {
  const pipelineInfo = pipelines?.[pipelineId];
  if (!pipelineInfo) return null;

  return (
    <Dialog open={open} onOpenChange={isOpen => !isOpen && onClose()}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Download Models</DialogTitle>
          <DialogDescription className="mt-3">
            This pipeline requires model weights to be downloaded.
          </DialogDescription>
        </DialogHeader>

        {pipelineInfo.estimatedVram && (
          <p className="text-sm text-muted-foreground mb-3">
            <span className="font-semibold">
              Estimated GPU VRAM Requirement:
            </span>{" "}
            {pipelineInfo.estimatedVram} GB
          </p>
        )}

        <DialogFooter>
          <Button onClick={onDownload} className="gap-2">
            <Download className="h-4 w-4" />
            Download
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
