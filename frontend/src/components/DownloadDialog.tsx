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
import { PIPELINES } from "../data/pipelines";
import type { PipelineId } from "../types";

interface DownloadDialogProps {
  open: boolean;
  pipelineId: PipelineId;
  onClose: () => void;
  onDownload: () => void;
}

export function DownloadDialog({
  open,
  pipelineId,
  onClose,
  onDownload,
}: DownloadDialogProps) {
  const pipelineInfo = PIPELINES[pipelineId];
  if (!pipelineInfo) return null;

  return (
    <Dialog open={open} onOpenChange={isOpen => !isOpen && onClose()}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Download Pipeline Models</DialogTitle>
          <DialogDescription>{pipelineInfo.about}</DialogDescription>
        </DialogHeader>

        {pipelineInfo.estimatedVram && (
          <p className="text-sm text-muted-foreground">
            Estimated required VRAM: {pipelineInfo.estimatedVram} GB
          </p>
        )}

        <DialogFooter>
          <Button onClick={onDownload} className="gap-2">
            <Download className="h-4 w-4" />
            Download
            {pipelineInfo.estimatedDownloadSize
              ? ` (${pipelineInfo.estimatedDownloadSize} GB)`
              : ""}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
