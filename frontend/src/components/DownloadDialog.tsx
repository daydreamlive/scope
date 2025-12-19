import { Download, Loader2 } from "lucide-react";
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
import type { PipelineId, DownloadProgress } from "../types";

interface DownloadDialogProps {
  open: boolean;
  pipelineId: PipelineId;
  onClose: () => void;
  onDownload: () => void;
  isDownloading?: boolean;
  progress?: DownloadProgress | null;
}

export function DownloadDialog({
  open,
  pipelineId,
  onClose,
  onDownload,
  isDownloading = false,
  progress = null,
}: DownloadDialogProps) {
  const pipelineInfo = PIPELINES[pipelineId];
  if (!pipelineInfo) return null;

  return (
    <Dialog
      open={open}
      onOpenChange={isOpen => !isOpen && !isDownloading && onClose()}
    >
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>
            {isDownloading ? "Downloading Models..." : "Download Models"}
          </DialogTitle>
          <DialogDescription className="mt-3">
            {isDownloading
              ? "Please wait while models are downloaded."
              : "This pipeline requires models to be downloaded."}
          </DialogDescription>
        </DialogHeader>

        {!isDownloading && pipelineInfo.estimatedVram && (
          <p className="text-sm text-muted-foreground mb-3">
            <span className="font-semibold">
              Estimated GPU VRAM Requirement:
            </span>{" "}
            {pipelineInfo.estimatedVram} GB
          </p>
        )}

        {/* Progress UI */}
        {isDownloading && (
          <div className="flex flex-col items-center justify-center py-6 gap-4">
            <Loader2 className="h-10 w-10 animate-spin text-primary" />
            <div className="text-center">
              <p className="text-lg font-medium">
                {progress
                  ? `${(progress.total_downloaded_mb / 1024).toFixed(2)} GB`
                  : "Starting..."}
              </p>
              <p className="text-sm text-muted-foreground">Total Downloaded</p>
            </div>
          </div>
        )}

        <DialogFooter>
          {!isDownloading && (
            <Button onClick={onDownload} className="gap-2">
              <Download className="h-4 w-4" />
              Download
            </Button>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
