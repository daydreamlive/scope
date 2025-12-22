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
import { Progress } from "./ui/progress";
import type { PipelineId, DownloadProgress } from "../types";
import type { PipelineInfo } from "../hooks/usePipelines";

interface DownloadDialogProps {
  open: boolean;
  pipelines: Record<string, PipelineInfo> | null;
  pipelineId: PipelineId;
  onClose: () => void;
  onDownload: () => void;
  isDownloading?: boolean;
  progress?: DownloadProgress | null;
}

export function DownloadDialog({
  open,
  pipelines,
  pipelineId,
  onClose,
  onDownload,
  isDownloading = false,
  progress = null,
}: DownloadDialogProps) {
  const pipelineInfo = pipelines?.[pipelineId];
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
        {isDownloading && progress && progress.current_artifact && (
          <div className="">
            <div className="text-sm text-muted-foreground"></div>
            <div>
              <div className="flex justify-between text-sm mb-2">
                <span className="text-muted-foreground font-sm">
                  {progress.current_artifact}{" "}
                </span>
                <span className="text-muted-foreground">
                  {progress.percentage.toFixed(1)}%
                </span>
              </div>
              <Progress value={progress.percentage} />
            </div>
          </div>
        )}

        {/* Loading State (no progress data yet) */}
        {isDownloading && !progress && (
          <div className="flex items-center justify-center py-8">
            <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
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
