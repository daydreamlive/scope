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
import type { DownloadProgress, PipelineInfo } from "../types";

interface DownloadDialogProps {
  open: boolean;
  pipelines: Record<string, PipelineInfo> | null;
  pipelineIds: string[];
  currentDownloadPipeline: string | null;
  onClose: () => void;
  onDownload: () => void;
  isDownloading?: boolean;
  progress?: DownloadProgress | null;
}

export function DownloadDialog({
  open,
  pipelines,
  pipelineIds,
  currentDownloadPipeline,
  onClose,
  onDownload,
  isDownloading = false,
  progress = null,
}: DownloadDialogProps) {
  if (pipelineIds.length === 0) return null;

  // Calculate total estimated VRAM for all pipelines
  const totalVram = pipelineIds.reduce((sum, id) => {
    const info = pipelines?.[id];
    return sum + (info?.estimatedVram ?? 0);
  }, 0);

  // Get current pipeline info if downloading
  const currentPipelineInfo = currentDownloadPipeline
    ? (pipelines?.[currentDownloadPipeline] ?? null)
    : null;

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
              ? currentDownloadPipeline
                ? `Downloading models for ${currentPipelineInfo?.name || currentDownloadPipeline}...`
                : "Please wait while models are downloaded."
              : "The following pipeline(s) require models to be downloaded:"}
          </DialogDescription>
        </DialogHeader>

        {/* List of missing pipelines/preprocessors */}
        {!isDownloading && (
          <div className="space-y-2 mb-3">
            {pipelineIds.map(id => {
              const info = pipelines?.[id];
              const isPreprocessor =
                info?.usage?.includes("preprocessor") ?? false;
              return (
                <div
                  key={id}
                  className="flex items-center justify-between rounded border bg-background p-2 text-sm"
                >
                  <div className="flex-1">
                    <div className="font-medium">
                      {info?.name || id}
                      {isPreprocessor && (
                        <span className="ml-2 text-xs text-muted-foreground">
                          (Preprocessor)
                        </span>
                      )}
                    </div>
                    {info?.estimatedVram && (
                      <div className="text-xs text-muted-foreground mt-1">
                        Estimated VRAM: {info.estimatedVram} GB
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {!isDownloading && totalVram > 0 && (
          <p className="text-sm text-muted-foreground mb-3">
            <span className="font-semibold">Total Estimated GPU VRAM:</span>{" "}
            {totalVram} GB
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

        {/* Show remaining pipelines count if downloading */}
        {isDownloading && currentDownloadPipeline && pipelineIds.length > 1 && (
          <p className="text-sm text-muted-foreground">
            {pipelineIds.length - 1} more pipeline(s) will be downloaded after
            this one.
          </p>
        )}

        <DialogFooter>
          {!isDownloading && (
            <Button onClick={onDownload} className="gap-2">
              <Download className="h-4 w-4" />
              Download All
            </Button>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
