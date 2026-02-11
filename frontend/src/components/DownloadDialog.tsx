import {
  AlertCircle,
  Download,
  Loader2,
  RotateCcw,
  Settings,
} from "lucide-react";
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
import { useAppStore } from "../stores";
import { useShallow } from "zustand/react/shallow";
import { usePipelinesContext } from "../contexts/PipelinesContext";
import { useStreamContext } from "../contexts/StreamContext";

export function DownloadDialog() {
  const {
    showDownloadDialog,
    isDownloading,
    downloadProgress,
    pipelinesNeedingModels,
    currentDownloadPipeline,
    downloadError,
    setShowDownloadDialog,
    setOpenSettingsTab,
  } = useAppStore(
    useShallow(s => ({
      showDownloadDialog: s.showDownloadDialog,
      isDownloading: s.isDownloading,
      downloadProgress: s.downloadProgress,
      pipelinesNeedingModels: s.pipelinesNeedingModels,
      currentDownloadPipeline: s.currentDownloadPipeline,
      downloadError: s.downloadError,
      setShowDownloadDialog: s.setShowDownloadDialog,
      setOpenSettingsTab: s.setOpenSettingsTab,
    }))
  );

  // Read from contexts
  const { pipelines } = usePipelinesContext();
  const { actions } = useStreamContext();

  if (pipelinesNeedingModels.length === 0) return null;

  // Calculate total estimated VRAM for all pipelines
  const totalVram = pipelinesNeedingModels.reduce((sum, id) => {
    const info = pipelines?.[id];
    return sum + (info?.estimatedVram ?? 0);
  }, 0);

  // Get current pipeline info if downloading
  const currentPipelineInfo = currentDownloadPipeline
    ? (pipelines?.[currentDownloadPipeline] ?? null)
    : null;

  const handleOpenSettings = (tab: string) => {
    setShowDownloadDialog(false);
    setOpenSettingsTab(tab);
  };

  return (
    <Dialog
      open={showDownloadDialog}
      onOpenChange={isOpen =>
        !isOpen && !isDownloading && actions.handleDialogClose()
      }
    >
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>
            {downloadError
              ? "Download Failed"
              : isDownloading
                ? "Downloading Models..."
                : "Download Models"}
          </DialogTitle>
          <DialogDescription className="mt-3">
            {downloadError
              ? "An error occurred while downloading models."
              : isDownloading
                ? currentDownloadPipeline
                  ? `Downloading models for ${currentPipelineInfo?.name || currentDownloadPipeline}...`
                  : "Please wait while models are downloaded."
                : "The following pipeline(s) require models to be downloaded:"}
          </DialogDescription>
        </DialogHeader>

        {/* List of missing pipelines/preprocessors */}
        {!isDownloading && !downloadError && (
          <div className="space-y-2 mb-3">
            {pipelinesNeedingModels.map(id => {
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

        {!isDownloading && !downloadError && totalVram > 0 && (
          <p className="text-sm text-muted-foreground mb-3">
            <span className="font-semibold">Total Estimated GPU VRAM:</span>{" "}
            {totalVram} GB
          </p>
        )}

        {/* Progress UI */}
        {isDownloading &&
          downloadProgress &&
          downloadProgress.current_artifact && (
            <div className="">
              <div className="text-sm text-muted-foreground"></div>
              <div>
                <div className="flex justify-between text-sm mb-2">
                  <span className="text-muted-foreground font-sm">
                    {downloadProgress.current_artifact}{" "}
                  </span>
                  <span className="text-muted-foreground">
                    {downloadProgress.percentage.toFixed(1)}%
                  </span>
                </div>
                <Progress value={downloadProgress.percentage} />
              </div>
            </div>
          )}

        {/* Loading State (no progress data yet) */}
        {isDownloading && !downloadProgress && !downloadError && (
          <div className="flex items-center justify-center py-8">
            <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
          </div>
        )}

        {/* Error State */}
        {downloadError && (
          <div className="rounded border border-destructive/50 bg-destructive/10 p-3">
            <div className="flex items-start gap-2">
              <AlertCircle className="h-5 w-5 text-destructive mt-0.5 shrink-0" />
              <div className="space-y-2">
                <p className="text-sm text-destructive">{downloadError}</p>
                {downloadError.toLowerCase().includes("authentication") && (
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => handleOpenSettings("api-keys")}
                    className="gap-1.5"
                  >
                    <Settings className="h-3.5 w-3.5" />
                    Open Settings &gt; API Keys
                  </Button>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Show remaining pipelines count if downloading */}
        {isDownloading &&
          currentDownloadPipeline &&
          pipelinesNeedingModels.length > 1 && (
            <p className="text-sm text-muted-foreground">
              {pipelinesNeedingModels.length - 1} more pipeline(s) will be
              downloaded after this one.
            </p>
          )}

        <DialogFooter>
          {downloadError && (
            <Button onClick={actions.handleDownloadModels} className="gap-2">
              <RotateCcw className="h-4 w-4" />
              Retry Download
            </Button>
          )}
          {!isDownloading && !downloadError && (
            <Button onClick={actions.handleDownloadModels} className="gap-2">
              <Download className="h-4 w-4" />
              Download All
            </Button>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
