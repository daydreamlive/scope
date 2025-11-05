import { useState } from "react";
import { Bug, Copy, ExternalLink, Check } from "lucide-react";
import { Button } from "./ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "./ui/dialog";
import { fetchCurrentLogs } from "../lib/api";

interface ReportBugDialogProps {
  open: boolean;
  onClose: () => void;
}

export function ReportBugDialog({ open, onClose }: ReportBugDialogProps) {
  const [isLoadingLogs, setIsLoadingLogs] = useState(false);
  const [copySuccess, setCopySuccess] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleCopyLogs = async () => {
    try {
      setIsLoadingLogs(true);
      setError(null);
      setCopySuccess(false);

      // Fetch logs from backend
      const logsText = await fetchCurrentLogs();

      // Copy to clipboard
      await navigator.clipboard.writeText(logsText);

      // Show success feedback
      setCopySuccess(true);
      setTimeout(() => setCopySuccess(false), 3000);
    } catch (err) {
      console.error("Failed to copy logs:", err);
      setError(
        err instanceof Error ? err.message : "Failed to fetch or copy logs"
      );
    } finally {
      setIsLoadingLogs(false);
    }
  };

  const handleCreateBug = () => {
    window.open(
      "https://github.com/daydreamlive/scope/issues/new?template=bug-report.yml",
      "_blank"
    );
  };

  const handleOpenChange = (isOpen: boolean) => {
    if (!isOpen) {
      // Reset state when closing
      setCopySuccess(false);
      setError(null);
      onClose();
    }
  };

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Bug className="h-5 w-5" />
            Report Bug
          </DialogTitle>
          <DialogDescription className="mt-3">
            Follow these steps to report a bug with diagnostic logs.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          {/* Step 1: Copy Logs */}
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <div className="flex h-6 w-6 items-center justify-center rounded-full bg-primary text-primary-foreground text-sm font-semibold">
                1
              </div>
              <h4 className="font-semibold text-sm">Copy Logs</h4>
            </div>
            <p className="text-sm text-muted-foreground pl-8">
              Click the button below to copy logs to your clipboard.
            </p>
            <div className="pl-8">
              <Button
                onClick={handleCopyLogs}
                disabled={isLoadingLogs}
                variant="outline"
                className="gap-2"
              >
                {copySuccess ? (
                  <>
                    <Check className="h-4 w-4" />
                    Copied!
                  </>
                ) : (
                  <>
                    <Copy className="h-4 w-4" />
                    {isLoadingLogs ? "Loading..." : "Copy Logs"}
                  </>
                )}
              </Button>
            </div>
            {error && <p className="text-sm text-destructive pl-8">{error}</p>}
          </div>

          {/* Step 2: Create Bug Report */}
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <div className="flex h-6 w-6 items-center justify-center rounded-full bg-primary text-primary-foreground text-sm font-semibold">
                2
              </div>
              <h4 className="font-semibold text-sm">Create Bug Report</h4>
            </div>
            <p className="text-sm text-muted-foreground pl-8">
              Click the button below to create a bug report on GitHub and then
              attach the logs from your clipboard.
            </p>
            <div className="pl-8">
              <Button onClick={handleCreateBug} className="gap-2">
                <ExternalLink className="h-4 w-4" />
                Create Bug Report
              </Button>
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
