import { Download, Share2 } from "lucide-react";
import { Button } from "./ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "./ui/dialog";

interface ExportDialogProps {
  open: boolean;
  onClose: () => void;
  onSaveGeneration: () => void;
  onSaveTimeline: () => void;
  isRecording?: boolean;
}

export function ExportDialog({
  open,
  onClose,
  onSaveGeneration,
  onSaveTimeline,
  isRecording = false,
}: ExportDialogProps) {
  return (
    <Dialog open={open} onOpenChange={isOpen => !isOpen && onClose()}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Export</DialogTitle>
          <DialogDescription className="mt-3">
            Choose what you want to export
          </DialogDescription>
        </DialogHeader>

        <div className="flex flex-col gap-3 mt-4">
          {isRecording && (
            <Button
              onClick={() => {
                onSaveGeneration();
                onClose();
              }}
              variant="outline"
              className="w-full justify-start gap-3 px-4 py-6"
            >
              <Download className="h-4 w-4" />
              <div className="flex flex-col items-start">
                <span className="font-semibold">Save Generation</span>
                <span className="text-xs text-muted-foreground">
                  Downloads MP4 to default Downloads folder
                </span>
              </div>
            </Button>
          )}

          <Button
            onClick={() => {
              onSaveTimeline();
              onClose();
            }}
            variant="outline"
            className="w-full justify-start gap-3 px-4 py-6"
          >
            <Share2 className="h-4 w-4" />
            <div className="flex flex-col items-start">
              <span className="font-semibold">Export Workflow</span>
              <span className="text-xs text-muted-foreground">
                Save pipeline settings, LoRAs, and timeline as a shareable file
              </span>
            </div>
          </Button>
        </div>

        <DialogFooter>
          <Button variant="ghost" onClick={onClose}>
            Cancel
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
