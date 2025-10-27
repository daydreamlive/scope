import { X } from "lucide-react";
import { Button } from "./ui/button";
import { PIPELINES } from "../data/pipelines";
import type { PipelineId } from "../types";

interface DownloadDialogProps {
  pipelineId: PipelineId;
  onClose: () => void;
  onDownload: () => void;
}

export function DownloadDialog({
  pipelineId,
  onClose,
  onDownload,
}: DownloadDialogProps) {
  const pipelineInfo = PIPELINES[pipelineId];
  if (!pipelineInfo) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/50" onClick={onClose} />

      {/* Dialog */}
      <div className="relative bg-white rounded-lg shadow-lg border border-black max-w-md w-full mx-4">
        {/* Close Button */}
        <button
          onClick={onClose}
          className="absolute top-4 left-4 p-1 hover:bg-gray-100 rounded"
          aria-label="Close"
        >
          <X className="h-5 w-5" />
        </button>

        {/* Content */}
        <div className="p-6 pt-10">
          <p className="text-sm text-gray-700 mb-4">{pipelineInfo.about}</p>

          {pipelineInfo.estimatedVram && (
            <p className="text-sm text-gray-700 mb-6">
              Estimated required VRAM: {pipelineInfo.estimatedVram} GB
            </p>
          )}

          <div className="flex justify-end">
            <Button onClick={onDownload} className="gap-2">
              <svg
                xmlns="http://www.w3.org/2000/svg"
                width="16"
                height="16"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="7 10 12 15 17 10" />
                <line x1="12" y1="15" x2="12" y2="3" />
              </svg>
              Download
              {pipelineInfo.estimatedDownloadSize
                ? ` (${pipelineInfo.estimatedDownloadSize} GB)`
                : ""}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
