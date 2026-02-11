import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../ui/select";
import { ArrowUp } from "lucide-react";
import { LabelWithTooltip } from "../ui/label-with-tooltip";
import { ImageManager } from "../ImageManager";
import { Button } from "../ui/button";
import type { ExtensionMode } from "../../types";

interface ExtensionFramesSectionProps {
  firstFrameImage?: string;
  lastFrameImage?: string;
  extensionMode: ExtensionMode;
  isDownloading: boolean;
  isStreaming: boolean;
  onFirstFrameImageChange?: (image: string | undefined) => void;
  onLastFrameImageChange?: (image: string | undefined) => void;
  onExtensionModeChange?: (mode: ExtensionMode) => void;
  onSendExtensionFrames?: () => void;
}

export function ExtensionFramesSection({
  firstFrameImage,
  lastFrameImage,
  extensionMode,
  isDownloading,
  isStreaming,
  onFirstFrameImageChange,
  onLastFrameImageChange,
  onExtensionModeChange,
  onSendExtensionFrames,
}: ExtensionFramesSectionProps) {
  return (
    <div>
      <LabelWithTooltip
        label="Extension Frames"
        tooltip="Set reference frames for video extension. First frame starts the video from that image, last frame generates toward that target."
        className="text-sm font-medium mb-2 block"
      />
      <div className="grid grid-cols-2 gap-2">
        <div className="space-y-1">
          <span className="text-xs text-muted-foreground">First Frame</span>
          <ImageManager
            images={firstFrameImage ? [firstFrameImage] : []}
            onImagesChange={images => {
              onFirstFrameImageChange?.(images[0] || undefined);
            }}
            disabled={isDownloading}
            maxImages={1}
            label="First Frame"
            hideLabel
          />
        </div>
        <div className="space-y-1">
          <span className="text-xs text-muted-foreground">Last Frame</span>
          <ImageManager
            images={lastFrameImage ? [lastFrameImage] : []}
            onImagesChange={images => {
              onLastFrameImageChange?.(images[0] || undefined);
            }}
            disabled={isDownloading}
            maxImages={1}
            label="Last Frame"
            hideLabel
          />
        </div>
      </div>
      {(firstFrameImage || lastFrameImage) && (
        <div className="space-y-2 mt-2">
          <div className="flex items-center justify-between gap-2">
            <span className="text-xs text-muted-foreground">Mode:</span>
            <Select
              value={extensionMode}
              onValueChange={value => {
                if (value && onExtensionModeChange) {
                  onExtensionModeChange(value as ExtensionMode);
                }
              }}
              disabled={!firstFrameImage && !lastFrameImage}
            >
              <SelectTrigger className="w-24 h-6 text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {firstFrameImage && (
                  <SelectItem value="firstframe">First</SelectItem>
                )}
                {lastFrameImage && (
                  <SelectItem value="lastframe">Last</SelectItem>
                )}
                {firstFrameImage && lastFrameImage && (
                  <SelectItem value="firstlastframe">Both</SelectItem>
                )}
              </SelectContent>
            </Select>
          </div>
          <div className="flex items-center justify-end">
            <Button
              onMouseDown={e => {
                e.preventDefault();
                onSendExtensionFrames?.();
              }}
              disabled={
                isDownloading ||
                !isStreaming ||
                (!firstFrameImage && !lastFrameImage)
              }
              size="sm"
              className="rounded-full w-8 h-8 p-0 bg-black hover:bg-gray-800 text-white disabled:opacity-50 disabled:cursor-not-allowed"
              title={
                !isStreaming
                  ? "Start streaming to send extension frames"
                  : "Send extension frames"
              }
            >
              <ArrowUp className="h-4 w-4" />
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}
