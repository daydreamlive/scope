interface VACEConditioningPreviewProps {
  /** Base64 JPEG data URL for the conditioning preview */
  imageData?: string | null;
  /** Width of the preview */
  width?: number;
  /** Height of the preview */
  height?: number;
  /** Additional class names */
  className?: string;
}

/**
 * Displays the VACE conditioning preview - shows what's being sent to the model.
 */
export function VACEConditioningPreview({
  imageData,
  width = 160,
  height = 120,
  className = "",
}: VACEConditioningPreviewProps) {
  if (!imageData) {
    return null;
  }

  return (
    <div className={`flex flex-col gap-1 ${className}`}>
      <div className="flex items-center justify-between">
        <span className="text-xs text-muted-foreground">VACE Input</span>
        <span className="text-xs text-green-500">● Live</span>
      </div>
      <div
        className="relative border border-border rounded overflow-hidden bg-black"
        style={{ width, height }}
      >
        <img
          src={imageData}
          alt="VACE conditioning preview"
          className="w-full h-full object-contain"
        />
      </div>
      <div className="text-xs text-muted-foreground text-center">
        Actual conditioning frames
      </div>
    </div>
  );
}
