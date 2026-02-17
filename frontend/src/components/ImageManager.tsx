import { useState, useEffect } from "react";
import { Plus, X } from "lucide-react";
import { LabelWithTooltip } from "./ui/label-with-tooltip";
import { useApi } from "../hooks/useApi";
import { getAssetUrl } from "../lib/api";
import { MediaPicker } from "./MediaPicker";

/** Helper component that loads an asset image, using data URL in cloud mode */
function AssetImage({
  assetPath,
  alt,
  isDirectCloudMode,
  getAssetDataUrl,
}: {
  assetPath: string;
  alt: string;
  isDirectCloudMode: boolean;
  getAssetDataUrl: (path: string) => Promise<string>;
}) {
  const [src, setSrc] = useState<string | null>(null);

  useEffect(() => {
    if (isDirectCloudMode) {
      getAssetDataUrl(assetPath)
        .then(setSrc)
        .catch(() => setSrc(null));
    } else {
      setSrc(getAssetUrl(assetPath));
    }
  }, [assetPath, isDirectCloudMode, getAssetDataUrl]);

  if (!src) {
    return (
      <div className="w-full h-full flex items-center justify-center bg-muted">
        <div className="animate-pulse w-8 h-8 bg-muted-foreground/20 rounded" />
      </div>
    );
  }

  return <img src={src} alt={alt} className="w-full h-full object-cover" />;
}

interface ImageManagerProps {
  images: string[];
  onImagesChange: (images: string[]) => void;
  disabled?: boolean;
  /** Maximum number of images allowed. When set to 1, replaces instead of adding. */
  maxImages?: number;
  /** Label for the component */
  label?: string;
  /** Tooltip for the label */
  tooltip?: string;
  /** Hide the label */
  hideLabel?: boolean;
  /** Use single column layout (full width). Defaults to true when maxImages=1, false otherwise. */
  singleColumn?: boolean;
}

export function ImageManager({
  images,
  onImagesChange,
  disabled,
  maxImages,
  label = "Reference Images",
  tooltip = "Select reference images for VACE conditioning. Images will guide the video generation style and content.",
  hideLabel = false,
  singleColumn,
}: ImageManagerProps) {
  const { isDirectCloudMode, getAssetDataUrl } = useApi();
  const [isMediaPickerOpen, setIsMediaPickerOpen] = useState(false);

  const handleAddImage = (imagePath: string) => {
    if (maxImages === 1) {
      // Single image mode - replace
      onImagesChange([imagePath]);
    } else {
      onImagesChange([...images, imagePath]);
    }
  };

  const handleRemoveImage = (index: number) => {
    onImagesChange(images.filter((_, i) => i !== index));
  };

  const canAddMore = maxImages === undefined || images.length < maxImages;

  return (
    <div>
      {!hideLabel && (
        <LabelWithTooltip
          label={label}
          tooltip={tooltip}
          className="text-sm font-medium mb-2 block"
        />
      )}

      <div
        className={
          (singleColumn ?? maxImages === 1)
            ? "grid grid-cols-1"
            : "grid grid-cols-2 gap-2"
        }
      >
        {images.length === 0 && (
          <button
            onClick={() => setIsMediaPickerOpen(true)}
            disabled={disabled}
            className="aspect-square border-2 border-dashed rounded-lg flex flex-col items-center justify-center hover:bg-accent hover:border-accent-foreground disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            <Plus className="h-6 w-6 mb-1 text-muted-foreground" />
            <span className="text-xs text-muted-foreground">Add Image</span>
          </button>
        )}

        {images.map((imagePath, index) => (
          <div
            key={index}
            className="aspect-square border rounded-lg overflow-hidden relative group"
          >
            <AssetImage
              assetPath={imagePath}
              alt={`${label} ${index + 1}`}
              isDirectCloudMode={isDirectCloudMode}
              getAssetDataUrl={getAssetDataUrl}
            />
            <button
              onClick={() => handleRemoveImage(index)}
              disabled={disabled}
              className="absolute top-1 right-1 bg-black/70 hover:bg-black text-white rounded p-1 opacity-0 group-hover:opacity-100 transition-opacity disabled:opacity-50"
              title="Remove image"
            >
              <X className="h-3 w-3" />
            </button>
          </div>
        ))}

        {/* Show add button if we have images but can add more (multi-image mode) */}
        {images.length > 0 && canAddMore && maxImages !== 1 && (
          <button
            onClick={() => setIsMediaPickerOpen(true)}
            disabled={disabled}
            className="aspect-square border-2 border-dashed rounded-lg flex flex-col items-center justify-center hover:bg-accent hover:border-accent-foreground disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            <Plus className="h-6 w-6 mb-1 text-muted-foreground" />
            <span className="text-xs text-muted-foreground">Add Image</span>
          </button>
        )}
      </div>

      <MediaPicker
        isOpen={isMediaPickerOpen}
        onClose={() => setIsMediaPickerOpen(false)}
        onSelectImage={handleAddImage}
        disabled={disabled}
      />
    </div>
  );
}
