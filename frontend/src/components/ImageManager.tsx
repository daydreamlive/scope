import { useState, useEffect } from "react";
import { Button } from "./ui/button";
import { Plus, X, RefreshCw, Send } from "lucide-react";
import { LabelWithTooltip } from "./ui/label-with-tooltip";
import { listImageFiles, type ImageFileInfo } from "../lib/api";
import { FilePicker } from "./ui/file-picker";

interface ImageManagerProps {
  images: string[];
  onImagesChange: (images: string[]) => void;
  disabled?: boolean;
  isStreaming?: boolean;
  onSendHint?: (imagePath: string) => void;
}

export function ImageManager({
  images,
  onImagesChange,
  disabled,
  isStreaming = false,
  onSendHint,
}: ImageManagerProps) {
  const [availableImages, setAvailableImages] = useState<ImageFileInfo[]>([]);
  const [isLoadingImages, setIsLoadingImages] = useState(false);

  const loadAvailableImages = async () => {
    setIsLoadingImages(true);
    try {
      const response = await listImageFiles();
      setAvailableImages(response.image_files);
    } catch (error) {
      console.error("loadAvailableImages: Failed to load image files:", error);
    } finally {
      setIsLoadingImages(false);
    }
  };

  useEffect(() => {
    loadAvailableImages();
  }, []);

  const handleAddImage = () => {
    onImagesChange([...images, ""]);
  };

  const handleRemoveImage = (index: number) => {
    onImagesChange(images.filter((_, i) => i !== index));
  };

  const handleImageChange = (index: number, path: string) => {
    const newImages = [...images];
    newImages[index] = path;
    onImagesChange(newImages);
  };

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <LabelWithTooltip
          label="Reference Images"
          tooltip="Select reference images for VACE conditioning. Images will guide the video generation style and content."
          className="text-sm font-medium"
        />
        <div className="flex items-center gap-2">
          <Button
            size="sm"
            variant="ghost"
            onClick={loadAvailableImages}
            disabled={disabled || isLoadingImages}
            className="h-6 w-6 p-0"
            title="Refresh image list"
          >
            <RefreshCw className="h-3 w-3" />
          </Button>
          <Button
            size="sm"
            onClick={handleAddImage}
            disabled={disabled}
            className="h-6 px-2"
            title="Add reference image"
          >
            <Plus className="h-3 w-3 mr-1" />
            Add Image
          </Button>
        </div>
      </div>

      {availableImages.length === 0 && !isLoadingImages && (
        <p className="text-xs text-muted-foreground">
          No images found. Place images in ~/.daydream-scope/models/images/
        </p>
      )}

      <div className="space-y-2">
        {images.map((imagePath, index) => (
          <div key={index} className="rounded-lg border bg-card p-3 space-y-2">
            <div className="flex items-center justify-between gap-2">
              <div className="flex-1 min-w-0">
                <FilePicker
                  value={imagePath}
                  onChange={path => handleImageChange(index, path)}
                  files={availableImages}
                  disabled={disabled}
                  placeholder="Select image file"
                  emptyMessage="No image files found"
                />
              </div>
              <div className="flex items-center gap-1 shrink-0">
                {onSendHint && (
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => imagePath && onSendHint(imagePath)}
                    disabled={disabled || !isStreaming || !imagePath}
                    className="h-6 px-2"
                    title={
                      !isStreaming
                        ? "Start streaming to send hint"
                        : !imagePath
                          ? "Select an image first"
                          : "Send hint"
                    }
                  >
                    <Send className="h-3 w-3 mr-1" />
                    Send Hint
                  </Button>
                )}
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => handleRemoveImage(index)}
                  disabled={disabled}
                  className="h-6 w-6 p-0 shrink-0"
                  title="Remove image"
                >
                  <X className="h-3 w-3" />
                </Button>
              </div>
            </div>
          </div>
        ))}
      </div>

      {images.length === 0 && (
        <p className="text-xs text-muted-foreground">
          No reference images selected. Add images to enable VACE conditioning.
        </p>
      )}
    </div>
  );
}
