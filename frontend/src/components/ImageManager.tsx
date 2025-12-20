import { useState, useEffect, useRef } from "react";
import { Button } from "./ui/button";
import { Plus, X, RefreshCw, Send } from "lucide-react";
import { LabelWithTooltip } from "./ui/label-with-tooltip";
import { listImageFiles, uploadImage, type ImageFileInfo } from "../lib/api";
import { FilePicker } from "./ui/file-picker";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./ui/select";

export type ImageConditioningMode = "r2v" | "firstframe" | "lastframe";

export interface ImageConditioningItem {
  mode: ImageConditioningMode;
  imagePath: string;
}

interface ImageManagerProps {
  images: string[] | ImageConditioningItem[];
  onImagesChange: (images: string[] | ImageConditioningItem[]) => void;
  disabled?: boolean;
  isStreaming?: boolean;
  onSendHint?: (mode: ImageConditioningMode, imagePath: string) => void;
  supportsModeSelection?: boolean;
}

export function ImageManager({
  images,
  onImagesChange,
  disabled,
  isStreaming = false,
  onSendHint,
  supportsModeSelection = false,
}: ImageManagerProps) {
  const [availableImages, setAvailableImages] = useState<ImageFileInfo[]>([]);
  const [isLoadingImages, setIsLoadingImages] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadTargetIndex, setUploadTargetIndex] = useState<number | null>(
    null
  );
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Helper to check if images array contains conditioning items
  const isConditioningMode = supportsModeSelection && images.length > 0 && typeof images[0] === 'object' && 'mode' in images[0];

  // Get typed images based on mode
  const imageItems: ImageConditioningItem[] = isConditioningMode
    ? (images as ImageConditioningItem[])
    : (images as string[]).map(imagePath => ({ mode: 'r2v' as ImageConditioningMode, imagePath }));

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
    if (supportsModeSelection) {
      onImagesChange([...imageItems, { mode: "r2v", imagePath: "" }]);
    } else {
      onImagesChange([...(images as string[]), ""]);
    }
  };

  const handleRemoveImage = (index: number) => {
    if (supportsModeSelection) {
      onImagesChange(imageItems.filter((_, i) => i !== index));
    } else {
      onImagesChange((images as string[]).filter((_, i) => i !== index));
    }
  };

  const handleImageChange = (index: number, path: string) => {
    if (supportsModeSelection) {
      const newItems = [...imageItems];
      newItems[index] = { ...newItems[index], imagePath: path };
      onImagesChange(newItems);
    } else {
      const newImages = [...(images as string[])];
      newImages[index] = path;
      onImagesChange(newImages);
    }
  };

  const handleModeChange = (index: number, mode: ImageConditioningMode) => {
    if (!supportsModeSelection) return;
    const newItems = [...imageItems];
    newItems[index] = { ...newItems[index], mode };
    onImagesChange(newItems);
  };

  const handleUploadClick = (index: number) => {
    setUploadTargetIndex(index);
    fileInputRef.current?.click();
  };

  const handleFileUpload = async (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const allowedTypes = [
      "image/png",
      "image/jpeg",
      "image/jpg",
      "image/webp",
      "image/bmp",
    ];
    if (!allowedTypes.includes(file.type)) {
      console.error(
        "handleFileUpload: Invalid file type. Allowed types: PNG, JPEG, JPG, WEBP, BMP"
      );
      return;
    }

    const maxSize = 50 * 1024 * 1024;
    if (file.size > maxSize) {
      console.error(
        `handleFileUpload: File size exceeds maximum of ${maxSize / (1024 * 1024)}MB`
      );
      return;
    }

    setIsUploading(true);
    try {
      const uploadedFile = await uploadImage(file);
      if (uploadTargetIndex !== null) {
        handleImageChange(uploadTargetIndex, uploadedFile.path);
      } else {
        if (supportsModeSelection) {
          onImagesChange([...imageItems, { mode: "r2v", imagePath: uploadedFile.path }]);
        } else {
          onImagesChange([...(images as string[]), uploadedFile.path]);
        }
      }
      await loadAvailableImages();
    } catch (error) {
      console.error("handleFileUpload: Failed to upload image:", error);
    } finally {
      setIsUploading(false);
      setUploadTargetIndex(null);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  };

  const getModeLabel = (mode: ImageConditioningMode): string => {
    switch (mode) {
      case "r2v":
        return "Reference (R2V)";
      case "firstframe":
        return "First Frame";
      case "lastframe":
        return "Last Frame";
    }
  };

  const getModeTooltip = (mode: ImageConditioningMode): string => {
    switch (mode) {
      case "r2v":
        return "Reference image guides the video generation for style and character consistency. Can send hints at any time.";
      case "firstframe":
        return "Set the first frame of the video generation. The model will generate subsequent frames starting from this image.";
      case "lastframe":
        return "Set the target last frame of the video generation. The model will generate frames leading to this image.";
    }
  };

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <LabelWithTooltip
          label={supportsModeSelection ? "Image Conditioning" : "Reference Images"}
          tooltip={supportsModeSelection
            ? "Condition video generation using reference images or frame anchors. Each image can have its own mode."
            : "Select reference images for VACE conditioning. Images will guide the video generation style and content."}
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
            disabled={disabled || isStreaming}
            className="h-6 px-2"
            title={supportsModeSelection ? "Add image" : "Add reference image"}
          >
            <Plus className="h-3 w-3 mr-1" />
            Add Image
          </Button>
        </div>
      </div>

      {availableImages.length === 0 && !isLoadingImages && (
        <p className="text-xs text-muted-foreground">
          No images found. Place images in ~/.daydream-scope/images/
        </p>
      )}

      <input
        type="file"
        accept="image/png,image/jpeg,image/jpg,image/webp,image/bmp"
        onChange={handleFileUpload}
        className="hidden"
        ref={fileInputRef}
        disabled={disabled || isUploading}
      />
      <div className="space-y-2">
        {imageItems.map((item, index) => (
          <div key={index} className="rounded-lg border bg-card p-3 space-y-2">
            <div className="flex items-start gap-2">
              <div className="flex-1 min-w-0 space-y-2">
                {supportsModeSelection && (
                  <div className="flex items-center gap-2">
                    <LabelWithTooltip
                      label="Mode"
                      tooltip={getModeTooltip(item.mode)}
                      className="text-xs font-medium shrink-0"
                    />
                    <Select
                      value={item.mode}
                      onValueChange={value =>
                        handleModeChange(index, value as ImageConditioningMode)
                      }
                      disabled={disabled}
                    >
                      <SelectTrigger className="h-7 text-xs flex-1">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="r2v">
                          {getModeLabel("r2v")}
                        </SelectItem>
                        <SelectItem value="firstframe">
                          {getModeLabel("firstframe")}
                        </SelectItem>
                        <SelectItem value="lastframe">
                          {getModeLabel("lastframe")}
                        </SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                )}
                <FilePicker
                  value={item.imagePath}
                  onChange={path => handleImageChange(index, path)}
                  files={availableImages}
                  disabled={disabled}
                  placeholder="Select image file"
                  emptyMessage="No image files found"
                  onUpload={() => handleUploadClick(index)}
                  isUploading={isUploading}
                />
              </div>
              <div className="flex items-center gap-1 shrink-0">
                {onSendHint && (
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() =>
                      item.imagePath && onSendHint(item.mode, item.imagePath)
                    }
                    disabled={disabled || !isStreaming || !item.imagePath}
                    className="h-6 px-2"
                    title={
                      !isStreaming
                        ? "Start streaming to send hint"
                        : !item.imagePath
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

      {imageItems.length === 0 && (
        <p className="text-xs text-muted-foreground">
          {supportsModeSelection
            ? "No images selected. Add images to enable conditioning."
            : "No reference images selected. Add images to enable VACE conditioning."}
        </p>
      )}
    </div>
  );
}
