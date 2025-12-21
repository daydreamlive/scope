import { useState, useEffect, useRef } from "react";
import { X, Upload } from "lucide-react";
import { Button } from "./ui/button";
import {
  listAssets,
  uploadAsset,
  getAssetUrl,
  type AssetFileInfo,
} from "../lib/api";

interface MediaPickerProps {
  isOpen: boolean;
  onClose: () => void;
  onSelectImage: (imagePath: string) => void;
  disabled?: boolean;
}

export function MediaPicker({
  isOpen,
  onClose,
  onSelectImage,
  disabled,
}: MediaPickerProps) {
  const [images, setImages] = useState<AssetFileInfo[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const modalRef = useRef<HTMLDivElement>(null);

  const loadImages = async () => {
    setIsLoading(true);
    try {
      const response = await listAssets("image");
      setImages(response.assets);
    } catch (error) {
      console.error("loadImages: Failed to load images:", error);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    if (isOpen) {
      loadImages();
    }
  }, [isOpen]);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        modalRef.current &&
        !modalRef.current.contains(event.target as Node)
      ) {
        onClose();
      }
    };

    if (isOpen) {
      document.addEventListener("mousedown", handleClickOutside);
      return () =>
        document.removeEventListener("mousedown", handleClickOutside);
    }
  }, [isOpen, onClose]);

  const handleUploadClick = () => {
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
      const uploadedFile = await uploadAsset(file);
      await loadImages();
      onSelectImage(uploadedFile.path);
    } catch (error) {
      console.error("handleFileUpload: Failed to upload image:", error);
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  };

  const handleSelectImage = (imagePath: string) => {
    onSelectImage(imagePath);
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div
        ref={modalRef}
        className="bg-card border rounded-lg shadow-lg p-6 max-w-2xl w-full mx-4"
      >
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold">Media Picker</h2>
          <Button
            size="sm"
            variant="ghost"
            onClick={onClose}
            className="h-6 w-6 p-0"
          >
            <X className="h-4 w-4" />
          </Button>
        </div>

        <input
          type="file"
          accept="image/png,image/jpeg,image/jpg,image/webp,image/bmp"
          onChange={handleFileUpload}
          className="hidden"
          ref={fileInputRef}
          disabled={disabled || isUploading}
        />

        {isLoading ? (
          <div className="text-center py-12 text-muted-foreground">
            Loading images...
          </div>
        ) : (
          <div className="max-h-96 overflow-y-auto">
            <div className="grid grid-cols-3 gap-4">
              <button
                onClick={handleUploadClick}
                disabled={disabled || isUploading}
                className="aspect-square border-2 border-dashed rounded-lg flex flex-col items-center justify-center hover:bg-accent hover:border-accent-foreground disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                <Upload className="h-8 w-8 mb-2 text-muted-foreground" />
                <span className="text-sm text-muted-foreground">
                  {isUploading ? "Uploading..." : "Upload"}
                </span>
              </button>

              {images.map(image => (
                <button
                  key={image.path}
                  onClick={() => handleSelectImage(image.path)}
                  disabled={disabled}
                  className="aspect-square border rounded-lg overflow-hidden hover:ring-2 hover:ring-primary disabled:opacity-50 disabled:cursor-not-allowed transition-all relative"
                  title={image.name}
                >
                  <img
                    src={getAssetUrl(image.path)}
                    alt={image.name}
                    className="w-full h-full object-cover"
                    loading="lazy"
                  />
                </button>
              ))}

              {images.length === 0 && (
                <div className="col-span-2 text-center py-8 text-muted-foreground text-sm">
                  No images found. Upload an image to get started.
                </div>
              )}
            </div>
          </div>
        )}

        <p className="text-xs text-muted-foreground mt-4">
          {images.length > 0
            ? `${images.length} images available, sorted by most recent`
            : "No images available"}
        </p>
      </div>
    </div>
  );
}
