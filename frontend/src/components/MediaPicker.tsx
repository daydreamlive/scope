import { useState, useEffect, useRef, useCallback } from "react";
import { X, Upload, Film, Music } from "lucide-react";
import { Button } from "./ui/button";
import {
  listAssets,
  uploadAsset,
  getAssetUrl,
  type AssetFileInfo,
} from "../lib/api";
import { useCloudStatus } from "../hooks/useCloudStatus";
import { isVideoAsset, isAudioAsset } from "../lib/mediaUtils";

interface MediaPickerProps {
  isOpen: boolean;
  onClose: () => void;
  onSelectImage: (imagePath: string) => void;
  disabled?: boolean;
  /** Which asset types to show. Default "image". */
  accept?: "image" | "video" | "audio" | "all";
}

export function MediaPicker({
  isOpen,
  onClose,
  onSelectImage,
  disabled,
  accept = "image",
}: MediaPickerProps) {
  const [assets, setAssets] = useState<AssetFileInfo[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const modalRef = useRef<HTMLDivElement>(null);

  const loadAssets = useCallback(async () => {
    setIsLoading(true);
    try {
      if (accept === "all") {
        const [imgRes, vidRes, audRes] = await Promise.all([
          listAssets("image"),
          listAssets("video"),
          listAssets("audio"),
        ]);
        const merged = [
          ...imgRes.assets,
          ...vidRes.assets,
          ...audRes.assets,
        ].sort((a, b) => b.created_at - a.created_at);
        setAssets(merged);
      } else {
        const response = await listAssets(accept);
        setAssets(response.assets);
      }
    } catch (error) {
      console.error("loadAssets: Failed to load assets:", error);
    } finally {
      setIsLoading(false);
    }
  }, [accept]);

  useEffect(() => {
    if (isOpen) {
      loadAssets();
    }
  }, [isOpen, loadAssets]);

  // Refresh asset list when cloud connection state changes while picker is open
  const { isConnected: isCloudConnected } = useCloudStatus();
  const prevCloudConnectedRef = useRef<boolean | null>(null);

  useEffect(() => {
    // Skip first render
    if (prevCloudConnectedRef.current === null) {
      prevCloudConnectedRef.current = isCloudConnected;
      return;
    }

    // If connection state changed and picker is open, reload assets
    if (prevCloudConnectedRef.current !== isCloudConnected && isOpen) {
      loadAssets();
    }

    prevCloudConnectedRef.current = isCloudConnected;
  }, [isCloudConnected, isOpen, loadAssets]);

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

  const imageTypes = [
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/webp",
    "image/bmp",
  ];
  const videoTypes = ["video/mp4", "video/webm", "video/quicktime"];
  const audioTypes = ["audio/wav", "audio/mpeg", "audio/flac", "audio/ogg"];

  const allowedTypes =
    accept === "all"
      ? [...imageTypes, ...videoTypes, ...audioTypes]
      : accept === "video"
        ? videoTypes
        : accept === "audio"
          ? audioTypes
          : imageTypes;
  const fileAcceptAttr = allowedTypes.join(",");

  const handleFileUpload = async (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    const file = event.target.files?.[0];
    if (!file) return;

    if (!allowedTypes.includes(file.type)) {
      console.error(
        `handleFileUpload: Invalid file type "${file.type}". Allowed: ${allowedTypes.join(", ")}`
      );
      return;
    }

    const maxCloudUploadSize = 50 * 1024 * 1024;
    if (isCloudConnected && file.size > maxCloudUploadSize) {
      console.error(
        `handleFileUpload: File size exceeds maximum of ${maxCloudUploadSize / (1024 * 1024)}MB while connected to cloud`
      );
      return;
    }

    setIsUploading(true);
    try {
      const uploadedFile = await uploadAsset(file);
      await loadAssets();
      onSelectImage(uploadedFile.path);
    } catch (error) {
      console.error("handleFileUpload: Failed to upload asset:", error);
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  };

  const handleSelectAsset = (path: string) => {
    onSelectImage(path);
  };

  if (!isOpen) return null;

  const titleMap = {
    image: "Image Picker",
    video: "Video Picker",
    audio: "Audio Picker",
    all: "Media Picker",
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div
        ref={modalRef}
        className="bg-card border rounded-lg shadow-lg p-6 max-w-2xl w-full mx-4"
      >
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold">{titleMap[accept]}</h2>
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
          accept={fileAcceptAttr}
          onChange={handleFileUpload}
          className="hidden"
          ref={fileInputRef}
          disabled={disabled || isUploading}
        />

        {isLoading ? (
          <div className="text-center py-12 text-muted-foreground">
            Loading assets...
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

              {assets.map(asset => {
                const isVideo = isVideoAsset(asset.name);
                const isAudio = isAudioAsset(asset.name);
                return (
                  <button
                    key={asset.path}
                    onClick={() => handleSelectAsset(asset.path)}
                    disabled={disabled}
                    className="aspect-square border rounded-lg overflow-hidden hover:ring-2 hover:ring-primary disabled:opacity-50 disabled:cursor-not-allowed transition-all relative"
                    title={asset.name}
                  >
                    {isAudio ? (
                      <div className="w-full h-full flex flex-col items-center justify-center bg-[#1a1a1a]">
                        <Music className="h-10 w-10 text-emerald-400 mb-1" />
                        <span className="text-[10px] text-[#999] truncate max-w-[90%] px-1">
                          {asset.name}
                        </span>
                      </div>
                    ) : isVideo ? (
                      <div className="w-full h-full flex flex-col items-center justify-center bg-[#1a1a1a]">
                        <Film className="h-10 w-10 text-blue-400 mb-1" />
                        <span className="text-[10px] text-[#999] truncate max-w-[90%] px-1">
                          {asset.name}
                        </span>
                      </div>
                    ) : (
                      <img
                        src={getAssetUrl(asset.path)}
                        alt={asset.name}
                        className="w-full h-full object-cover"
                        loading="lazy"
                      />
                    )}
                  </button>
                );
              })}

              {assets.length === 0 && (
                <div className="col-span-2 text-center py-8 text-muted-foreground text-sm">
                  No assets found. Upload to get started.
                </div>
              )}
            </div>
          </div>
        )}

        <p className="text-xs text-muted-foreground mt-4">
          {assets.length > 0
            ? `${assets.length} assets available, sorted by most recent`
            : "No assets available"}
        </p>
      </div>
    </div>
  );
}
