import { useEffect, useRef, useState, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Spinner } from "./ui/spinner";
import { PlayOverlay } from "./ui/play-overlay";

interface VideoOutputProps {
  className?: string;
  remoteStream: MediaStream | null;
  isPipelineLoading?: boolean;
  isConnecting?: boolean;
  pipelineError?: string | null;
  isPlaying?: boolean;
  isDownloading?: boolean;
  onPlayPauseToggle?: () => void;
  onStartStream?: () => void;
  onVideoPlaying?: () => void;
  // Controller input props
  supportsControllerInput?: boolean;
  isPointerLocked?: boolean;
  onRequestPointerLock?: () => void;
  /** Ref to expose the video container element for pointer lock */
  videoContainerRef?: React.RefObject<HTMLDivElement | null>;
  /** Video scale mode: 'fit' fills available space, 'native' shows at actual resolution */
  videoScaleMode?: "fit" | "native";
}

export function VideoOutput({
  className = "",
  remoteStream,
  isPipelineLoading = false,
  isConnecting = false,
  pipelineError: _pipelineError = null,
  isPlaying = true,
  isDownloading = false,
  onPlayPauseToggle,
  onStartStream,
  onVideoPlaying,
  supportsControllerInput = false,
  isPointerLocked = false,
  onRequestPointerLock,
  videoContainerRef,
  videoScaleMode = "fit",
}: VideoOutputProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const internalContainerRef = useRef<HTMLDivElement>(null);
  const [showOverlay, setShowOverlay] = useState(false);
  const [isFadingOut, setIsFadingOut] = useState(false);
  const overlayTimeoutRef = useRef<number | null>(null);

  // Use external ref if provided, otherwise use internal
  const containerRef = videoContainerRef || internalContainerRef;

  useEffect(() => {
    if (videoRef.current && remoteStream) {
      videoRef.current.srcObject = remoteStream;
    }
  }, [remoteStream]);

  // Listen for video playing event to notify parent
  useEffect(() => {
    const video = videoRef.current;
    if (!video || !remoteStream) return;

    const handlePlaying = () => {
      onVideoPlaying?.();
    };

    // Check if video is already playing when effect runs
    // This handles cases where the video was already playing before the callback was set
    if (!video.paused && video.currentTime > 0 && !video.ended) {
      // Use setTimeout to avoid calling during render
      setTimeout(() => onVideoPlaying?.(), 0);
    }

    video.addEventListener("playing", handlePlaying);
    return () => {
      video.removeEventListener("playing", handlePlaying);
    };
  }, [onVideoPlaying, remoteStream]);

  const triggerPlayPause = useCallback(() => {
    if (onPlayPauseToggle && remoteStream) {
      onPlayPauseToggle();

      // Show overlay and immediately start fade out animation
      setShowOverlay(true);
      setIsFadingOut(false);

      if (overlayTimeoutRef.current) {
        clearTimeout(overlayTimeoutRef.current);
      }

      // Start fade out immediately (CSS transition handles the timing)
      requestAnimationFrame(() => {
        setIsFadingOut(true);
      });

      // Remove overlay after animation completes (400ms transition)
      overlayTimeoutRef.current = setTimeout(() => {
        setShowOverlay(false);
        setIsFadingOut(false);
      }, 400);
    }
  }, [onPlayPauseToggle, remoteStream]);

  const handleVideoClick = () => {
    // If controller input is supported and not locked, request pointer lock
    if (supportsControllerInput && !isPointerLocked && onRequestPointerLock) {
      onRequestPointerLock();
      return;
    }

    // Otherwise toggle play/pause
    if (!isPointerLocked) {
      triggerPlayPause();
    }
  };

  // Handle spacebar press for play/pause
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Only trigger if spacebar is pressed and stream is active
      if (e.code === "Space" && remoteStream) {
        // Don't trigger if user is typing in an input/textarea/select or any contenteditable element
        const target = e.target as HTMLElement;
        const isInputFocused =
          target.tagName === "INPUT" ||
          target.tagName === "TEXTAREA" ||
          target.tagName === "SELECT" ||
          target.isContentEditable;

        if (!isInputFocused) {
          // Prevent default spacebar behavior (page scroll)
          e.preventDefault();
          triggerPlayPause();
        }
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [remoteStream, triggerPlayPause]);

  // Cleanup timeout on unmount
  useEffect(() => {
    return () => {
      if (overlayTimeoutRef.current) {
        clearTimeout(overlayTimeoutRef.current);
      }
    };
  }, []);

  return (
    <Card className={`h-full flex flex-col ${className}`}>
      <CardHeader className="flex-shrink-0">
        <CardTitle className="text-base font-medium">Video Output</CardTitle>
      </CardHeader>
      <CardContent className="flex-1 flex items-center justify-center min-h-0 p-4">
        {remoteStream ? (
          <div
            ref={containerRef}
            className="relative w-full h-full cursor-pointer flex items-center justify-center"
            onClick={handleVideoClick}
          >
            <video
              ref={videoRef}
              className={
                videoScaleMode === "fit"
                  ? "w-full h-full object-contain"
                  : "max-w-full max-h-full object-contain"
              }
              autoPlay
              muted
              playsInline
            />
            {/* Play/Pause Overlay */}
            {showOverlay && (
              <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                <div
                  className={`transition-all duration-400 ${
                    isFadingOut
                      ? "opacity-0 scale-150"
                      : "opacity-100 scale-100"
                  }`}
                >
                  <PlayOverlay isPlaying={isPlaying} size="lg" />
                </div>
              </div>
            )}
            {/* Controller Input Overlay - only show before pointer lock (browser shows ESC hint) */}
            {supportsControllerInput && !isPointerLocked && (
              <div className="absolute bottom-4 left-1/2 -translate-x-1/2 bg-black/70 text-white px-4 py-2 rounded-lg text-sm pointer-events-none">
                Click to enable controller input
              </div>
            )}
          </div>
        ) : isDownloading ? (
          <div className="text-center text-muted-foreground text-lg">
            <Spinner size={24} className="mx-auto mb-3" />
            <p>Downloading...</p>
          </div>
        ) : isPipelineLoading ? (
          <div className="text-center text-muted-foreground text-lg">
            <Spinner size={24} className="mx-auto mb-3" />
            <p>Loading...</p>
          </div>
        ) : isConnecting ? (
          <div className="text-center text-muted-foreground text-lg">
            <Spinner size={24} className="mx-auto mb-3" />
            <p>Connecting...</p>
          </div>
        ) : (
          <div className="relative w-full h-full flex items-center justify-center">
            {/* YouTube-style play button overlay */}
            <PlayOverlay
              isPlaying={false}
              onClick={onStartStream}
              size="lg"
              variant="themed"
            />
          </div>
        )}
      </CardContent>
    </Card>
  );
}
