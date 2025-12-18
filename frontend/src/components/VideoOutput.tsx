import { useEffect, useRef, useState, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Spinner } from "./ui/spinner";
import { PlayOverlay } from "./ui/play-overlay";
import { getDownloadLogs } from "../lib/api";

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
}: VideoOutputProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [showOverlay, setShowOverlay] = useState(false);
  const [isFadingOut, setIsFadingOut] = useState(false);
  const overlayTimeoutRef = useRef<number | null>(null);
  const [downloadLogs, setDownloadLogs] = useState<string>("");
  const logsTextAreaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (videoRef.current && remoteStream) {
      videoRef.current.srcObject = remoteStream;
    }
  }, [remoteStream]);

  // Poll for download logs when downloading
  useEffect(() => {
    if (!isDownloading) {
      return;
    }

    const pollLogs = async () => {
      try {
        const response = await getDownloadLogs();
        setDownloadLogs(response.logs);
      } catch (error) {
        console.error("Failed to fetch download logs:", error);
      }
    };

    // Poll immediately and then every 500ms
    pollLogs();
    const interval = setInterval(pollLogs, 500);

    return () => clearInterval(interval);
  }, [isDownloading]);

  // Auto-scroll logs to bottom when new content arrives
  useEffect(() => {
    if (logsTextAreaRef.current) {
      logsTextAreaRef.current.scrollTop = logsTextAreaRef.current.scrollHeight;
    }
  }, [downloadLogs]);

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
    triggerPlayPause();
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
            className="relative w-full h-full cursor-pointer flex items-center justify-center"
            onClick={handleVideoClick}
          >
            <video
              ref={videoRef}
              className="max-w-full max-h-full object-contain"
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
          </div>
        ) : isDownloading ? (
          <div className="flex flex-col items-center justify-center w-full h-full gap-3 p-4">
            <div className="flex items-center gap-2 text-muted-foreground text-lg">
              <Spinner size={20} />
              <span>Downloading models...</span>
            </div>
            <textarea
              ref={logsTextAreaRef}
              readOnly
              value={downloadLogs || "Waiting for download to start..."}
              className="w-full flex-1 min-h-[120px] max-h-[300px] p-3 text-xs font-mono bg-muted/50 border border-border rounded-md resize-none text-muted-foreground overflow-y-auto"
              style={{ whiteSpace: "pre-wrap", wordBreak: "break-all" }}
            />
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
