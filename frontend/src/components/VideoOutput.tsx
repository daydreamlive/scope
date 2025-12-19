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
  isAudioOnly?: boolean;
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
  isAudioOnly = false,
}: VideoOutputProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const audioRef = useRef<HTMLAudioElement>(null);
  const [showOverlay, setShowOverlay] = useState(false);
  const [isFadingOut, setIsFadingOut] = useState(false);
  const [isAudioPlaying, setIsAudioPlaying] = useState(false);
  const overlayTimeoutRef = useRef<number | null>(null);

  useEffect(() => {
    if (!isAudioOnly && videoRef.current && remoteStream) {
      videoRef.current.srcObject = remoteStream;
    }
  }, [remoteStream, isAudioOnly]);

  useEffect(() => {
    if (isAudioOnly && audioRef.current && remoteStream) {
      audioRef.current.srcObject = remoteStream;
    }
  }, [remoteStream, isAudioOnly]);

  // Notify when audio starts playing (parity with video)
  useEffect(() => {
    if (!isAudioOnly) return;
    const audio = audioRef.current;
    if (!audio || !remoteStream) return;

    const handlePlaying = () => {
      setIsAudioPlaying(true);
      onVideoPlaying?.();
    };

    const handlePause = () => {
      setIsAudioPlaying(false);
    };

    const handleEnded = () => {
      setIsAudioPlaying(false);
    };

    if (!audio.paused && audio.currentTime > 0 && !audio.ended) {
      setIsAudioPlaying(true);
      setTimeout(() => onVideoPlaying?.(), 0);
    }

    audio.addEventListener("playing", handlePlaying);
    audio.addEventListener("pause", handlePause);
    audio.addEventListener("ended", handleEnded);
    return () => {
      audio.removeEventListener("playing", handlePlaying);
      audio.removeEventListener("pause", handlePause);
      audio.removeEventListener("ended", handleEnded);
    };
  }, [isAudioOnly, onVideoPlaying, remoteStream]);

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
        {remoteStream && isAudioOnly ? (
          <div className="w-full h-full flex flex-col items-center justify-center">
            {/* Hidden audio element for actual playback */}
            <audio
              ref={audioRef}
              className="hidden"
              autoPlay
              playsInline
            />
            {/* Audio visualization */}
            <div className="relative flex items-center justify-center mb-6">
              {/* Animated concentric circles */}
              {isAudioPlaying && (
                <>
                  <div className="absolute w-32 h-32 rounded-full border-2 border-purple-500/30 animate-ping" />
                  <div className="absolute w-40 h-40 rounded-full border-2 border-blue-500/20 animate-ping" style={{ animationDelay: "0.2s" }} />
                  <div className="absolute w-48 h-48 rounded-full border-2 border-purple-500/10 animate-ping" style={{ animationDelay: "0.4s" }} />
                </>
              )}
              {/* Speaker icon */}
              <div className="relative z-10 w-24 h-24 flex items-center justify-center">
                <svg
                  className="w-full h-full text-white"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                  strokeWidth={2}
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M19.114 5.636a9 9 0 010 12.728M16.463 8.289a5 5 0 010 7.072M12 3v18M8 8l-4-4v12l4-4"
                  />
                </svg>
              </div>
            </div>
            {/* Status text */}
            <div className="text-center">
              <p className="text-lg font-medium text-foreground">
                {isAudioPlaying ? "Playing audio..." : "Audio ready"}
              </p>
            </div>
          </div>
        ) : remoteStream ? (
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
