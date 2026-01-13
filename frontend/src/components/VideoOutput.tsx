import { useEffect, useRef, useState, useCallback } from "react";
import { Volume2, VolumeX } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Spinner } from "./ui/spinner";
import { PlayOverlay } from "./ui/play-overlay";
import { Slider } from "./ui/slider";

interface VideoOutputProps {
  className?: string;
  remoteStream: MediaStream | null;
  isPipelineLoading?: boolean;
  isConnecting?: boolean;
  pipelineError?: string | null;
  isPlaying?: boolean;
  isDownloading?: boolean;
  hasAudio?: boolean;
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
  hasAudio = false,
  onPlayPauseToggle,
  onStartStream,
  onVideoPlaying,
}: VideoOutputProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [showOverlay, setShowOverlay] = useState(false);
  const [isFadingOut, setIsFadingOut] = useState(false);
  const overlayTimeoutRef = useRef<number | null>(null);

  // Audio state - start muted for autoplay compatibility
  const [isMuted, setIsMuted] = useState(true);
  const [volume, setVolume] = useState(0.7);
  const [showVolumeSlider, setShowVolumeSlider] = useState(false);

  useEffect(() => {
    if (videoRef.current && remoteStream) {
      videoRef.current.srcObject = remoteStream;
    }
  }, [remoteStream]);

  // Update video element volume when volume state changes
  useEffect(() => {
    if (videoRef.current) {
      videoRef.current.volume = volume;
      videoRef.current.muted = isMuted;
    }
  }, [volume, isMuted]);

  // Toggle mute/unmute
  const toggleMute = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation(); // Prevent triggering play/pause
      if (videoRef.current) {
        const newMuted = !isMuted;
        videoRef.current.muted = newMuted;
        setIsMuted(newMuted);
      }
    },
    [isMuted]
  );

  // Handle volume change
  const handleVolumeChange = useCallback((values: number[]) => {
    const newVolume = values[0];
    setVolume(newVolume);
    if (videoRef.current) {
      videoRef.current.volume = newVolume;
      // Unmute if volume is changed while muted
      if (newVolume > 0 && videoRef.current.muted) {
        videoRef.current.muted = false;
        setIsMuted(false);
      }
    }
  }, []);

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
              muted={isMuted}
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

            {/* Audio Controls - only show if stream has audio */}
            {hasAudio && (
              <div
                className="absolute bottom-3 right-3 flex items-center gap-2"
                onMouseEnter={() => setShowVolumeSlider(true)}
                onMouseLeave={() => setShowVolumeSlider(false)}
                onClick={e => e.stopPropagation()}
              >
                {/* Volume Slider - shows on hover */}
                <div
                  className={`transition-all duration-200 overflow-hidden ${
                    showVolumeSlider ? "w-20 opacity-100" : "w-0 opacity-0"
                  }`}
                >
                  <Slider
                    value={[isMuted ? 0 : volume]}
                    min={0}
                    max={1}
                    step={0.05}
                    onValueChange={handleVolumeChange}
                    className="w-full"
                  />
                </div>

                {/* Mute/Unmute Button */}
                <button
                  onClick={toggleMute}
                  className="p-2 bg-black/60 hover:bg-black/80 rounded-full transition-colors"
                  title={isMuted ? "Unmute" : "Mute"}
                >
                  {isMuted ? (
                    <VolumeX className="w-5 h-5 text-white" />
                  ) : (
                    <Volume2 className="w-5 h-5 text-white" />
                  )}
                </button>
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
