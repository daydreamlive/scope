import { useState, useEffect, useCallback, useRef } from "react";

export type VideoSourceMode = "video" | "camera" | "spout";

interface UseVideoSourceProps {
  onStreamUpdate?: (stream: MediaStream) => Promise<boolean>;
  onStopStream?: () => void;
  shouldReinitialize?: boolean;
  enabled?: boolean;
  // Called when a custom video is uploaded with its detected resolution
  onCustomVideoResolution?: (resolution: {
    width: number;
    height: number;
  }) => void;
}

// Standardized FPS for both video and camera modes
export const FPS = 15;
export const MIN_FPS = 5;
export const MAX_FPS = 30;

export function useVideoSource(props?: UseVideoSourceProps) {
  const [localStream, setLocalStream] = useState<MediaStream | null>(null);
  const [isInitializing, setIsInitializing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [mode, setMode] = useState<VideoSourceMode>("video");
  const [selectedVideoFile, setSelectedVideoFile] = useState<string | File>(
    "/assets/test.mp4"
  );
  const [videoResolution, setVideoResolution] = useState<{
    width: number;
    height: number;
  } | null>(null);

  const videoElementRef = useRef<HTMLVideoElement | null>(null);

  const createVideoFromSource = useCallback((videoSource: string | File) => {
    const video = document.createElement("video");

    if (typeof videoSource === "string") {
      video.src = videoSource;
    } else {
      video.src = URL.createObjectURL(videoSource);
    }

    video.loop = true;
    video.muted = true;
    video.playsInline = true;
    video.autoplay = true;
    videoElementRef.current = video;
    return video;
  }, []);

  const createVideoFileStreamFromFile = useCallback(
    (videoSource: string | File, fps: number) => {
      const video = createVideoFromSource(videoSource);

      return new Promise<{
        stream: MediaStream;
        resolution: { width: number; height: number };
      }>((resolve, reject) => {
        // Add timeout to prevent hanging promises
        const timeout = setTimeout(() => {
          reject(new Error("Video loading timeout"));
        }, 10000); // 10 second timeout

        video.onloadedmetadata = () => {
          clearTimeout(timeout);
          try {
            // Detect and store video resolution
            const detectedResolution = {
              width: video.videoWidth,
              height: video.videoHeight,
            };
            setVideoResolution(detectedResolution);

            // Create canvas matching the video resolution
            const canvas = document.createElement("canvas");
            canvas.width = detectedResolution.width;
            canvas.height = detectedResolution.height;
            const ctx = canvas.getContext("2d")!;

            video
              .play()
              .then(() => {
                // Draw video frame to canvas at original resolution
                const drawFrame = () => {
                  ctx.drawImage(
                    video,
                    0,
                    0,
                    detectedResolution.width,
                    detectedResolution.height
                  );
                  if (!video.paused && !video.ended) {
                    requestAnimationFrame(drawFrame);
                  }
                };
                drawFrame();

                // Capture stream from canvas at original resolution
                const stream = canvas.captureStream(fps);

                resolve({ stream, resolution: detectedResolution });
              })
              .catch(error => {
                clearTimeout(timeout);
                reject(error);
              });
          } catch (error) {
            clearTimeout(timeout);
            reject(error);
          }
        };

        video.onerror = () => {
          clearTimeout(timeout);
          reject(new Error("Failed to load video file"));
        };
      });
    },
    [createVideoFromSource]
  );

  const createVideoFileStream = useCallback(
    async (fps: number) => {
      const result = await createVideoFileStreamFromFile(
        selectedVideoFile,
        fps
      );
      return result.stream;
    },
    [createVideoFileStreamFromFile, selectedVideoFile]
  );

  const requestCameraAccess = useCallback(async () => {
    try {
      setError(null);
      setIsInitializing(true);

      // Request camera access - browser will handle device selection
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 512, min: 256, max: 512 },
          height: { ideal: 512, min: 256, max: 512 },
          frameRate: { ideal: FPS, min: MIN_FPS, max: MAX_FPS },
        },
        audio: false,
      });

      setVideoResolution({ width: 512, height: 512 });
      setLocalStream(stream);
      setIsInitializing(false);
      return stream;
    } catch (error) {
      console.error("Failed to request camera access:", error);
      setError(
        error instanceof Error ? error.message : "Failed to access camera"
      );
      setIsInitializing(false);
      return null;
    }
  }, []);

  const switchMode = useCallback(
    async (newMode: VideoSourceMode) => {
      // Don't switch modes if not enabled
      if (!props?.enabled) {
        return;
      }

      setMode(newMode);
      setError(null);

      // Spout mode - no local stream needed, input comes from Spout receiver
      if (newMode === "spout") {
        // Stop current stream
        if (localStream) {
          localStream.getTracks().forEach(track => track.stop());
        }
        if (videoElementRef.current) {
          videoElementRef.current.pause();
          videoElementRef.current = null;
        }
        setLocalStream(null);
        return;
      }

      let newStream: MediaStream | null = null;

      if (newMode === "video") {
        // Create video file stream
        try {
          newStream = await createVideoFileStream(FPS);
        } catch (error) {
          console.error("Failed to create video file stream:", error);
          setError("Failed to load test video");
        }
      } else if (newMode === "camera") {
        // Switch to camera mode
        try {
          newStream = await requestCameraAccess();
        } catch (error) {
          console.error("Failed to switch to camera mode:", error);
          // Error is already set by requestCameraAccess
        }
      }

      if (newStream) {
        // Try to update WebRTC track if streaming, otherwise just switch locally
        let trackReplaced = false;
        if (props?.onStreamUpdate) {
          trackReplaced = await props.onStreamUpdate(newStream);
        }

        // If track replacement failed and we're streaming, stop the stream
        // Otherwise, just switch locally
        if (!trackReplaced && props?.onStreamUpdate && props?.onStopStream) {
          // Track replacement failed - stop stream to allow clean switch
          props.onStopStream();
        }

        // Stop current stream only after successful replacement or if not streaming
        if (localStream && (trackReplaced || !props?.onStreamUpdate)) {
          localStream.getTracks().forEach(track => track.stop());
        }

        // Stop video element if switching away from video mode
        if (videoElementRef.current && newMode === "camera") {
          videoElementRef.current.pause();
          videoElementRef.current = null;
        }

        setLocalStream(newStream);
      }
    },
    [localStream, createVideoFileStream, requestCameraAccess, props]
  );

  const handleVideoFileUpload = useCallback(
    async (file: File) => {
      // Validate file size (10MB limit)
      const maxSize = 10 * 1024 * 1024; // 10MB in bytes
      if (file.size > maxSize) {
        setError("File size must be less than 10MB");
        return false;
      }

      // Validate file type
      if (!file.type.startsWith("video/")) {
        setError("Please select a video file");
        return false;
      }

      setError(null);

      // Create new stream directly with the uploaded file (avoid race condition)
      try {
        setIsInitializing(true);
        const { stream: newStream, resolution } =
          await createVideoFileStreamFromFile(file, FPS);

        // Try to update WebRTC track if streaming, otherwise just switch locally
        let trackReplaced = false;
        if (props?.onStreamUpdate) {
          trackReplaced = await props.onStreamUpdate(newStream);
        }

        // If track replacement failed and we're streaming, stop the stream
        // Otherwise, just switch locally
        if (!trackReplaced && props?.onStreamUpdate && props?.onStopStream) {
          // Track replacement failed - stop stream to allow clean switch
          props.onStopStream();
        }

        // Stop current stream only after successful replacement or if not streaming
        if (localStream && (trackReplaced || !props?.onStreamUpdate)) {
          localStream.getTracks().forEach(track => track.stop());
        }

        // Update selected video file only after successful stream creation
        setSelectedVideoFile(file);
        setLocalStream(newStream);
        setIsInitializing(false);

        // Notify about custom video resolution so caller can sync output resolution
        props?.onCustomVideoResolution?.(resolution);

        return true;
      } catch (error) {
        console.error("Failed to create stream from uploaded file:", error);
        setError("Failed to load uploaded video file");
        setIsInitializing(false);
        return false;
      }
    },
    [localStream, createVideoFileStreamFromFile, props]
  );

  const stopVideo = useCallback(() => {
    if (localStream) {
      localStream.getTracks().forEach(track => track.stop());
      setLocalStream(null);
    }

    if (videoElementRef.current) {
      videoElementRef.current.pause();
      videoElementRef.current = null;
    }
  }, [localStream]);

  const reinitializeVideoSource = useCallback(async () => {
    setIsInitializing(true);
    setError(null);

    // Ensure we're in video mode when reinitializing
    setMode("video");

    try {
      // Stop current stream if it exists
      if (localStream) {
        localStream.getTracks().forEach(track => track.stop());
      }

      // Create new video file stream
      const stream = await createVideoFileStream(FPS);
      setLocalStream(stream);
    } catch (error) {
      console.error("Failed to reinitialize video source:", error);
      setError("Failed to load test video");
    } finally {
      setIsInitializing(false);
    }
  }, [localStream, createVideoFileStream]);

  // Initialize with video mode on mount (only if enabled)
  useEffect(() => {
    if (!props?.enabled) {
      // If not enabled, stop any existing stream and clear state
      if (localStream) {
        localStream.getTracks().forEach(track => track.stop());
        setLocalStream(null);
      }
      if (videoElementRef.current) {
        videoElementRef.current.pause();
        videoElementRef.current = null;
      }
      return;
    }

    const initializeVideoMode = async () => {
      setIsInitializing(true);
      try {
        const stream = await createVideoFileStream(FPS);
        setLocalStream(stream);
      } catch (error) {
        console.error("Failed to create initial video file stream:", error);
        setError("Failed to load test video");
      } finally {
        setIsInitializing(false);
      }
    };

    initializeVideoMode();

    // Cleanup on unmount
    return () => {
      if (localStream) {
        localStream.getTracks().forEach(track => track.stop());
      }
      if (videoElementRef.current) {
        videoElementRef.current.pause();
      }
    };
  }, [props?.enabled, createVideoFileStream]); // eslint-disable-line react-hooks/exhaustive-deps

  // Handle reinitialization when shouldReinitialize changes
  useEffect(() => {
    if (props?.shouldReinitialize) {
      reinitializeVideoSource();
    }
  }, [props?.shouldReinitialize, reinitializeVideoSource]);

  return {
    localStream,
    isInitializing,
    error,
    mode,
    videoResolution,
    switchMode,
    stopVideo,
    handleVideoFileUpload,
    reinitializeVideoSource,
  };
}
