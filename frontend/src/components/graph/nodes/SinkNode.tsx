import { useEffect, useRef, useState, useCallback } from "react";
import { createPortal } from "react-dom";
import { Handle, Position } from "@xyflow/react";
import type { NodeProps, Node } from "@xyflow/react";
import {
  Maximize2,
  Minimize2,
  Pause,
  Play,
  Volume2,
  VolumeX,
} from "lucide-react";
import type { FlowNodeData } from "../../../lib/graphUtils";
import { useNodeData } from "../hooks/node/useNodeData";
import { useNodeCollapse } from "../hooks/node/useNodeCollapse";
import { NodeCard, NodeHeader, collapsedHandleStyle } from "../ui";

type SinkNodeType = Node<FlowNodeData, "sink">;

const HEADER_H = 28;
const BODY_PAD = 6;
const PREVIEW_H = 120;

export function SinkNode({ id, data, selected }: NodeProps<SinkNodeType>) {
  const { updateData } = useNodeData(id);
  const { collapsed, toggleCollapse } = useNodeCollapse();
  const remoteStream = data.remoteStream as MediaStream | null | undefined;
  const sinkStats = data.sinkStats as
    | { fps: number; bitrate: number }
    | undefined;
  const isPlaying = (data.isPlaying as boolean | undefined) ?? true;
  const onPlayPauseToggle = data.onPlayPauseToggle as (() => void) | undefined;
  const videoRef = useRef<HTMLVideoElement>(null);
  const [videoSize, setVideoSize] = useState<{
    width: number;
    height: number;
  } | null>(null);

  const [isMuted, setIsMuted] = useState(true);
  const [hasAudioTrack, setHasAudioTrack] = useState(false);
  const [hasVideoTrack, setHasVideoTrack] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);

  const handleResize = useCallback(() => {
    const v = videoRef.current;
    if (v && v.videoWidth > 0 && v.videoHeight > 0) {
      setVideoSize({ width: v.videoWidth, height: v.videoHeight });
    }
  }, []);

  useEffect(() => {
    if (videoRef.current && remoteStream instanceof MediaStream) {
      videoRef.current.srcObject = remoteStream;
      setHasAudioTrack(remoteStream.getAudioTracks().length > 0);
      setHasVideoTrack(remoteStream.getVideoTracks().length > 0);

      const handleTrackAdded = () => {
        setHasAudioTrack(remoteStream.getAudioTracks().length > 0);
        setHasVideoTrack(remoteStream.getVideoTracks().length > 0);
      };
      remoteStream.addEventListener("addtrack", handleTrackAdded);
      return () => {
        remoteStream.removeEventListener("addtrack", handleTrackAdded);
      };
    }
  }, [remoteStream, isFullscreen]);

  useEffect(() => {
    if (videoRef.current) {
      videoRef.current.muted = isMuted;
    }
  }, [isMuted, isFullscreen]);

  useEffect(() => {
    if (!isFullscreen) return;
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setIsFullscreen(false);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [isFullscreen]);

  const toggleMute = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    setIsMuted(prev => !prev);
  }, []);

  const handlePlayPause = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation();
      if (onPlayPauseToggle) onPlayPauseToggle();
    },
    [onPlayPauseToggle]
  );

  const handleFullscreen = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    setIsFullscreen(prev => !prev);
  }, []);

  const handleY = HEADER_H + BODY_PAD + PREVIEW_H / 2;

  const videoContainer = (
    <div
      className={
        isFullscreen
          ? "fixed inset-0 z-[9999] bg-black"
          : "relative rounded-md overflow-hidden bg-black/50 flex-1 min-h-[60px]"
      }
      onPointerDown={e => e.stopPropagation()}
      onWheel={isFullscreen ? e => e.stopPropagation() : undefined}
    >
      {remoteStream ? (
        <>
          <video
            ref={videoRef}
            className={
              hasVideoTrack
                ? isFullscreen
                  ? "w-full h-full object-contain"
                  : "w-full h-full object-cover"
                : "absolute w-0 h-0 overflow-hidden"
            }
            autoPlay
            muted={isMuted}
            playsInline
            onResize={handleResize}
          />
          {!hasVideoTrack && (
            <div className="flex flex-col items-center justify-center h-full gap-1 text-[#8c8c8d]">
              <Volume2 className="h-5 w-5" />
              <span className="text-[10px]">Audio Only</span>
            </div>
          )}
        </>
      ) : data.isLoading ? (
        <div className="flex flex-col items-center justify-center h-full gap-1.5">
          <span
            key={data.loadingStage as string}
            className="text-[10px] font-medium animate-fade-in"
            style={{
              background:
                "linear-gradient(90deg, #8c8c8d 0%, #c0c0c0 50%, #8c8c8d 100%)",
              backgroundSize: "200% 100%",
              WebkitBackgroundClip: "text",
              WebkitTextFillColor: "transparent",
              animation: "shimmer-text 2s ease-in-out infinite",
            }}
          >
            {(data.loadingStage as string) || "Loading pipeline…"}
          </span>
          <span className="text-[8px] text-[#b0b0b0]">
            First run may take up to a minute
          </span>
          <style>{`
            @keyframes shimmer-text {
              0% { background-position: 200% 0; }
              100% { background-position: -200% 0; }
            }
          `}</style>
        </div>
      ) : (
        <div className="flex flex-col items-center justify-center h-full gap-1 text-[#8c8c8d]">
          <span className="text-[10px]">No output stream</span>
          <span className="text-[9px] text-[#666] text-center px-2">
            Resize node for a bigger preview or use Spout/NDI/Syphon for
            external output
          </span>
        </div>
      )}
      {hasVideoTrack && (
        <div
          className={
            isFullscreen
              ? "absolute bottom-3 right-3 flex items-center gap-1"
              : "absolute bottom-1 right-1 flex items-center gap-0.5"
          }
        >
          {remoteStream && onPlayPauseToggle && (
            <button
              onClick={handlePlayPause}
              onPointerDown={e => e.stopPropagation()}
              className={
                isFullscreen
                  ? "flex items-center justify-center bg-black/60 px-2 rounded cursor-pointer"
                  : "flex items-center justify-center bg-black/60 px-1 rounded cursor-pointer"
              }
              style={{ height: isFullscreen ? 28 : 16 }}
              title={isPlaying ? "Pause" : "Play"}
            >
              {isPlaying ? (
                <Pause
                  className={
                    isFullscreen
                      ? "h-4 w-4 text-white"
                      : "h-2.5 w-2.5 text-white"
                  }
                />
              ) : (
                <Play
                  className={
                    isFullscreen
                      ? "h-4 w-4 text-white"
                      : "h-2.5 w-2.5 text-white"
                  }
                />
              )}
            </button>
          )}
          <button
            onClick={handleFullscreen}
            onPointerDown={e => e.stopPropagation()}
            className={
              isFullscreen
                ? "flex items-center justify-center bg-black/60 px-2 rounded cursor-pointer"
                : "flex items-center justify-center bg-black/60 px-1 rounded cursor-pointer"
            }
            style={{ height: isFullscreen ? 28 : 16 }}
            title={isFullscreen ? "Exit fullscreen" : "Fullscreen"}
          >
            {isFullscreen ? (
              <Minimize2 className="h-4 w-4 text-white" />
            ) : (
              <Maximize2 className="h-2.5 w-2.5 text-white" />
            )}
          </button>
          {videoSize && !isFullscreen && (
            <span
              className="text-[9px] text-[#8c8c8d] bg-black/60 px-1 rounded leading-none"
              style={{
                height: 16,
                display: "flex",
                alignItems: "center",
              }}
            >
              {videoSize.width}&times;{videoSize.height}
            </span>
          )}
        </div>
      )}
      {hasAudioTrack && (
        <button
          onClick={toggleMute}
          onPointerDown={e => e.stopPropagation()}
          className={
            isFullscreen
              ? "absolute bottom-3 left-3 p-2 rounded bg-black/60 text-[#ccc] hover:text-white transition-colors"
              : "absolute bottom-1 left-1 p-1 rounded bg-black/60 text-[#ccc] hover:text-white transition-colors"
          }
          title={isMuted ? "Unmute audio" : "Mute audio"}
        >
          {isMuted ? (
            <VolumeX className={isFullscreen ? "h-5 w-5" : "h-3.5 w-3.5"} />
          ) : (
            <Volume2 className={isFullscreen ? "h-5 w-5" : "h-3.5 w-3.5"} />
          )}
        </button>
      )}
    </div>
  );

  return (
    <NodeCard selected={selected} collapsed={collapsed}>
      <NodeHeader
        title={data.customTitle || "Sink"}
        onTitleChange={newTitle => updateData({ customTitle: newTitle })}
        collapsed={collapsed}
        onCollapseToggle={toggleCollapse}
      />
      {!collapsed && (
        <div className="p-2 flex-1 min-h-0 flex flex-col">
          {isFullscreen
            ? createPortal(videoContainer, document.body)
            : videoContainer}
          {sinkStats && (sinkStats.fps > 0 || sinkStats.bitrate > 0) && (
            <div className="flex items-center gap-3 mt-1 text-[10px] text-[#8c8c8d] font-mono px-0.5">
              <span>FPS: {sinkStats.fps.toFixed(1)}</span>
              <span>
                Bitrate:{" "}
                {sinkStats.bitrate >= 1000000
                  ? `${(sinkStats.bitrate / 1000000).toFixed(1)} Mbps`
                  : `${Math.round(sinkStats.bitrate / 1000)} kbps`}
              </span>
            </div>
          )}
        </div>
      )}
      <Handle
        type="target"
        position={Position.Left}
        id="stream:video"
        className="!w-2.5 !h-2.5 !border-0"
        style={
          collapsed
            ? collapsedHandleStyle("left")
            : { top: handleY, left: 0, backgroundColor: "#ffffff" }
        }
      />
      <Handle
        type="source"
        position={Position.Right}
        id="stream:out"
        className={
          collapsed
            ? "!w-0 !h-0 !border-0 !min-w-0 !min-h-0"
            : "!w-2.5 !h-2.5 !border-0"
        }
        style={
          collapsed
            ? { ...collapsedHandleStyle("right"), opacity: 0 }
            : { top: handleY, right: 0, backgroundColor: "#ffffff" }
        }
      />
    </NodeCard>
  );
}
