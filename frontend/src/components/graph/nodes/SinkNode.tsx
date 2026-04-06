import { useEffect, useRef, useState, useCallback } from "react";
import { Handle, Position } from "@xyflow/react";
import type { NodeProps, Node } from "@xyflow/react";
import { Maximize2, Pause, Play, Volume2, VolumeX } from "lucide-react";
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
  }, [remoteStream]);

  useEffect(() => {
    if (videoRef.current) {
      videoRef.current.muted = isMuted;
    }
  }, [isMuted]);

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
    const v = videoRef.current;
    if (!v) return;
    if (document.fullscreenElement) {
      document.exitFullscreen();
    } else {
      v.requestFullscreen();
    }
  }, []);

  const handleY = HEADER_H + BODY_PAD + PREVIEW_H / 2;

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
          <div
            className="relative rounded-md overflow-hidden bg-black/50 flex-1 min-h-[60px]"
            onPointerDown={e => e.stopPropagation()}
          >
            {remoteStream ? (
              <>
                <video
                  ref={videoRef}
                  className={
                    hasVideoTrack
                      ? "w-full h-full object-cover"
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
                  Resize node for a bigger preview or use Spout/NDI/Syphon for external output
                </span>
              </div>
            )}
            {hasVideoTrack && (
              <div className="absolute bottom-1 right-1 flex items-center gap-0.5">
                {remoteStream && onPlayPauseToggle && (
                  <button
                    onClick={handlePlayPause}
                    onPointerDown={e => e.stopPropagation()}
                    className="flex items-center justify-center bg-black/60 px-1 rounded cursor-pointer"
                    style={{ height: 16 }}
                    title={isPlaying ? "Pause" : "Play"}
                  >
                    {isPlaying ? (
                      <Pause className="h-2.5 w-2.5 text-white" />
                    ) : (
                      <Play className="h-2.5 w-2.5 text-white" />
                    )}
                  </button>
                )}
                <button
                  onClick={handleFullscreen}
                  onPointerDown={e => e.stopPropagation()}
                  className="flex items-center justify-center bg-black/60 px-1 rounded cursor-pointer"
                  style={{ height: 16 }}
                  title="Fullscreen"
                >
                  <Maximize2 className="h-2.5 w-2.5 text-white" />
                </button>
                {videoSize && (
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
                className="absolute bottom-1 left-1 p-1 rounded bg-black/60 text-[#ccc] hover:text-white transition-colors"
                title={isMuted ? "Unmute audio" : "Mute audio"}
              >
                {isMuted ? (
                  <VolumeX className="h-3.5 w-3.5" />
                ) : (
                  <Volume2 className="h-3.5 w-3.5" />
                )}
              </button>
            )}
          </div>
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
