/**
 * WebRTC hook for fal.ai deployment
 *
 * This is a drop-in replacement for useWebRTC that routes all signaling
 * through the FalAdapter WebSocket connection instead of direct HTTP calls.
 *
 * Usage:
 *   // In your app initialization
 *   import { initFalAdapter } from "../lib/falAdapter";
 *   const adapter = initFalAdapter("wss://your-fal-endpoint/ws");
 *   await adapter.connect();
 *
 *   // In your component (same interface as useWebRTC)
 *   const { startStream, stopStream, ... } = useWebRTCFal({ adapter });
 */

import { useState, useEffect, useRef, useCallback } from "react";
import { toast } from "sonner";
import type { FalAdapter } from "../lib/falAdapter";
import type { PromptItem, PromptTransition } from "../lib/api";

interface InitialParameters {
  prompts?: string[] | PromptItem[];
  prompt_interpolation_method?: "linear" | "slerp";
  transition?: PromptTransition;
  denoising_step_list?: number[];
  noise_scale?: number;
  noise_controller?: boolean;
  manage_cache?: boolean;
  kv_cache_attention_bias?: number;
  vace_ref_images?: string[];
  vace_context_scale?: number;
  pipeline_ids?: string[];
  images?: string[];
  first_frame_image?: string;
  last_frame_image?: string;
}

interface UseWebRTCFalOptions {
  /** The FalAdapter instance to use for signaling */
  adapter: FalAdapter;
  /** Callback function called when the stream stops on the backend */
  onStreamStop?: () => void;
}

/**
 * Hook for managing WebRTC connections via fal.ai WebSocket signaling.
 *
 * This provides the same interface as useWebRTC but routes signaling
 * through the FalAdapter WebSocket connection.
 */
export function useWebRTCFal(options: UseWebRTCFalOptions) {
  const { adapter, onStreamStop } = options;

  const [remoteStream, setRemoteStream] = useState<MediaStream | null>(null);
  const [connectionState, setConnectionState] =
    useState<RTCPeerConnectionState>("new");
  const [isConnecting, setIsConnecting] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);

  const peerConnectionRef = useRef<RTCPeerConnection | null>(null);
  const dataChannelRef = useRef<RTCDataChannel | null>(null);
  const currentStreamRef = useRef<MediaStream | null>(null);
  const sessionIdRef = useRef<string | null>(null);
  const queuedCandidatesRef = useRef<RTCIceCandidate[]>([]);

  const startStream = useCallback(
    async (initialParameters?: InitialParameters, stream?: MediaStream) => {
      if (isConnecting || peerConnectionRef.current) return;

      setIsConnecting(true);

      try {
        currentStreamRef.current = stream || null;

        // Fetch ICE servers via FalAdapter
        console.log("[WebRTCFal] Fetching ICE servers via FalAdapter...");
        let config: RTCConfiguration;
        try {
          const iceServersResponse = await adapter.getIceServers();
          config = {
            iceServers: iceServersResponse.iceServers,
          };
          console.log(
            `[WebRTCFal] Using ${iceServersResponse.iceServers.length} ICE servers`
          );
        } catch (error) {
          console.warn(
            "[WebRTCFal] Failed to fetch ICE servers, using default STUN:",
            error
          );
          config = {
            iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
          };
        }

        const pc = new RTCPeerConnection(config);
        peerConnectionRef.current = pc;

        // Create data channel for parameter updates
        const dataChannel = pc.createDataChannel("parameters", {
          ordered: true,
        });
        dataChannelRef.current = dataChannel;

        dataChannel.onopen = () => {
          console.log("[WebRTCFal] Data channel opened");
        };

        dataChannel.onmessage = (event) => {
          console.log("[WebRTCFal] Data channel message:", event.data);

          try {
            const data = JSON.parse(event.data);

            // Handle stream stop notification from backend
            if (data.type === "stream_stopped") {
              console.log("[WebRTCFal] Stream stopped by backend");
              setIsStreaming(false);
              setIsConnecting(false);
              setRemoteStream(null);

              if (data.error_message) {
                toast.error("Stream Error", {
                  description: data.error_message,
                  duration: 5000,
                });
              }

              if (peerConnectionRef.current) {
                peerConnectionRef.current.close();
                peerConnectionRef.current = null;
              }

              onStreamStop?.();
            }
          } catch (error) {
            console.error("[WebRTCFal] Failed to parse data channel message:", error);
          }
        };

        dataChannel.onerror = (error) => {
          console.error("[WebRTCFal] Data channel error:", error);
        };

        // Add video track for sending to server
        let transceiver: RTCRtpTransceiver | undefined;
        if (stream) {
          stream.getTracks().forEach((track) => {
            if (track.kind === "video") {
              console.log("[WebRTCFal] Adding video track for sending");
              const sender = pc.addTrack(track, stream);
              transceiver = pc.getTransceivers().find((t) => t.sender === sender);
            }
          });
        } else {
          console.log("[WebRTCFal] No video stream - adding transceiver for no-input pipeline");
          transceiver = pc.addTransceiver("video");
        }

        // Force VP8-only for aiortc compatibility
        if (transceiver) {
          const codecs = RTCRtpReceiver.getCapabilities("video")?.codecs || [];
          const vp8Codecs = codecs.filter(
            (c) => c.mimeType.toLowerCase() === "video/vp8"
          );
          if (vp8Codecs.length > 0) {
            transceiver.setCodecPreferences(vp8Codecs);
            console.log("[WebRTCFal] Forced VP8-only codec");
          }
        }

        // Event handlers
        pc.ontrack = (evt: RTCTrackEvent) => {
          if (evt.streams && evt.streams[0]) {
            console.log("[WebRTCFal] Setting remote stream");
            setRemoteStream(evt.streams[0]);
          }
        };

        pc.onconnectionstatechange = () => {
          console.log("[WebRTCFal] Connection state:", pc.connectionState);
          setConnectionState(pc.connectionState);

          if (pc.connectionState === "connected") {
            setIsConnecting(false);
            setIsStreaming(true);
          } else if (
            pc.connectionState === "disconnected" ||
            pc.connectionState === "failed"
          ) {
            setIsConnecting(false);
            setIsStreaming(false);
          }
        };

        pc.oniceconnectionstatechange = () => {
          console.log("[WebRTCFal] ICE state:", pc.iceConnectionState);
        };

        pc.onicecandidate = async ({ candidate }: RTCPeerConnectionIceEvent) => {
          if (candidate) {
            console.log("[WebRTCFal] ICE candidate generated");

            if (sessionIdRef.current) {
              try {
                await adapter.sendIceCandidate(sessionIdRef.current, candidate);
                console.log("[WebRTCFal] Sent ICE candidate via FalAdapter");
              } catch (error) {
                console.error("[WebRTCFal] Failed to send ICE candidate:", error);
              }
            } else {
              console.log("[WebRTCFal] Queuing ICE candidate (no session ID yet)");
              queuedCandidatesRef.current.push(candidate);
            }
          } else {
            console.log("[WebRTCFal] ICE gathering complete");
          }
        };

        // Create and send offer via FalAdapter
        const offer = await pc.createOffer();
        await pc.setLocalDescription(offer);

        console.log("[WebRTCFal] Sending offer via FalAdapter");
        try {
          const answer = await adapter.sendOffer(
            pc.localDescription!.sdp,
            pc.localDescription!.type,
            initialParameters
          );

          console.log("[WebRTCFal] Received answer, sessionId:", answer.sessionId);
          sessionIdRef.current = answer.sessionId;

          // Flush queued ICE candidates
          if (queuedCandidatesRef.current.length > 0) {
            console.log(
              `[WebRTCFal] Flushing ${queuedCandidatesRef.current.length} queued candidates`
            );
            try {
              await adapter.sendIceCandidates(
                sessionIdRef.current,
                queuedCandidatesRef.current
              );
            } catch (error) {
              console.error("[WebRTCFal] Failed to send queued candidates:", error);
            }
            queuedCandidatesRef.current = [];
          }

          await pc.setRemoteDescription({
            sdp: answer.sdp,
            type: answer.type as RTCSdpType,
          });
        } catch (error) {
          console.error("[WebRTCFal] Offer/answer exchange failed:", error);
          setIsConnecting(false);
        }
      } catch (error) {
        console.error("[WebRTCFal] Failed to start stream:", error);
        setIsConnecting(false);
      }
    },
    [adapter, isConnecting, onStreamStop]
  );

  const updateVideoTrack = useCallback(
    async (newStream: MediaStream) => {
      if (peerConnectionRef.current && isStreaming) {
        try {
          const videoTrack = newStream.getVideoTracks()[0];
          if (!videoTrack) {
            console.error("[WebRTCFal] No video track in new stream");
            return false;
          }

          const sender = peerConnectionRef.current
            .getSenders()
            .find((s) => s.track?.kind === "video");

          if (sender) {
            console.log("[WebRTCFal] Replacing video track");
            await sender.replaceTrack(videoTrack);
            currentStreamRef.current = newStream;
            return true;
          } else {
            console.error("[WebRTCFal] No video sender found");
            return false;
          }
        } catch (error) {
          console.error("[WebRTCFal] Failed to replace track:", error);
          return false;
        }
      }
      return false;
    },
    [isStreaming]
  );

  const sendParameterUpdate = useCallback(
    (params: {
      prompts?: string[] | PromptItem[];
      prompt_interpolation_method?: "linear" | "slerp";
      transition?: PromptTransition;
      denoising_step_list?: number[];
      noise_scale?: number;
      noise_controller?: boolean;
      manage_cache?: boolean;
      reset_cache?: boolean;
      kv_cache_attention_bias?: number;
      paused?: boolean;
      spout_sender?: { enabled: boolean; name: string };
      spout_receiver?: { enabled: boolean; name: string };
      vace_ref_images?: string[];
      vace_use_input_video?: boolean;
      vace_context_scale?: number;
      ctrl_input?: { button: string[]; mouse: [number, number] };
      images?: string[];
      first_frame_image?: string;
      last_frame_image?: string;
    }) => {
      if (
        dataChannelRef.current &&
        dataChannelRef.current.readyState === "open"
      ) {
        try {
          const filteredParams: Record<string, unknown> = {};
          for (const [key, value] of Object.entries(params)) {
            if (value !== undefined && value !== null) {
              filteredParams[key] = value;
            }
          }

          const message = JSON.stringify(filteredParams);
          dataChannelRef.current.send(message);
          console.log("[WebRTCFal] Sent parameter update:", filteredParams);
        } catch (error) {
          console.error("[WebRTCFal] Failed to send parameter update:", error);
        }
      } else {
        console.warn("[WebRTCFal] Data channel not available");
      }
    },
    []
  );

  const stopStream = useCallback(() => {
    if (peerConnectionRef.current) {
      peerConnectionRef.current.close();
      peerConnectionRef.current = null;
    }

    dataChannelRef.current = null;
    currentStreamRef.current = null;
    sessionIdRef.current = null;
    queuedCandidatesRef.current = [];

    setRemoteStream(null);
    setConnectionState("new");
    setIsStreaming(false);
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (peerConnectionRef.current) {
        peerConnectionRef.current.close();
      }
    };
  }, []);

  return {
    remoteStream,
    connectionState,
    isConnecting,
    isStreaming,
    peerConnectionRef,
    sessionId: sessionIdRef.current,
    startStream,
    stopStream,
    updateVideoTrack,
    sendParameterUpdate,
  };
}
