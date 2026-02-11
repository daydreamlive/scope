/**
 * Unified WebRTC hook that automatically uses the right implementation
 * based on whether we're in cloud mode or local mode.
 */

import { useState, useEffect, useRef, useCallback } from "react";
import { type PromptItem, type PromptTransition } from "../lib/api";
import { useApi } from "./useApi";
import { toast } from "sonner";

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

interface UseUnifiedWebRTCOptions {
  /** Callback function called when the stream stops on the backend */
  onStreamStop?: () => void;
}

/**
 * Unified WebRTC hook that works in both local and cloud modes.
 *
 * In local mode, uses direct HTTP for signaling.
 * In cloud mode, uses the CloudAdapter WebSocket for signaling.
 */
export function useUnifiedWebRTC(options?: UseUnifiedWebRTCOptions) {
  const api = useApi();

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

  // Helper to get ICE servers with STUN fallback
  const fetchIceServers = useCallback(async (): Promise<RTCConfiguration> => {
    try {
      console.log("[UnifiedWebRTC] Fetching ICE servers...");
      const iceServersResponse = await api.getIceServers();
      console.log(
        `[UnifiedWebRTC] Using ${iceServersResponse.iceServers.length} ICE servers`
      );
      return { iceServers: iceServersResponse.iceServers };
    } catch (error) {
      console.warn(
        "[UnifiedWebRTC] Failed to fetch ICE servers, using default STUN:",
        error
      );
      return { iceServers: [{ urls: "stun:stun.l.google.com:19302" }] };
    }
  }, [api]);

  const startStream = useCallback(
    async (initialParameters?: InitialParameters, stream?: MediaStream) => {
      if (isConnecting || peerConnectionRef.current) return;

      setIsConnecting(true);

      try {
        currentStreamRef.current = stream || null;

        // Fetch ICE servers
        const config = await fetchIceServers();

        const pc = new RTCPeerConnection(config);
        peerConnectionRef.current = pc;

        // Log peer connection configuration
        console.log("[UnifiedWebRTC] Created RTCPeerConnection with config:", {
          mode: api.isCloudMode ? "CLOUD (frontend direct)" : "LOCAL (backend)",
          iceServers: config.iceServers?.map((s: RTCIceServer) => ({
            urls: s.urls,
            hasCredentials: !!(s.username && s.credential),
          })),
        });

        // Create data channel for parameter updates
        const dataChannel = pc.createDataChannel("parameters", {
          ordered: true,
        });
        dataChannelRef.current = dataChannel;

        dataChannel.onopen = () => {
          console.log("[UnifiedWebRTC] Data channel opened");
        };

        dataChannel.onmessage = event => {
          console.log("[UnifiedWebRTC] Data channel message:", event.data);

          try {
            const data = JSON.parse(event.data);

            // Handle stream stop notification from backend
            if (data.type === "stream_stopped") {
              console.log("[UnifiedWebRTC] Stream stopped by backend");
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

              options?.onStreamStop?.();
            }
          } catch (error) {
            console.error(
              "[UnifiedWebRTC] Failed to parse data channel message:",
              error
            );
          }
        };

        dataChannel.onerror = error => {
          console.error("[UnifiedWebRTC] Data channel error:", error);
        };

        // Add video track for sending to server
        let transceiver: RTCRtpTransceiver | undefined;
        if (stream) {
          stream.getTracks().forEach(track => {
            if (track.kind === "video") {
              console.log("[UnifiedWebRTC] Adding video track for sending");
              const sender = pc.addTrack(track, stream);
              transceiver = pc.getTransceivers().find(t => t.sender === sender);
            }
          });
        } else {
          console.log(
            "[UnifiedWebRTC] No video stream - adding transceiver for no-input pipeline"
          );
          transceiver = pc.addTransceiver("video");
        }

        // Force VP8-only for aiortc compatibility
        if (transceiver) {
          const codecs = RTCRtpReceiver.getCapabilities("video")?.codecs || [];
          const vp8Codecs = codecs.filter(
            c => c.mimeType.toLowerCase() === "video/vp8"
          );
          if (vp8Codecs.length > 0) {
            transceiver.setCodecPreferences(vp8Codecs);
            console.log("[UnifiedWebRTC] Forced VP8-only codec");
          }
        }

        // Event handlers
        pc.ontrack = (evt: RTCTrackEvent) => {
          if (evt.streams && evt.streams[0]) {
            console.log("[UnifiedWebRTC] Setting remote stream");
            setRemoteStream(evt.streams[0]);
          }
        };

        pc.onconnectionstatechange = () => {
          console.log("[UnifiedWebRTC] Connection state:", pc.connectionState);
          setConnectionState(pc.connectionState);

          if (pc.connectionState === "connected") {
            setIsConnecting(false);
            setIsStreaming(true);

            // Log detailed connection info
            console.log(
              "[UnifiedWebRTC] ========== CONNECTION ESTABLISHED =========="
            );
            console.log(
              "[UnifiedWebRTC] Mode:",
              api.isCloudMode
                ? "CLOUD (frontend → cloud)"
                : "LOCAL (frontend → backend)"
            );
            console.log("[UnifiedWebRTC] Session ID:", sessionIdRef.current);

            // Log negotiated codec
            const senders = pc.getSenders();
            const videoSender = senders.find(s => s.track?.kind === "video");
            if (videoSender) {
              const params = videoSender.getParameters();
              const codec = params.codecs?.[0];
              if (codec) {
                console.log(
                  `[UnifiedWebRTC] Negotiated codec: ${codec.mimeType}`
                );
              }
            }

            // Log remote description info
            if (pc.remoteDescription) {
              const sdpLines = pc.remoteDescription.sdp.split("\n");
              const originLine = sdpLines.find((l: string) =>
                l.startsWith("o=")
              );
              const connectionLine = sdpLines.find((l: string) =>
                l.startsWith("c=")
              );
              console.log("[UnifiedWebRTC] Remote SDP origin:", originLine);
              console.log(
                "[UnifiedWebRTC] Remote SDP connection:",
                connectionLine
              );
            }

            // Get connection stats after a short delay
            setTimeout(async () => {
              try {
                const stats = await pc.getStats();
                stats.forEach(report => {
                  if (
                    report.type === "candidate-pair" &&
                    report.state === "succeeded"
                  ) {
                    console.log("[UnifiedWebRTC] Active candidate pair:", {
                      localCandidateId: report.localCandidateId,
                      remoteCandidateId: report.remoteCandidateId,
                      bytesSent: report.bytesSent,
                      bytesReceived: report.bytesReceived,
                    });
                  }
                  if (report.type === "remote-candidate") {
                    console.log("[UnifiedWebRTC] Remote candidate:", {
                      address: report.address,
                      port: report.port,
                      protocol: report.protocol,
                      candidateType: report.candidateType,
                    });
                  }
                });
              } catch (e) {
                console.warn("[UnifiedWebRTC] Failed to get stats:", e);
              }
            }, 1000);

            console.log(
              "[UnifiedWebRTC] =============================================="
            );
          } else if (
            pc.connectionState === "disconnected" ||
            pc.connectionState === "failed"
          ) {
            setIsConnecting(false);
            setIsStreaming(false);
          }
        };

        pc.oniceconnectionstatechange = () => {
          console.log("[UnifiedWebRTC] ICE state:", pc.iceConnectionState);
        };

        pc.onicecandidate = async ({
          candidate,
        }: RTCPeerConnectionIceEvent) => {
          if (candidate) {
            console.log("[UnifiedWebRTC] ICE candidate generated");

            if (sessionIdRef.current) {
              try {
                await api.sendIceCandidates(sessionIdRef.current, candidate);
                console.log("[UnifiedWebRTC] Sent ICE candidate");
              } catch (error) {
                console.error(
                  "[UnifiedWebRTC] Failed to send ICE candidate:",
                  error
                );
              }
            } else {
              console.log(
                "[UnifiedWebRTC] Queuing ICE candidate (no session ID yet)"
              );
              queuedCandidatesRef.current.push(candidate);
            }
          } else {
            console.log("[UnifiedWebRTC] ICE gathering complete");
          }
        };

        // Create and send offer
        const offer = await pc.createOffer();
        await pc.setLocalDescription(offer);

        console.log("[UnifiedWebRTC] Sending offer");
        try {
          const answer = await api.sendWebRTCOffer({
            sdp: pc.localDescription!.sdp,
            type: pc.localDescription!.type,
            initialParameters,
          });

          console.log(
            "[UnifiedWebRTC] Received answer, sessionId:",
            answer.sessionId
          );
          sessionIdRef.current = answer.sessionId;

          // Parse the SDP to show where we're connecting
          const sdpLines = answer.sdp.split("\n");
          const originLine = sdpLines.find((l: string) => l.startsWith("o="));
          const sessionName = sdpLines.find((l: string) => l.startsWith("s="));
          console.log("[UnifiedWebRTC] Server SDP origin:", originLine);
          console.log("[UnifiedWebRTC] Server SDP session:", sessionName);
          console.log(
            "[UnifiedWebRTC] Stream target:",
            api.isCloudMode ? "cloud backend" : "local backend"
          );

          // Flush queued ICE candidates
          if (queuedCandidatesRef.current.length > 0) {
            console.log(
              `[UnifiedWebRTC] Flushing ${queuedCandidatesRef.current.length} queued candidates`
            );
            for (const candidate of queuedCandidatesRef.current) {
              try {
                await api.sendIceCandidates(sessionIdRef.current, candidate);
              } catch (error) {
                console.error(
                  "[UnifiedWebRTC] Failed to send queued candidate:",
                  error
                );
              }
            }
            queuedCandidatesRef.current = [];
          }

          await pc.setRemoteDescription({
            sdp: answer.sdp,
            type: answer.type as RTCSdpType,
          });
        } catch (error) {
          console.error("[UnifiedWebRTC] Offer/answer exchange failed:", error);
          setIsConnecting(false);
        }
      } catch (error) {
        console.error("[UnifiedWebRTC] Failed to start stream:", error);
        setIsConnecting(false);
      }
    },
    [isConnecting, options, fetchIceServers, api]
  );

  const updateVideoTrack = useCallback(
    async (newStream: MediaStream) => {
      if (peerConnectionRef.current && isStreaming) {
        try {
          const videoTrack = newStream.getVideoTracks()[0];
          if (!videoTrack) {
            console.error("[UnifiedWebRTC] No video track in new stream");
            return false;
          }

          const sender = peerConnectionRef.current
            .getSenders()
            .find(s => s.track?.kind === "video");

          if (sender) {
            console.log("[UnifiedWebRTC] Replacing video track");
            await sender.replaceTrack(videoTrack);
            currentStreamRef.current = newStream;
            return true;
          } else {
            console.error("[UnifiedWebRTC] No video sender found");
            return false;
          }
        } catch (error) {
          console.error("[UnifiedWebRTC] Failed to replace track:", error);
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
          console.log("[UnifiedWebRTC] Sent parameter update:", filteredParams);
        } catch (error) {
          console.error(
            "[UnifiedWebRTC] Failed to send parameter update:",
            error
          );
        }
      } else {
        console.warn("[UnifiedWebRTC] Data channel not available");
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
