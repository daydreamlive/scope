import { useState, useEffect, useRef, useCallback } from "react";
import {
  sendWebRTCOffer,
  sendIceCandidates,
  getIceServers,
  type PromptItem,
  type PromptTransition,
} from "../lib/api";
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

interface UseWebRTCOptions {
  /** Callback function called when the stream stops on the backend */
  onStreamStop?: () => void;
}

/**
 * Hook for managing WebRTC connections and streaming.
 *
 * Automatically handles stream stop notifications from the backend
 * and updates the UI state accordingly.
 */
export function useWebRTC(options?: UseWebRTCOptions) {
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

        // Fetch ICE servers from backend
        console.log("Fetching ICE servers from backend...");
        let config: RTCConfiguration;
        try {
          const iceServersResponse = await getIceServers();
          config = {
            iceServers: iceServersResponse.iceServers,
          };
          console.log(
            `Using ${iceServersResponse.iceServers.length} ICE servers from backend`
          );
        } catch (error) {
          console.warn(
            "Failed to fetch ICE servers from backend, using default STUN:",
            error
          );
          // Fallback to default STUN server
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
          console.log("Data channel opened");
        };

        dataChannel.onmessage = event => {
          console.log("Data channel message received:", event.data);

          try {
            const data = JSON.parse(event.data);

            // Handle stream stop notification from backend
            if (data.type === "stream_stopped") {
              console.log("Stream stopped by backend, updating UI");
              setIsStreaming(false);
              setIsConnecting(false);
              setRemoteStream(null);

              // Show error toast if there's an error message
              if (data.error_message) {
                toast.error("Stream Error", {
                  description: data.error_message,
                  duration: 5000,
                });
              }

              // Close the peer connection to clean up
              if (peerConnectionRef.current) {
                peerConnectionRef.current.close();
                peerConnectionRef.current = null;
              }
              // Notify parent component
              if (options?.onStreamStop) {
                options.onStreamStop();
              }
            }
          } catch (error) {
            console.error("Failed to parse data channel message:", error);
          }
        };

        dataChannel.onerror = error => {
          console.error("Data channel error:", error);
        };

        // Add video track for sending to server only if stream is provided
        let transceiver: RTCRtpTransceiver | undefined;
        if (stream) {
          stream.getTracks().forEach(track => {
            if (track.kind === "video") {
              console.log("Adding video track for sending");
              const sender = pc.addTrack(track, stream);
              transceiver = pc.getTransceivers().find(t => t.sender === sender);
            }
          });
        } else {
          console.log(
            "No video stream provided - adding video transceiver for no-input pipelines"
          );
          // For no-video-input pipelines, add a video transceiver to establish proper WebRTC connection
          transceiver = pc.addTransceiver("video");
        }

        // Force VP8-only to match aiortc's reliable codec support
        // This prevents codec mismatch issues with VP9/AV1/H264
        if (transceiver) {
          const codecs = RTCRtpReceiver.getCapabilities("video")?.codecs || [];
          const vp8Codecs = codecs.filter(
            c => c.mimeType.toLowerCase() === "video/vp8"
          );
          if (vp8Codecs.length > 0) {
            transceiver.setCodecPreferences(vp8Codecs);
            console.log("Forced VP8-only codec for aiortc compatibility");
          }
        }

        // Named event handlers
        const onTrack = (evt: RTCTrackEvent) => {
          if (evt.streams && evt.streams[0]) {
            console.log("Setting remote stream:", evt.streams[0]);
            setRemoteStream(evt.streams[0]);
          }
        };

        const onConnectionStateChange = () => {
          console.log("Connection state changed:", pc.connectionState);
          setConnectionState(pc.connectionState);

          if (pc.connectionState === "connected") {
            setIsConnecting(false);
            setIsStreaming(true);

            // Log the actual negotiated codec for verification
            const senders = pc.getSenders();
            const videoSender = senders.find(s => s.track?.kind === "video");
            if (videoSender) {
              const params = videoSender.getParameters();
              const codec = params.codecs?.[0];
              if (codec) {
                console.log(`Negotiated video codec: ${codec.mimeType}`);
              }
            }
          } else if (
            pc.connectionState === "disconnected" ||
            pc.connectionState === "failed"
          ) {
            setIsConnecting(false);
            setIsStreaming(false);
          }
        };

        const onIceConnectionStateChange = () => {
          console.log("ICE connection state changed:", pc.iceConnectionState);
        };

        const onIceCandidate = async ({
          candidate,
        }: RTCPeerConnectionIceEvent) => {
          if (candidate) {
            console.log("ICE candidate:", candidate);

            // Trickle ICE: Send candidate to server immediately
            if (sessionIdRef.current) {
              try {
                await sendIceCandidates(sessionIdRef.current, candidate);
                console.log("Sent ICE candidate to server");
              } catch (error) {
                console.error("Failed to send ICE candidate:", error);
              }
            } else {
              console.log("Session ID not available yet, queuing candidate");
              queuedCandidatesRef.current.push(candidate);
            }
          } else {
            console.log("ICE gathering complete");
          }
        };

        // Attach event handlers
        pc.ontrack = onTrack;
        pc.onconnectionstatechange = onConnectionStateChange;
        pc.oniceconnectionstatechange = onIceConnectionStateChange;
        pc.onicecandidate = onIceCandidate;

        // Create offer and start ICE gathering
        const offer = await pc.createOffer();
        await pc.setLocalDescription(offer);

        // Trickle ICE: Send offer immediately without waiting for ICE gathering
        console.log("Sending offer to server");
        try {
          const answer = await sendWebRTCOffer({
            sdp: pc.localDescription!.sdp,
            type: pc.localDescription!.type,
            initialParameters,
          });

          console.log("Received server answer:", answer);

          // Store session ID for sending candidates
          sessionIdRef.current = answer.sessionId;
          console.log("Session ID:", answer.sessionId);

          // Flush any queued ICE candidates
          if (queuedCandidatesRef.current.length > 0) {
            try {
              await sendIceCandidates(
                sessionIdRef.current,
                queuedCandidatesRef.current
              );
              console.log("Sent queued ICE candidates to server");
            } catch (error) {
              console.error("Failed to send queued ICE candidates:", error);
            }
            queuedCandidatesRef.current = [];
          }

          await pc.setRemoteDescription({
            sdp: answer.sdp,
            type: answer.type as RTCSdpType,
          });
        } catch (error) {
          console.error("Error in offer/answer exchange:", error);
          setIsConnecting(false);
        }
      } catch (error) {
        console.error("Failed to start stream:", error);
        setIsConnecting(false);
      }
    },
    [isConnecting, options]
  );

  const updateVideoTrack = useCallback(
    async (newStream: MediaStream) => {
      if (peerConnectionRef.current && isStreaming) {
        try {
          const videoTrack = newStream.getVideoTracks()[0];
          if (!videoTrack) {
            console.error("No video track found in new stream");
            return false;
          }

          const sender = peerConnectionRef.current
            .getSenders()
            .find(s => s.track?.kind === "video");

          if (sender) {
            console.log("Replacing video track");
            await sender.replaceTrack(videoTrack);
            currentStreamRef.current = newStream;
            console.log("Video track replaced successfully");
            return true;
          } else {
            console.error("No video sender found in peer connection");
            return false;
          }
        } catch (error) {
          console.error("Failed to replace video track:", error);
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
          // Filter out undefined/null parameters
          const filteredParams: Record<string, unknown> = {};
          for (const [key, value] of Object.entries(params)) {
            if (value !== undefined && value !== null) {
              filteredParams[key] = value;
            }
          }

          const message = JSON.stringify(filteredParams);
          dataChannelRef.current.send(message);
          console.log("Sent parameter update:", filteredParams);
        } catch (error) {
          console.error("Failed to send parameter update:", error);
        }
      } else {
        console.warn("Data channel not available for parameter update");
      }
    },
    []
  );

  const stopStream = useCallback(() => {
    // Close peer connection
    if (peerConnectionRef.current) {
      peerConnectionRef.current.close();
      peerConnectionRef.current = null;
    }

    // Clear data channel reference
    dataChannelRef.current = null;

    // Clear current stream reference (but don't stop it - that's handled by useLocalVideo)
    currentStreamRef.current = null;

    // Clear session ID
    sessionIdRef.current = null;

    // Clear any queued ICE candidates
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
