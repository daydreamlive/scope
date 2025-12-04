// from http://github.com/livepeer/pipelines/blob/main/apps/streamdiffusion/src/components/WhipConnection.tsx
"use client";

import { usePlaybackUrl } from "@/hooks/usePlaybackUrl";
import { memo, useEffect, useRef } from "react";
import { WhipClient } from "@/lib/WhipClient";

interface WhipConnectionProps {
  whipUrl: string;
  stream: MediaStream | null;
  onConnectionStateChange?: (state: RTCPeerConnectionState) => void;
  onRetryLimitExceeded?: () => void;
  iceServers?: RTCIceServer[];
  sdpTimeout?: number;
  enableRetry?: boolean;
  accessKey?: string;
  jwt?: string;
  debugStats?: boolean;
}

export const WhipConnection = memo(
  ({
    whipUrl,
    stream,
    onConnectionStateChange,
    onRetryLimitExceeded,
    iceServers,
    sdpTimeout = 10000,
    enableRetry = true,
    accessKey,
    jwt,
    debugStats,
  }: WhipConnectionProps) => {
    const { setPlaybackUrl, setLoading } = usePlaybackUrl();
    const clientRef = useRef<WhipClient | null>(null);
    const streamRef = useRef<MediaStream | null>(stream);

    useEffect(() => {
      streamRef.current = stream || null;
      const client = clientRef.current;
      if (client && stream) {
        void client.setStream(stream);
        void client.connect();
      }
    }, [stream]);

    useEffect(() => {
      console.log("[WhipConnection] Re-rendering WhipConnection");
      if (!whipUrl) return;

      const client = new WhipClient({
        whipUrl,
        getStream: () => streamRef.current,
        onConnectionStateChange,
        onRetryLimitExceeded,
        onPlaybackUrl: (url: string) => {
          setPlaybackUrl(url);
          setLoading(false);
        },
        iceServers,
        sdpTimeout,
        enableRetry,
        accessKey,
        jwt,
        debugStats,
        initialMaxFramerate: 30,
      });
      clientRef.current = client;
      console.log("[WhipConnection] WhipClient created");
      void client.connect();
      console.log("[WhipConnection] WhipClient connected");
      return () => {
        console.log("[WhipConnection] WhipClient stopped");
        void client.stop();
        clientRef.current = null;
      };
    }, [
      whipUrl,
      onConnectionStateChange,
      onRetryLimitExceeded,
      iceServers,
      sdpTimeout,
      enableRetry,
      accessKey,
      jwt,
      debugStats,
      setPlaybackUrl,
      setLoading,
    ]);

    return null;
  },
);

WhipConnection.displayName = "WhipConnection";
