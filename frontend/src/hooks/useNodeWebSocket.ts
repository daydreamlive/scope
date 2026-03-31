/**
 * WebSocket hook for real-time backend node output observation.
 *
 * Connects to `ws://.../api/v1/nodes/ws` and dispatches output state
 * updates to a callback. The WebSocket is observation-only: execution
 * does not depend on this connection.
 */

import { useEffect, useRef, useCallback, useState } from "react";

export interface NodeOutputMessage {
  type: "node_output";
  instance_id: string;
  port: string;
  value: unknown;
}

export interface NodeFullStateMessage {
  type: "full_state";
  states: Record<string, Record<string, unknown>>;
}

export type NodeWSMessage =
  | NodeOutputMessage
  | NodeFullStateMessage
  | { type: "ping" };

interface UseNodeWebSocketOptions {
  enabled?: boolean;
  onOutput?: (instanceId: string, port: string, value: unknown) => void;
  onFullState?: (states: Record<string, Record<string, unknown>>) => void;
}

export function useNodeWebSocket(options: UseNodeWebSocketOptions = {}) {
  const { enabled = true, onOutput, onFullState } = options;
  const wsRef = useRef<WebSocket | null>(null);
  const [connected, setConnected] = useState(false);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const onOutputRef = useRef(onOutput);
  onOutputRef.current = onOutput;
  const onFullStateRef = useRef(onFullState);
  onFullStateRef.current = onFullState;

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const url = `${protocol}//${window.location.host}/api/v1/nodes/ws`;

    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
    };

    ws.onmessage = event => {
      try {
        const msg: NodeWSMessage = JSON.parse(event.data);
        if (msg.type === "node_output") {
          onOutputRef.current?.(msg.instance_id, msg.port, msg.value);
        } else if (msg.type === "full_state") {
          onFullStateRef.current?.(msg.states);
        }
      } catch {
        // Ignore malformed messages
      }
    };

    ws.onclose = () => {
      setConnected(false);
      if (enabled) {
        reconnectTimerRef.current = setTimeout(connect, 3000);
      }
    };

    ws.onerror = () => ws.close();
  }, [enabled]);

  useEffect(() => {
    if (enabled) {
      connect();
    }
    return () => {
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
      wsRef.current?.close();
      wsRef.current = null;
    };
  }, [enabled, connect]);

  const sendInput = useCallback(
    async (instanceId: string, name: string, value: unknown) => {
      try {
        await fetch(`/api/v1/nodes/instances/${instanceId}/input`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name, value }),
        });
      } catch {
        // Network error — node instance may not exist yet
      }
    },
    []
  );

  const sendConfig = useCallback(
    async (instanceId: string, config: Record<string, unknown>) => {
      try {
        await fetch(`/api/v1/nodes/instances/${instanceId}/config`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ config }),
        });
      } catch {
        // Network error — node instance may not exist yet
      }
    },
    []
  );

  return { connected, sendInput, sendConfig };
}
