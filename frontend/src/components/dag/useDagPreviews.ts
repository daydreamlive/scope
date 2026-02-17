import { useEffect, useRef, useState } from "react";
import type { PreviewMap } from "./DagPreviewContext";

/**
 * Connects to the DAG previews WebSocket and returns live preview data URLs
 * keyed by node id.
 */
export function useDagPreviews(enabled: boolean): PreviewMap {
  const [previews, setPreviews] = useState<PreviewMap>({});
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (!enabled) {
      // Close existing connection when disabled
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      setPreviews({});
      return;
    }

    function connect() {
      const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
      const ws = new WebSocket(
        `${protocol}//${window.location.host}/api/v1/dag/previews`
      );
      wsRef.current = ws;

      ws.onmessage = event => {
        try {
          const data = JSON.parse(event.data);
          if (data.previews) {
            setPreviews(data.previews);
          }
        } catch {
          // ignore malformed messages
        }
      };

      ws.onclose = () => {
        wsRef.current = null;
        if (enabled) {
          // Auto-reconnect after 2 seconds
          reconnectTimer.current = setTimeout(connect, 2000);
        }
      };

      ws.onerror = () => {
        ws.close();
      };
    }

    connect();

    return () => {
      if (reconnectTimer.current) {
        clearTimeout(reconnectTimer.current);
        reconnectTimer.current = null;
      }
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [enabled]);

  return previews;
}
