import { useState, useEffect, useRef, useCallback } from "react";
import { useCloudContext } from "../lib/cloudContext";

const MAX_LOG_LINES = 2000;
const LOCAL_POLL_INTERVAL_MS = 2000;

export type LogLevel = "ERROR" | "WARNING" | "INFO" | "DEBUG" | "UNKNOWN";

export interface LogLine {
  text: string;
  level: LogLevel;
  isCloud: boolean;
}

function parseLogLine(raw: string): LogLine {
  const isCloud = raw.includes("[CLOUD]");
  let level: LogLevel = "UNKNOWN";
  if (raw.includes(" - ERROR - ")) level = "ERROR";
  else if (raw.includes(" - WARNING - ")) level = "WARNING";
  else if (raw.includes(" - INFO - ")) level = "INFO";
  else if (raw.includes(" - DEBUG - ")) level = "DEBUG";

  return { text: raw, level, isCloud };
}

export function useLogStream() {
  const [logs, setLogs] = useState<LogLine[]>([]);
  const [isOpen, setIsOpen] = useState(false);
  const [unreadCount, setUnreadCount] = useState(0);
  const { adapter, isCloudMode } = useCloudContext();
  const offsetRef = useRef(0);
  const isOpenRef = useRef(false);

  // Keep ref in sync so callbacks don't trigger re-renders
  useEffect(() => {
    isOpenRef.current = isOpen;
  }, [isOpen]);

  const addLines = useCallback((rawLines: string[]) => {
    const parsed = rawLines.map(parseLogLine);
    setLogs(prev => {
      const combined = [...prev, ...parsed];
      return combined.length > MAX_LOG_LINES
        ? combined.slice(-MAX_LOG_LINES)
        : combined;
    });
    if (!isOpenRef.current) {
      setUnreadCount(prev => prev + rawLines.length);
    }
  }, []);

  // Direct cloud mode: receive logs via WebSocket push
  useEffect(() => {
    if (!isCloudMode || !adapter) return;
    return adapter.onLogs(addLines);
  }, [adapter, isCloudMode, addLines]);

  // Local / relay mode: poll /api/v1/logs/tail
  useEffect(() => {
    if (isCloudMode) return;

    const poll = async () => {
      try {
        const resp = await fetch(
          `/api/v1/logs/tail?lines=200&since_offset=${offsetRef.current}`
        );
        if (resp.ok) {
          const data = await resp.json();
          if (data.lines.length > 0) {
            addLines(data.lines);
          }
          offsetRef.current = data.offset;
        }
      } catch {
        // Silently ignore polling errors
      }
    };

    poll();
    const interval = setInterval(poll, LOCAL_POLL_INTERVAL_MS);
    return () => clearInterval(interval);
  }, [isCloudMode, addLines]);

  const clearLogs = useCallback(() => {
    setLogs([]);
    setUnreadCount(0);
  }, []);

  const toggle = useCallback(() => {
    setIsOpen(prev => !prev);
    setUnreadCount(0);
  }, []);

  return { logs, isOpen, toggle, clearLogs, unreadCount };
}
