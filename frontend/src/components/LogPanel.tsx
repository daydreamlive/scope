import { useRef, useEffect, useState, useCallback } from "react";
import { Terminal, Copy, Trash2, X, Check, Bug } from "lucide-react";
import { Button } from "./ui/button";
import type { LogLine, LogLevel } from "../hooks/useLogStream";

type LogFilter = "all" | "errors" | "cloud";

interface LogPanelProps {
  logs: LogLine[];
  isOpen: boolean;
  onClose: () => void;
  onClear: () => void;
}

const levelColor: Record<LogLevel, string> = {
  ERROR: "text-red-400",
  WARNING: "text-amber-400",
  INFO: "text-blue-300",
  DEBUG: "text-neutral-500",
  UNKNOWN: "text-neutral-400",
};

export function LogPanel({ logs, isOpen, onClose, onClear }: LogPanelProps) {
  const bottomRef = useRef<HTMLDivElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const [autoScroll, setAutoScroll] = useState(true);
  const [filter, setFilter] = useState<LogFilter>("all");
  const [copied, setCopied] = useState(false);
  const [debugEnabled, setDebugEnabled] = useState(false);
  const [debugLoading, setDebugLoading] = useState(false);

  // Fetch initial debug state on mount
  useEffect(() => {
    fetch("/api/v1/logs/debug")
      .then(r => r.json())
      .then(data => setDebugEnabled(data.enabled))
      .catch(() => {});
  }, []);

  const handleDebugToggle = useCallback(async () => {
    setDebugLoading(true);
    try {
      const resp = await fetch("/api/v1/logs/debug", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enabled: !debugEnabled }),
      });
      if (resp.ok) {
        const data = await resp.json();
        setDebugEnabled(data.enabled);
      }
    } catch {
      // Silently ignore
    } finally {
      setDebugLoading(false);
    }
  }, [debugEnabled]);

  // Auto-scroll to bottom when new logs arrive or panel opens
  useEffect(() => {
    if (isOpen && autoScroll && bottomRef.current) {
      bottomRef.current.scrollIntoView({ block: "end" });
    }
  }, [logs, filter, autoScroll, isOpen]);

  if (!isOpen) return null;

  const filteredLogs = logs.filter(log => {
    if (filter === "errors")
      return log.level === "ERROR" || log.level === "WARNING";
    if (filter === "cloud") return log.isCloud;
    return true;
  });

  const handleCopy = async () => {
    const text = filteredLogs.map(l => l.text).join("\n");
    await navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleScroll = (e: React.UIEvent<HTMLDivElement>) => {
    const el = e.currentTarget;
    const isAtBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 50;
    setAutoScroll(isAtBottom);
  };

  return (
    <div
      className="border-t bg-background flex flex-col flex-shrink-0"
      style={{ height: 250 }}
    >
      {/* Toolbar */}
      <div className="flex items-center gap-2 px-3 py-1 border-b bg-muted/30 flex-shrink-0">
        <Terminal className="h-3.5 w-3.5 text-muted-foreground" />
        <span className="text-xs font-medium">Logs</span>

        <div className="flex-1" />

        {/* Filter buttons */}
        {(["all", "errors", "cloud"] as const).map(f => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            className={`text-[11px] px-1.5 py-0.5 rounded transition-colors ${
              filter === f
                ? "bg-accent text-accent-foreground"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            {f === "all" ? "All" : f === "errors" ? "Errors" : "Cloud"}
          </button>
        ))}

        <div className="w-px h-4 bg-border mx-1" />

        <button
          onClick={handleDebugToggle}
          disabled={debugLoading}
          className={`text-[11px] px-1.5 py-0.5 rounded transition-colors flex items-center gap-1 ${
            debugEnabled
              ? "bg-amber-900/50 text-amber-300"
              : "text-muted-foreground hover:text-foreground"
          }`}
          title={
            debugEnabled ? "Disable debug logging" : "Enable debug logging"
          }
        >
          <Bug className="h-3 w-3" />
          {debugEnabled ? "Debug ON" : "Debug OFF"}
        </button>
        <Button
          variant="ghost"
          size="icon"
          className="h-6 w-6"
          onClick={handleCopy}
          title="Copy logs"
        >
          {copied ? (
            <Check className="h-3 w-3" />
          ) : (
            <Copy className="h-3 w-3" />
          )}
        </Button>
        <Button
          variant="ghost"
          size="icon"
          className="h-6 w-6"
          onClick={onClear}
          title="Clear logs"
        >
          <Trash2 className="h-3 w-3" />
        </Button>
        <Button
          variant="ghost"
          size="icon"
          className="h-6 w-6"
          onClick={onClose}
          title="Close log panel"
        >
          <X className="h-3 w-3" />
        </Button>
      </div>

      {/* Log content */}
      <div
        ref={scrollRef}
        className="flex-1 overflow-auto font-mono text-xs p-2"
        onScroll={handleScroll}
      >
        {filteredLogs.length === 0 ? (
          <div className="text-muted-foreground text-center py-8">
            No logs yet
          </div>
        ) : (
          filteredLogs.map((log, i) => (
            <div
              key={i}
              className={`${levelColor[log.level]} whitespace-pre-wrap leading-5${log.isCloud ? " border-l-2 border-blue-500/50 bg-blue-950/20 pl-2" : ""}`}
            >
              {log.isCloud && (
                <span className="bg-blue-900/40 text-blue-300 text-[10px] px-1 rounded mr-1 inline-block">
                  CLOUD
                </span>
              )}
              {log.text}
            </div>
          ))
        )}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
