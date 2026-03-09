import React, {
  useRef,
  useEffect,
  useState,
  useMemo,
  useCallback,
} from "react";
import { useVirtualizer } from "@tanstack/react-virtual";
import { Terminal, Copy, Trash2, X, Check } from "lucide-react";
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

const LogRow = React.memo(function LogRow({ log }: { log: LogLine }) {
  return (
    <div
      className={`${levelColor[log.level]} whitespace-pre-wrap leading-5${log.isCloud ? " border-l-2 border-blue-500/50 bg-blue-950/20 pl-2" : ""}`}
    >
      {log.isCloud && (
        <span className="bg-blue-900/40 text-blue-300 text-[10px] px-1 rounded mr-1 inline-block">
          CLOUD
        </span>
      )}
      {log.text}
    </div>
  );
});

export function LogPanel({ logs, isOpen, onClose, onClear }: LogPanelProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const autoScrollRef = useRef(true);
  const [filter, setFilter] = useState<LogFilter>("all");
  const [copied, setCopied] = useState(false);

  const filteredLogs = useMemo(() => {
    if (filter === "errors")
      return logs.filter(l => l.level === "ERROR" || l.level === "WARNING");
    if (filter === "cloud") return logs.filter(l => l.isCloud);
    return logs;
  }, [logs, filter]);

  const virtualizer = useVirtualizer({
    count: filteredLogs.length,
    getScrollElement: () => scrollRef.current,
    estimateSize: () => 20,
    overscan: 20,
  });

  useEffect(() => {
    if (isOpen && autoScrollRef.current && filteredLogs.length > 0) {
      virtualizer.scrollToIndex(filteredLogs.length - 1, { align: "end" });
    }
  }, [filteredLogs.length, isOpen, virtualizer]);

  const handleScroll = useCallback((e: React.UIEvent<HTMLDivElement>) => {
    const el = e.currentTarget;
    const isAtBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 50;
    autoScrollRef.current = isAtBottom;
  }, []);

  const handleCopy = useCallback(async () => {
    const text = filteredLogs.map(l => l.text).join("\n");
    await navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }, [filteredLogs]);

  if (!isOpen) return null;

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
          <div
            style={{
              height: virtualizer.getTotalSize(),
              width: "100%",
              position: "relative",
            }}
          >
            {virtualizer.getVirtualItems().map(virtualRow => {
              const log = filteredLogs[virtualRow.index];
              return (
                <div
                  key={log.id}
                  ref={virtualizer.measureElement}
                  data-index={virtualRow.index}
                  style={{
                    position: "absolute",
                    top: 0,
                    left: 0,
                    width: "100%",
                    transform: `translateY(${virtualRow.start}px)`,
                  }}
                >
                  <LogRow log={log} />
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
