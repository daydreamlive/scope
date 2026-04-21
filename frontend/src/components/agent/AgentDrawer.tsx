import { useCallback, useEffect, useRef, useState } from "react";
import { X, Trash2, StopCircle } from "lucide-react";
import { useAgent } from "@/contexts/AgentContext";
import { Button } from "@/components/ui/button";
import { ChatTranscript } from "./ChatTranscript";
import { Composer } from "./Composer";

const DRAWER_WIDTH_KEY = "scope:agent:drawer:width";
const MIN_WIDTH = 320;
const MAX_WIDTH = 900;
const DEFAULT_WIDTH = 440;

export function AgentDrawer() {
  const {
    drawerOpen,
    setDrawerOpen,
    messages,
    isStreaming,
    config,
    configError,
    sendMessage,
    abort,
    resetSession,
    decideProposal,
    pendingProposal,
  } = useAgent();

  const [width, setWidth] = useState<number>(() => {
    const stored = localStorage.getItem(DRAWER_WIDTH_KEY);
    const parsed = stored ? parseInt(stored, 10) : NaN;
    return Number.isFinite(parsed)
      ? Math.min(MAX_WIDTH, Math.max(MIN_WIDTH, parsed))
      : DEFAULT_WIDTH;
  });
  const draggingRef = useRef(false);

  useEffect(() => {
    localStorage.setItem(DRAWER_WIDTH_KEY, String(width));
  }, [width]);

  const onDragStart = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault();
      draggingRef.current = true;
      const startX = e.clientX;
      const startWidth = width;
      const onMove = (me: MouseEvent) => {
        if (!draggingRef.current) return;
        const delta = startX - me.clientX;
        const next = Math.min(
          MAX_WIDTH,
          Math.max(MIN_WIDTH, startWidth + delta)
        );
        setWidth(next);
      };
      const onUp = () => {
        draggingRef.current = false;
        window.removeEventListener("mousemove", onMove);
        window.removeEventListener("mouseup", onUp);
      };
      window.addEventListener("mousemove", onMove);
      window.addEventListener("mouseup", onUp);
    },
    [width]
  );

  if (!drawerOpen) return null;

  const needsKey =
    !!config &&
    !configError &&
    config.key_sources[config.provider] == null &&
    config.provider !== "self_hosted";

  return (
    <div
      className="relative h-full shrink-0 bg-[#0f0f0f] border-l border-[rgba(255,255,255,0.08)] flex flex-col"
      style={{ width }}
      role="complementary"
      aria-label="Scope Agent"
    >
      {/* Resize handle — dragging grows the drawer toward the graph (the
          delta is inverted because the handle sits on the LEFT edge of the
          drawer but we track the mouse moving LEFTWARD as "wider drawer"). */}
      <div
        role="separator"
        aria-orientation="vertical"
        onMouseDown={onDragStart}
        className="absolute top-0 left-0 bottom-0 w-1 -ml-0.5 cursor-col-resize hover:bg-[rgba(255,255,255,0.12)]"
        title="Drag to resize"
      />

      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-[rgba(255,255,255,0.08)]">
        <div className="flex items-center gap-2">
          <div className="text-sm font-medium text-[#fafafa]">Scope Agent</div>
          {config && (
            <span className="text-[10px] uppercase tracking-wide text-[#8c8c8d] rounded px-1.5 py-0.5 border border-[rgba(255,255,255,0.08)]">
              {config.provider === "anthropic"
                ? `Claude • ${config.model}`
                : config.provider === "openai_compatible"
                  ? `OpenAI • ${config.model}`
                  : `Local • ${config.model}`}
            </span>
          )}
        </div>
        <div className="flex items-center gap-1">
          {isStreaming && (
            <Button
              variant="ghost"
              size="sm"
              onClick={abort}
              title="Stop current response"
            >
              <StopCircle className="h-4 w-4" />
            </Button>
          )}
          <Button
            variant="ghost"
            size="sm"
            onClick={() => {
              if (confirm("Clear this agent conversation?")) resetSession();
            }}
            title="New conversation"
            disabled={messages.length === 0}
          >
            <Trash2 className="h-4 w-4" />
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setDrawerOpen(false)}
            title="Close"
          >
            <X className="h-4 w-4" />
          </Button>
        </div>
      </div>

      {/* Banner: missing key / config error */}
      {configError && (
        <div className="px-4 py-2 text-xs bg-red-900/30 text-red-300 border-b border-red-900/40">
          Failed to load agent config: {configError}
        </div>
      )}
      {needsKey && (
        <div className="px-4 py-2 text-xs bg-amber-900/20 text-amber-300 border-b border-amber-900/30">
          No API key configured for{" "}
          {config?.provider === "anthropic" ? "Anthropic" : "OpenAI-compatible"}
          . Open Settings → API Keys to add one.
        </div>
      )}

      {/* Transcript */}
      <ChatTranscript
        messages={messages}
        pendingProposal={pendingProposal}
        onDecide={decideProposal}
      />

      {/* Composer */}
      <Composer
        onSend={sendMessage}
        disabled={isStreaming || (needsKey && messages.length === 0)}
        placeholder={
          needsKey && messages.length === 0
            ? "Add an API key in Settings to start."
            : isStreaming
              ? "Agent is working…"
              : "Ask the agent…"
        }
      />
    </div>
  );
}
