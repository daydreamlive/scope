import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { toast } from "sonner";
import {
  decideAgentProposal,
  getAgentConfig,
  type AgentConfigResponse,
  type GraphConfig,
} from "@/lib/api";
import { streamAgentChat, type AgentStreamEvent } from "@/lib/agentClient";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface AgentToolCall {
  id: string;
  name: string;
  input?: Record<string, unknown>;
  status: "running" | "done" | "error";
  summary?: string;
  ok?: boolean;
}

export interface AgentMessage {
  id: string;
  role: "user" | "assistant" | "system";
  text: string;
  toolCalls: AgentToolCall[];
  isContinuation?: boolean;
  pending?: boolean;
  createdAt: number;
}

export interface AgentProposal {
  proposalId: string;
  graph: GraphConfig;
  graphHash: string;
  rationale: string;
  pipelinesToLoad: string[];
  diff: Record<string, unknown>;
  decision?: "approved" | "rejected";
}

export type GraphImporter = (graph: GraphConfig, label?: string) => void;

interface AgentContextValue {
  drawerOpen: boolean;
  setDrawerOpen: (open: boolean) => void;
  toggleDrawer: () => void;

  messages: AgentMessage[];
  isStreaming: boolean;
  sessionId: string | null;
  pendingProposal: AgentProposal | null;
  config: AgentConfigResponse | null;
  configError: string | null;

  sendMessage: (text: string) => Promise<void>;
  abort: () => void;
  resetSession: () => void;
  decideProposal: (approved: boolean, reason?: string) => Promise<void>;
  refreshConfig: () => Promise<void>;
  // Registered once by StreamPage so the agent can write approved proposals
  // into the React Flow canvas. Returns an unregister fn.
  registerGraphImporter: (importer: GraphImporter) => () => void;
}

const AgentContext = createContext<AgentContextValue | null>(null);

function makeId(prefix: string): string {
  return `${prefix}_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;
}

// ---------------------------------------------------------------------------
// Provider
// ---------------------------------------------------------------------------

export function AgentProvider({ children }: { children: ReactNode }) {
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [messages, setMessages] = useState<AgentMessage[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [pendingProposal, setPendingProposal] = useState<AgentProposal | null>(
    null
  );
  const [config, setConfig] = useState<AgentConfigResponse | null>(null);
  const [configError, setConfigError] = useState<string | null>(null);

  const abortRef = useRef<AbortController | null>(null);
  const sessionIdRef = useRef<string | null>(null);
  sessionIdRef.current = sessionId;
  const graphImporterRef = useRef<GraphImporter | null>(null);

  const registerGraphImporter = useCallback((importer: GraphImporter) => {
    graphImporterRef.current = importer;
    return () => {
      if (graphImporterRef.current === importer) {
        graphImporterRef.current = null;
      }
    };
  }, []);

  const refreshConfig = useCallback(async () => {
    try {
      const cfg = await getAgentConfig();
      setConfig(cfg);
      setConfigError(null);
    } catch (e) {
      setConfigError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    void refreshConfig();
  }, [refreshConfig]);

  const abort = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    setIsStreaming(false);
  }, []);

  const resetSession = useCallback(() => {
    abort();
    setMessages([]);
    setPendingProposal(null);
    setSessionId(null);
  }, [abort]);

  // Core streaming pipeline. Shared between sendMessage and decideProposal.
  const runStream = useCallback(
    async (
      text: string,
      { isContinuation = false }: { isContinuation?: boolean } = {}
    ) => {
      const controller = new AbortController();
      abortRef.current?.abort();
      abortRef.current = controller;

      const userMsgId = makeId("m");
      const assistantMsgId = makeId("m");

      // Append user message (hide continuations from the visible list — they're
      // synthetic, not authored by the user).
      if (!isContinuation) {
        setMessages(prev => [
          ...prev,
          {
            id: userMsgId,
            role: "user",
            text,
            toolCalls: [],
            createdAt: Date.now(),
          },
        ]);
      }

      // Append streaming assistant placeholder.
      setMessages(prev => [
        ...prev,
        {
          id: assistantMsgId,
          role: "assistant",
          text: "",
          toolCalls: [],
          pending: true,
          createdAt: Date.now(),
        },
      ]);
      setIsStreaming(true);

      const updateAssistant = (mutator: (m: AgentMessage) => AgentMessage) => {
        setMessages(prev =>
          prev.map(m => (m.id === assistantMsgId ? mutator(m) : m))
        );
      };

      const handleEvent = (ev: AgentStreamEvent) => {
        const data = ev.data as Record<string, unknown>;
        switch (ev.event) {
          case "text_delta":
            updateAssistant(m => ({
              ...m,
              text: m.text + String(data.delta ?? ""),
            }));
            break;
          case "tool_call_start":
            updateAssistant(m => ({
              ...m,
              toolCalls: [
                ...m.toolCalls,
                {
                  id: String(data.id ?? ""),
                  name: String(data.name ?? ""),
                  status: "running",
                },
              ],
            }));
            break;
          case "tool_call_input":
            updateAssistant(m => ({
              ...m,
              toolCalls: m.toolCalls.map(tc =>
                tc.id === String(data.id)
                  ? {
                      ...tc,
                      input: data.input as Record<string, unknown> | undefined,
                    }
                  : tc
              ),
            }));
            break;
          case "tool_call_result":
            updateAssistant(m => ({
              ...m,
              toolCalls: m.toolCalls.map(tc =>
                tc.id === String(data.id)
                  ? {
                      ...tc,
                      status: data.ok ? "done" : "error",
                      ok: Boolean(data.ok),
                      summary: String(data.summary ?? ""),
                    }
                  : tc
              ),
            }));
            break;
          case "workflow_proposal":
            setPendingProposal({
              proposalId: String(data.proposal_id ?? ""),
              graph: data.graph as GraphConfig,
              graphHash: String(data.graph_hash ?? ""),
              rationale: String(data.rationale ?? ""),
              pipelinesToLoad: (data.pipelines_to_load as string[]) ?? [],
              diff: (data.diff as Record<string, unknown>) ?? {},
            });
            break;
          case "error":
            toast.error(String(data.message ?? "Agent error"));
            updateAssistant(m => ({
              ...m,
              text:
                m.text + `\n\n[Error: ${String(data.message ?? "unknown")}]`,
            }));
            break;
          case "turn_end":
            updateAssistant(m => ({ ...m, pending: false }));
            break;
          default:
            break;
        }
      };

      try {
        const returnedSessionId = await streamAgentChat(text, {
          sessionId: sessionIdRef.current,
          isContinuation,
          signal: controller.signal,
          onEvent: handleEvent,
        });
        if (returnedSessionId && !sessionIdRef.current) {
          setSessionId(returnedSessionId);
        }
      } catch (e) {
        if ((e as Error).name === "AbortError") {
          updateAssistant(m => ({
            ...m,
            pending: false,
            text: m.text + "\n[stopped]",
          }));
        } else {
          toast.error(e instanceof Error ? e.message : String(e));
          updateAssistant(m => ({
            ...m,
            pending: false,
            text: `${m.text}\n\n[${e instanceof Error ? e.message : "stream failed"}]`,
          }));
        }
      } finally {
        setIsStreaming(false);
        if (abortRef.current === controller) {
          abortRef.current = null;
        }
      }
    },
    []
  );

  const sendMessage = useCallback(
    async (text: string) => {
      if (!text.trim()) return;
      await runStream(text.trim(), { isContinuation: false });
    },
    [runStream]
  );

  const decideProposal = useCallback(
    async (approved: boolean, reason?: string) => {
      const proposal = pendingProposal;
      const sid = sessionIdRef.current;
      if (!proposal || !sid) return;

      // On approval, write the proposed graph into the React Flow canvas BEFORE
      // we tell the backend. The backend's apply_workflow tool no longer starts
      // a session — the user presses Play. This also means an approval with no
      // importer registered still succeeds at the API layer (graceful fallback
      // for any surface that doesn't render the canvas).
      if (approved) {
        const importer = graphImporterRef.current;
        if (importer) {
          try {
            importer(proposal.graph, `agent-proposal-${proposal.proposalId}`);
          } catch (e) {
            toast.error(
              `Failed to apply proposal to canvas: ${e instanceof Error ? e.message : String(e)}`
            );
            return;
          }
        } else {
          toast.warning("Graph canvas not ready; proposal not applied.");
          return;
        }
      }

      try {
        const response = await decideAgentProposal({
          session_id: sid,
          proposal_id: proposal.proposalId,
          approved,
          reason,
        });
        setPendingProposal(prev =>
          prev && prev.proposalId === proposal.proposalId
            ? { ...prev, decision: approved ? "approved" : "rejected" }
            : prev
        );
        if (approved) {
          toast.success("Proposal applied to graph. Press Play to start.");
        }
        await runStream(response.next_message, { isContinuation: true });
        // Clear after the continuation turn finishes (or immediately on reject;
        // we clear here regardless so the card disappears from the transcript).
        setPendingProposal(null);
      } catch (e) {
        toast.error(e instanceof Error ? e.message : String(e));
      }
    },
    [pendingProposal, runStream]
  );

  const toggleDrawer = useCallback(() => setDrawerOpen(o => !o), []);

  const value = useMemo<AgentContextValue>(
    () => ({
      drawerOpen,
      setDrawerOpen,
      toggleDrawer,
      messages,
      isStreaming,
      sessionId,
      pendingProposal,
      config,
      configError,
      sendMessage,
      abort,
      resetSession,
      decideProposal,
      refreshConfig,
      registerGraphImporter,
    }),
    [
      drawerOpen,
      toggleDrawer,
      messages,
      isStreaming,
      sessionId,
      pendingProposal,
      config,
      configError,
      sendMessage,
      abort,
      resetSession,
      decideProposal,
      refreshConfig,
      registerGraphImporter,
    ]
  );

  return (
    <AgentContext.Provider value={value}>{children}</AgentContext.Provider>
  );
}

export function useAgent(): AgentContextValue {
  const ctx = useContext(AgentContext);
  if (!ctx) throw new Error("useAgent must be used within AgentProvider");
  return ctx;
}
