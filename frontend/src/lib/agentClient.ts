/**
 * Streaming client for POST /api/v1/agent/chat (SSE over POST).
 *
 * We use fetch + ReadableStream rather than EventSource because EventSource
 * only supports GET and we want to POST the prompt body. The server emits
 * standard SSE frames (`event: ...\ndata: ...\n\n`).
 */

export interface AgentStreamEvent {
  event: string;
  data: Record<string, unknown>;
}

export interface AgentStreamOptions {
  sessionId?: string | null;
  isContinuation?: boolean;
  signal?: AbortSignal;
  onEvent: (event: AgentStreamEvent) => void;
}

/**
 * Open an SSE-over-POST stream to the agent chat endpoint and dispatch events.
 * Resolves with the session id once the stream closes.
 */
export async function streamAgentChat(
  message: string,
  opts: AgentStreamOptions
): Promise<string | null> {
  const response = await fetch("/api/v1/agent/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message,
      session_id: opts.sessionId ?? null,
      is_continuation: opts.isContinuation ?? false,
    }),
    signal: opts.signal,
  });

  if (!response.ok) {
    const body = await response.text().catch(() => "");
    throw new Error(
      `Agent chat failed (${response.status}): ${body.slice(0, 300)}`
    );
  }

  const sessionId = response.headers.get("X-Agent-Session-Id");

  if (!response.body) {
    throw new Error("Agent chat stream returned no body");
  }

  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();

  let buffer = "";
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += value;

      let delimIdx = buffer.indexOf("\n\n");
      while (delimIdx !== -1) {
        const frame = buffer.slice(0, delimIdx);
        buffer = buffer.slice(delimIdx + 2);
        const parsed = parseFrame(frame);
        if (parsed) opts.onEvent(parsed);
        delimIdx = buffer.indexOf("\n\n");
      }
    }
  } finally {
    try {
      reader.releaseLock();
    } catch {
      /* noop */
    }
  }

  return sessionId;
}

function parseFrame(frame: string): AgentStreamEvent | null {
  let eventName = "message";
  const dataLines: string[] = [];
  for (const line of frame.split("\n")) {
    if (!line || line.startsWith(":")) continue;
    if (line.startsWith("event:")) {
      eventName = line.slice(6).trim();
    } else if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).trim());
    }
  }
  if (dataLines.length === 0) return null;
  const joined = dataLines.join("\n");
  try {
    return { event: eventName, data: JSON.parse(joined) };
  } catch {
    return { event: eventName, data: { raw: joined } };
  }
}
