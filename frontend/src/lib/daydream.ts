const DAYDREAM_API_BASE =
  (import.meta as any).env?.VITE_DAYDREAM_API_BASE || "https://api.daydream.live";
const DAYDREAM_API_KEY = (import.meta as any).env?.VITE_DAYDREAM_API_KEY;

interface CreateStreamRequest {
  pipeline_id: string;
  load_params?: Record<string, unknown> | null;
}

interface CreateStreamResponse {
  id: string;
  whip_url?: string;
  stream_key?: string;
  // other fields may exist; we mainly need id and whip_url
}

export interface DaydreamWebRTCOfferRequest {
  sdp?: string;
  type?: string;
  initialParameters?: Record<string, unknown>;
}

export async function createCloudStream(req: CreateStreamRequest) {
  if (!DAYDREAM_API_KEY) {
    throw new Error("VITE_DAYDREAM_API_KEY is not set");
  }

  const response = await fetch(`${DAYDREAM_API_BASE}/v1/streams`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${DAYDREAM_API_KEY}`,
    },
    body: JSON.stringify(req),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(
      `Daydream create stream failed: ${response.status} ${response.statusText}: ${errorText}`
    );
  }

  const result = (await response.json()) as CreateStreamResponse;
  return result;
}

export async function sendDaydreamWebRTCOffer(
  streamId: string,
  data: DaydreamWebRTCOfferRequest
): Promise<RTCSessionDescriptionInit> {
  if (!DAYDREAM_API_KEY) {
    throw new Error("VITE_DAYDREAM_API_KEY is not set");
  }

  const response = await fetch(
    `${DAYDREAM_API_BASE}/v1/streams/${encodeURIComponent(streamId)}/webrtc/offer`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${DAYDREAM_API_KEY}`,
      },
      body: JSON.stringify(data),
    }
  );

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(
      `Daydream WebRTC offer failed: ${response.status} ${response.statusText}: ${errorText}`
    );
  }

  const result = await response.json();
  return result;
}
