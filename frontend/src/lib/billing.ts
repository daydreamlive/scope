/**
 * Billing API client for communicating with the Daydream API credits endpoints.
 */

const DAYDREAM_API_BASE =
  (import.meta.env.VITE_DAYDREAM_API_BASE as string | undefined) ||
  "https://api.daydream.live";

// ─── Types ───────────────────────────────────────────────────────────────────

export interface TrialStatus {
  secondsUsed: number;
  secondsLimit: number;
  exhausted: boolean;
}

export interface TrialHeartbeatResponse extends TrialStatus {
  hasSubscription: boolean;
}

export interface CreditsBalance {
  tier: "free" | "basic" | "pro";
  credits: { balance: number; periodCredits: number } | null;
  subscription: {
    status: string;
    currentPeriodEnd: string;
    cancelAtPeriodEnd: boolean;
    overageEnabled: boolean;
  } | null;
  trial: TrialStatus | null;
  creditsPerMin: number;
}

// ─── API functions ───────────────────────────────────────────────────────────

function headers(apiKey: string | null): Record<string, string> {
  const h: Record<string, string> = { "Content-Type": "application/json" };
  if (apiKey) h["Authorization"] = `Bearer ${apiKey}`;
  return h;
}

export async function fetchCreditsBalance(
  apiKey: string,
  deviceId?: string
): Promise<CreditsBalance> {
  const h = headers(apiKey);
  if (deviceId) h["x-device-id"] = deviceId;

  const res = await fetch(`${DAYDREAM_API_BASE}/credits/balance`, {
    headers: h,
  });
  if (!res.ok)
    throw new Error(`Failed to fetch credits balance: ${res.status}`);
  return res.json();
}

export async function sendTrialHeartbeat(
  apiKey: string | null,
  deviceId: string
): Promise<TrialHeartbeatResponse> {
  const res = await fetch(`${DAYDREAM_API_BASE}/credits/trial/heartbeat`, {
    method: "POST",
    headers: headers(apiKey),
    body: JSON.stringify({ deviceId }),
  });
  if (!res.ok) throw new Error(`Failed to send trial heartbeat: ${res.status}`);
  return res.json();
}

export async function fetchTrialStatus(deviceId: string): Promise<TrialStatus> {
  const res = await fetch(
    `${DAYDREAM_API_BASE}/credits/trial/status?deviceId=${encodeURIComponent(deviceId)}`
  );
  if (!res.ok) throw new Error(`Failed to fetch trial status: ${res.status}`);
  return res.json();
}

export async function createCheckoutSession(
  apiKey: string,
  tier: "basic" | "pro"
): Promise<{ checkoutUrl: string }> {
  const res = await fetch(`${DAYDREAM_API_BASE}/credits/checkout`, {
    method: "POST",
    headers: headers(apiKey),
    body: JSON.stringify({ tier }),
  });
  if (!res.ok)
    throw new Error(`Failed to create checkout session: ${res.status}`);
  return res.json();
}

export async function createPortalSession(
  apiKey: string
): Promise<{ portalUrl: string }> {
  const res = await fetch(`${DAYDREAM_API_BASE}/credits/portal`, {
    method: "POST",
    headers: headers(apiKey),
  });
  if (!res.ok)
    throw new Error(`Failed to create portal session: ${res.status}`);
  return res.json();
}

export async function setOverageEnabled(
  apiKey: string,
  enabled: boolean
): Promise<void> {
  const res = await fetch(`${DAYDREAM_API_BASE}/credits/overage`, {
    method: "POST",
    headers: headers(apiKey),
    body: JSON.stringify({ enabled }),
  });
  if (!res.ok) throw new Error(`Failed to set overage: ${res.status}`);
}

export interface RedeemCodeResponse {
  credits: number;
  label: string | null;
  newBalance: number;
}

export async function redeemCreditCode(
  apiKey: string,
  code: string,
): Promise<RedeemCodeResponse> {
  const res = await fetch(`${DAYDREAM_API_BASE}/credits/codes/redeem`, {
    method: "POST",
    headers: headers(apiKey),
    body: JSON.stringify({ code }),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new Error(body?.message ?? `Failed to redeem code: ${res.status}`);
  }
  return res.json();
}
