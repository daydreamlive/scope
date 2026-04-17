/**
 * Billing API client for communicating with the Daydream API credits endpoints.
 */

const DAYDREAM_API_BASE =
  (import.meta.env.VITE_DAYDREAM_API_BASE as string | undefined) ||
  "https://api.daydream.live";

// ─── Types ───────────────────────────────────────────────────────────────────

export interface CreditsBalance {
  tier: "free" | "pro" | "max";
  credits: {
    balance: number;
    periodCredits: number;
    rolloverBalance?: number;
    total?: number;
    apiBalance?: number;
    lastApiResetMonth?: string | null;
  } | null;
  subscription: {
    status: string;
    currentPeriodEnd: string;
    cancelAtPeriodEnd: boolean;
    overageEnabled: boolean;
  } | null;
  creditsPerMin: number | Record<string, number>;
}

export const DASHBOARD_USAGE_URL = `${(import.meta.env.VITE_DAYDREAM_APP_BASE as string | undefined) || "https://app.daydream.live"}/dashboard/usage`;

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
  const url = deviceId
    ? `${DAYDREAM_API_BASE}/credits/balance?deviceId=${encodeURIComponent(deviceId)}`
    : `${DAYDREAM_API_BASE}/credits/balance`;
  const res = await fetch(url, { headers: headers(apiKey) });
  if (!res.ok)
    throw new Error(`Failed to fetch credits balance: ${res.status}`);
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
  code: string
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
