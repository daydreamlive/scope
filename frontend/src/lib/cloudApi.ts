import { getDaydreamAPIKey, getDaydreamUserId } from "./auth";

/**
 * Connect to the cloud relay. Reads credentials from local auth storage
 * internally so callers don't need to pass them around.
 *
 * Returns the fetch Response so callers can inspect status if needed,
 * or `null` if no user is signed in.
 */
export async function connectToCloud(): Promise<Response | null> {
  const userId = getDaydreamUserId();
  if (!userId) return null;

  const apiKey = getDaydreamAPIKey();
  return fetch("/api/v1/cloud/connect", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_id: userId, api_key: apiKey }),
  });
}
