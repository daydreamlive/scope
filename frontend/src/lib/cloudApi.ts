import { getDaydreamAPIKey, getDaydreamUserId } from "./auth";
import type { CloudGpu } from "./cloudGpu";

/**
 * Connect to the cloud relay. Reads credentials from local auth storage
 * internally so callers don't need to pass them around.
 *
 * @param gpu Optional GPU selector for Livepeer cloud app routing. When
 *   omitted, the backend falls back to the SCOPE_CLOUD_GPU env var (or H100).
 *   Onboarding paths should call with no argument so first-time users always
 *   land on H100.
 *
 * Returns the fetch Response so callers can inspect status if needed,
 * or `null` if no user is signed in.
 */
export async function connectToCloud(gpu?: CloudGpu): Promise<Response | null> {
  const userId = getDaydreamUserId();
  if (!userId) return null;

  const apiKey = getDaydreamAPIKey();
  return fetch("/api/v1/cloud/connect", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_id: userId, api_key: apiKey, gpu }),
  });
}
