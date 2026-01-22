const DAYDREAM_AUTH_URL = "https://app.livepeer.monster/sign-in/local?port=8000";
const DAYDREAM_API_BASE = (import.meta as any).env?.VITE_DAYDREAM_API_BASE || "https://api.daydream.monster";
const API_KEY_STORAGE_KEY = "daydream_api_key";

/**
 * Get the stored Daydream API key from localStorage or environment variable
 */
export function getDaydreamAPIKey(): string | null {
  // First check localStorage for a user-authenticated key
  const storedKey = localStorage.getItem(API_KEY_STORAGE_KEY);
  if (storedKey) {
    return storedKey;
  }

  // Fall back to environment variable if available
  const envKey = (import.meta as any).env?.VITE_DAYDREAM_API_KEY;
  return envKey || null;
}

/**
 * Save the Daydream API key to localStorage
 */
export function saveDaydreamAPIKey(apiKey: string): void {
  localStorage.setItem(API_KEY_STORAGE_KEY, apiKey);
  // Dispatch custom event to notify components of auth state change
  window.dispatchEvent(new CustomEvent("daydream-auth-change"));
}

/**
 * Clear the stored Daydream API key
 */
export function clearDaydreamAPIKey(): void {
  localStorage.removeItem(API_KEY_STORAGE_KEY);
  // Dispatch custom event to notify components of auth state change
  window.dispatchEvent(new CustomEvent("daydream-auth-change"));
}

/**
 * Check if user is authenticated (has an API key)
 */
export function isAuthenticated(): boolean {
  return getDaydreamAPIKey() !== null;
}

/**
 * Redirect to Daydream sign-in page
 */
export function redirectToSignIn(): void {
  window.location.href = DAYDREAM_AUTH_URL;
}

/**
 * Exchange a short-lived token for a long-lived API key
 */
export async function exchangeTokenForAPIKey(token: string): Promise<string> {
  // TODO if this fails, log out and redirect to sign in page
  // TODO check state queryparam
  const response = await fetch(`${DAYDREAM_API_BASE}/v1/api-key`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${token}`,
    },
    body: JSON.stringify({
      "name": "foo",
    }),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(
      `Failed to exchange token for API key: ${response.status} ${response.statusText}: ${errorText}`
    );
  }

  const result = await response.json();

  // The API should return an object with an api_key field
  // Adjust this based on the actual API response structure
  if (!result.api_key && !result.apiKey && !result.key) {
    throw new Error("API response did not contain an API key");
  }

  return result.api_key || result.apiKey || result.key;
}

/**
 * Handle OAuth callback - extract token from URL and exchange it for API key
 * Returns true if callback was handled, false otherwise
 */
export async function handleOAuthCallback(): Promise<boolean> {
  const urlParams = new URLSearchParams(window.location.search);
  const token = urlParams.get("token");

  if (!token) {
    return false;
  }

  try {
    // Exchange the short-lived token for a long-lived API key
    const apiKey = await exchangeTokenForAPIKey(token);

    // Save the API key to localStorage
    saveDaydreamAPIKey(apiKey);

    // Clean up the URL by removing the token parameter
    const url = new URL(window.location.href);
    url.searchParams.delete("token");
    window.history.replaceState({}, document.title, url.toString());

    return true;
  } catch (error) {
    console.error("Failed to exchange token for API key:", error);
    throw error;
  }
}
