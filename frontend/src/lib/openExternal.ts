/**
 * Open a URL in the system browser in both web and Electron contexts.
 *
 * - Electron: uses the IPC-exposed `window.scope.openExternal(url)`.
 * - Browser:  uses `window.open(url, "_blank")`.
 */
export function openExternalUrl(url: string): void {
  const scope = (
    window as unknown as {
      scope?: { openExternal?: (u: string) => Promise<boolean> };
    }
  ).scope;
  if (scope?.openExternal) {
    scope.openExternal(url).catch(err => {
      console.error("Failed to open URL via Electron IPC:", err);
    });
  } else {
    window.open(url, "_blank");
  }
}
