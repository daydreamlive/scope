/**
 * CloudGpuSelector — split button + dropdown for picking a cloud GPU.
 *
 * Replaces the on/off Switch in Settings → Daydream Account. The main button
 * reflects the currently-selected (or last-used) GPU, and the caret opens a
 * dropdown with the three GPU options + per-minute credit cost. Selecting an
 * item persists the choice and immediately initiates a cloud connection.
 *
 * States (driven off `useCloudStatus()`):
 *   - Disconnected: "Run on {GPU}"          + [caret]
 *   - Connecting:   "Connecting to {GPU}…"  + [X cancel], main button disabled
 *   - Connected:    "Connected to {GPU}"    + [caret] → "Switch GPU" / "Disconnect"
 *
 * Onboarding paths (CloudAuthStep, CloudConnectingStep) still call
 * `connectToCloud()` with no argument, so first-time users always land on H100
 * regardless of what this component has stored.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { ChevronDown, Loader2, X } from "lucide-react";
import { Button } from "../ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "../ui/dropdown-menu";
import { useCloudStatus } from "../../hooks/useCloudStatus";
import { connectToCloud } from "../../lib/cloudApi";
import {
  CLOUD_GPUS,
  cloudGpuLabel,
  formatCreditsPerMin,
  getStoredCloudGpu,
  setStoredCloudGpu,
  type CloudGpu,
} from "../../lib/cloudGpu";

interface CloudGpuSelectorProps {
  /** Disable interaction (e.g. when not signed in or when streaming). */
  disabled?: boolean;
  /** Called after a successful connect transition so the pipeline list refreshes. */
  onPipelinesRefresh?: () => Promise<unknown>;
}

export function CloudGpuSelector({
  disabled = false,
  onPipelinesRefresh,
}: CloudGpuSelectorProps) {
  const { status, refresh } = useCloudStatus();
  const [selectedGpu, setSelectedGpu] = useState<CloudGpu>(() =>
    getStoredCloudGpu()
  );
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  /** When true, auto-open the dropdown as soon as we return to disconnected. */
  const pendingSwitchRef = useRef(false);
  const prevConnectedRef = useRef(false);

  // Fire onPipelinesRefresh on the connecting → connected transition.
  useEffect(() => {
    if (!prevConnectedRef.current && status.connected) {
      onPipelinesRefresh?.().catch(e =>
        console.error("[CloudGpuSelector] Failed to refresh pipelines:", e)
      );
    }
    prevConnectedRef.current = status.connected;
  }, [status.connected, onPipelinesRefresh]);

  // After a "Switch GPU" click, auto-open the picker when disconnect lands.
  useEffect(() => {
    if (pendingSwitchRef.current && !status.connected && !status.connecting) {
      pendingSwitchRef.current = false;
      setOpen(true);
    }
  }, [status.connected, status.connecting]);

  const disconnect = useCallback(async (): Promise<boolean> => {
    try {
      const res = await fetch("/api/v1/cloud/disconnect", { method: "POST" });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.detail || "Disconnect failed");
      }
      await refresh();
      return true;
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Disconnect failed";
      setError(msg);
      console.error("[CloudGpuSelector] Disconnect failed:", e);
      return false;
    }
  }, [refresh]);

  const handlePickGpu = useCallback(
    async (gpu: CloudGpu) => {
      setSelectedGpu(gpu);
      setStoredCloudGpu(gpu);
      setOpen(false);
      setError(null);
      setBusy(true);
      try {
        const res = await connectToCloud(gpu);
        if (!res || !res.ok) {
          const data = res ? await res.json().catch(() => ({})) : {};
          throw new Error(data.detail || "Connection failed");
        }
        await refresh();
      } catch (e) {
        const msg = e instanceof Error ? e.message : "Connection failed";
        setError(msg);
        console.error("[CloudGpuSelector] Connect failed:", e);
      } finally {
        setBusy(false);
      }
    },
    [refresh]
  );

  const handleCancel = useCallback(async () => {
    setBusy(true);
    await disconnect();
    setBusy(false);
  }, [disconnect]);

  const handleDisconnect = useCallback(async () => {
    setOpen(false);
    setBusy(true);
    pendingSwitchRef.current = false;
    await disconnect();
    setBusy(false);
  }, [disconnect]);

  const handleSwitchGpu = useCallback(async () => {
    setOpen(false);
    setBusy(true);
    pendingSwitchRef.current = true;
    const ok = await disconnect();
    if (!ok) pendingSwitchRef.current = false;
    setBusy(false);
  }, [disconnect]);

  const { connected, connecting } = status;
  const label = cloudGpuLabel(selectedGpu);

  // Main button text varies with state.
  let mainText: string;
  if (connecting) mainText = `Connecting to ${label}…`;
  else if (connected) mainText = `Connected to ${label}`;
  else mainText = `Run on ${label}`;

  const mainDisabled = disabled || connecting || busy;
  const caretDisabled = disabled || busy;

  const handleMainClick = () => {
    if (mainDisabled) return;
    setOpen(true);
  };

  return (
    <div className="flex flex-col gap-2" data-testid="cloud-gpu-selector">
      <DropdownMenu open={open} onOpenChange={setOpen}>
        <div className="inline-flex items-stretch">
          {/* Main (left) button — opens dropdown in disconnected/connected states. */}
          <Button
            type="button"
            variant="default"
            size="sm"
            onClick={handleMainClick}
            disabled={mainDisabled}
            className="rounded-r-none border-r border-primary-foreground/20 min-w-[10rem] justify-start"
          >
            {connecting && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
            <span className="truncate">{mainText}</span>
          </Button>

          {/* Right (caret or cancel). */}
          {connecting ? (
            <Button
              type="button"
              variant="default"
              size="sm"
              onClick={handleCancel}
              disabled={caretDisabled}
              className="rounded-l-none px-2"
              aria-label="Cancel connection"
              title="Cancel"
            >
              <X className="h-4 w-4" />
            </Button>
          ) : (
            <DropdownMenuTrigger asChild>
              <Button
                type="button"
                variant="default"
                size="sm"
                disabled={caretDisabled}
                className="rounded-l-none px-2"
                aria-label="Select GPU"
              >
                <ChevronDown className="h-4 w-4" />
              </Button>
            </DropdownMenuTrigger>
          )}
        </div>

        {/* Dropdown content differs between disconnected and connected. */}
        {connected ? (
          <DropdownMenuContent align="end" className="min-w-[12rem]">
            <DropdownMenuItem onSelect={handleSwitchGpu}>
              Switch GPU
            </DropdownMenuItem>
            <DropdownMenuItem onSelect={handleDisconnect}>
              Disconnect
            </DropdownMenuItem>
          </DropdownMenuContent>
        ) : (
          <DropdownMenuContent align="end" className="min-w-[16rem]">
            {CLOUD_GPUS.map(g => (
              <DropdownMenuItem
                key={g.id}
                onSelect={() => handlePickGpu(g.id)}
                className="justify-between gap-6"
              >
                <span>{g.label}</span>
                <span className="text-xs text-muted-foreground">
                  {formatCreditsPerMin(g.creditsPerMin)}
                </span>
              </DropdownMenuItem>
            ))}
          </DropdownMenuContent>
        )}
      </DropdownMenu>

      {(error || status.error) && (
        <p className="text-xs text-destructive">{error || status.error}</p>
      )}
    </div>
  );
}
