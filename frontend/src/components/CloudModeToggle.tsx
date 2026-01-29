import { useState } from "react";
import { Toggle } from "./ui/toggle";
import { Input } from "./ui/input";
import { LabelWithTooltip } from "./ui/label-with-tooltip";
import { Loader2, Cloud, Monitor, AlertCircle } from "lucide-react";
import type { CloudModeState } from "../types";

interface CloudModeToggleProps {
  cloudMode: CloudModeState;
  onCloudModeChange: (cloudMode: Partial<CloudModeState>) => void;
  disabled?: boolean;
}

export function CloudModeToggle({
  cloudMode,
  onCloudModeChange,
  disabled = false,
}: CloudModeToggleProps) {
  const [isConnecting, setIsConnecting] = useState(false);

  const handleToggle = async (enabled: boolean) => {
    if (enabled) {
      // Validate credentials before connecting
      if (!cloudMode.appId || !cloudMode.apiKey) {
        onCloudModeChange({
          status: "error",
          errorMessage: "Please enter fal App ID and API Key",
        });
        return;
      }

      setIsConnecting(true);
      onCloudModeChange({ status: "connecting" });

      try {
        const response = await fetch("/api/v1/fal/connect", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            app_id: cloudMode.appId,
            api_key: cloudMode.apiKey,
          }),
        });

        if (!response.ok) {
          const error = await response.json();
          throw new Error(error.detail || "Failed to connect to cloud");
        }

        onCloudModeChange({
          enabled: true,
          status: "connected",
          errorMessage: undefined,
        });
      } catch (error) {
        onCloudModeChange({
          enabled: false,
          status: "error",
          errorMessage:
            error instanceof Error ? error.message : "Connection failed",
        });
      } finally {
        setIsConnecting(false);
      }
    } else {
      // Disconnect from cloud
      setIsConnecting(true);
      try {
        await fetch("/api/v1/fal/disconnect", { method: "POST" });
        onCloudModeChange({
          enabled: false,
          status: "disconnected",
          errorMessage: undefined,
        });
      } catch (error) {
        console.error("Failed to disconnect from cloud:", error);
        // Still mark as disconnected since we're disabling
        onCloudModeChange({
          enabled: false,
          status: "disconnected",
          errorMessage: undefined,
        });
      } finally {
        setIsConnecting(false);
      }
    }
  };

  const isLoading = isConnecting || cloudMode.status === "connecting";

  return (
    <div className="space-y-3">
      {/* Cloud Mode Toggle */}
      <div className="flex items-center justify-between gap-2">
        <LabelWithTooltip
          label="Cloud GPU"
          tooltip="Route video processing to fal.ai cloud for remote GPU inference. Useful when you don't have a local GPU or want to use cloud GPUs."
          className="text-sm font-medium"
        />
        <div className="flex items-center gap-2">
          {isLoading && <Loader2 className="h-4 w-4 animate-spin" />}
          <Toggle
            pressed={cloudMode.enabled}
            onPressedChange={handleToggle}
            variant="outline"
            size="sm"
            className="h-7"
            disabled={disabled || isLoading}
          >
            {cloudMode.enabled ? (
              <>
                <Cloud className="h-3.5 w-3.5 mr-1" />
                Cloud
              </>
            ) : (
              <>
                <Monitor className="h-3.5 w-3.5 mr-1" />
                Local
              </>
            )}
          </Toggle>
        </div>
      </div>

      {/* Status indicator */}
      {cloudMode.status === "connected" && (
        <div className="flex items-center gap-1.5 text-xs text-green-600 dark:text-green-500">
          <div className="h-1.5 w-1.5 rounded-full bg-green-500" />
          Connected to cloud
        </div>
      )}

      {/* Error message */}
      {cloudMode.status === "error" && cloudMode.errorMessage && (
        <div className="flex items-start gap-1.5 p-2 rounded-md bg-red-500/10 border border-red-500/20">
          <AlertCircle className="h-3.5 w-3.5 mt-0.5 shrink-0 text-red-600 dark:text-red-500" />
          <p className="text-xs text-red-600 dark:text-red-500">
            {cloudMode.errorMessage}
          </p>
        </div>
      )}

      {/* Cloud credentials - always show for configuration */}
      <div className="rounded-lg border bg-card p-3 space-y-3">
        <div className="space-y-2">
          <LabelWithTooltip
            label="fal App ID"
            tooltip="Your deployed fal app ID (e.g., 'username/scope-fal/webrtc')"
            className="text-xs text-muted-foreground"
          />
          <Input
            type="text"
            value={cloudMode.appId}
            onChange={e => onCloudModeChange({ appId: e.target.value })}
            placeholder="username/scope-fal/webrtc"
            className="h-8 text-sm"
            disabled={cloudMode.enabled || isLoading}
          />
        </div>

        <div className="space-y-2">
          <LabelWithTooltip
            label="fal API Key"
            tooltip="Your fal API key for authentication"
            className="text-xs text-muted-foreground"
          />
          <Input
            type="password"
            value={cloudMode.apiKey}
            onChange={e => onCloudModeChange({ apiKey: e.target.value })}
            placeholder="Enter your fal API key"
            className="h-8 text-sm"
            disabled={cloudMode.enabled || isLoading}
          />
        </div>
      </div>
    </div>
  );
}
