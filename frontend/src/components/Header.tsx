import { useState, useEffect, useRef } from "react";
import {
  Settings,
  Cloud,
  CloudOff,
  Plug,
  Workflow,
  Monitor,
  Clock,
  AlertTriangle,
  HelpCircle,
} from "lucide-react";
import { Button } from "./ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "./ui/tooltip";
import { SettingsDialog } from "./SettingsDialog";
import { PluginsDialog } from "./PluginsDialog";
import { PaywallModal } from "./PaywallModal";
import { toast } from "sonner";
import { useCloudStatus } from "../hooks/useCloudStatus";
import { useBilling } from "../contexts/BillingContext";
import { isAuthenticated, redirectToSignIn } from "../lib/auth";
import { openExternalUrl } from "../lib/openExternal";

const DAYDREAM_APP_BASE =
  (import.meta.env.VITE_DAYDREAM_APP_BASE as string | undefined) ||
  "https://app.daydream.live";

function formatTrialTime(totalSeconds: number): string {
  const s = Math.max(0, Math.floor(totalSeconds));
  const m = Math.floor(s / 60);
  const sec = s % 60;
  return `${m}:${String(sec).padStart(2, "0")}`;
}

interface HeaderProps {
  className?: string;
  onPipelinesRefresh?: () => Promise<unknown>;
  cloudDisabled?: boolean;
  // External settings tab control
  openSettingsTab?: string | null;
  onSettingsTabOpened?: () => void;
  // External plugins tab control (e.g. from starter workflows chip)
  openPluginsTab?: string | null;
  onPluginsTabOpened?: () => void;
  // Graph mode toggle
  graphMode?: boolean;
  onGraphModeToggle?: () => void;
  // Workflow loading from Workflows tab
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  onLoadWorkflow?: (workflowData: Record<string, any>) => void;
}

export function Header({
  className = "",
  onPipelinesRefresh,
  cloudDisabled,
  openSettingsTab,
  onSettingsTabOpened,
  openPluginsTab,
  onPluginsTabOpened,
  graphMode = false,
  onGraphModeToggle,
  onLoadWorkflow,
}: HeaderProps) {
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [pluginsOpen, setPluginsOpen] = useState(false);
  const [initialTab, setInitialTab] = useState<
    "general" | "account" | "api-keys" | "loras" | "osc" | "billing"
  >("general");
  const [initialPluginPath, setInitialPluginPath] = useState("");
  const [pluginsInitialTab, setPluginsInitialTab] = useState<
    string | undefined
  >(undefined);

  // Use shared cloud status hook - single source of truth
  const { isConnected, isConnecting, lastCloseCode, lastCloseReason } =
    useCloudStatus();

  // Billing state
  const billing = useBilling();

  // Auth state — reactive to sign-in / sign-out
  const [isSignedIn, setIsSignedIn] = useState(() => isAuthenticated());

  useEffect(() => {
    const handleAuthChange = () => setIsSignedIn(isAuthenticated());
    window.addEventListener("daydream-auth-change", handleAuthChange);
    window.addEventListener("daydream-auth-success", handleAuthChange);
    return () => {
      window.removeEventListener("daydream-auth-change", handleAuthChange);
      window.removeEventListener("daydream-auth-success", handleAuthChange);
    };
  }, []);

  // Track the last close code we've shown a toast for to avoid duplicates
  const lastNotifiedCloseCodeRef = useRef<number | null>(null);

  // Only show "connection lost" after we've seen a successful connection this session
  const hasBeenConnectedRef = useRef(false);

  // Track previous connection state to detect transitions for pipeline refresh
  const prevConnectedRef = useRef(false);

  // Detect unexpected disconnection and show toast
  useEffect(() => {
    if (isConnected) {
      hasBeenConnectedRef.current = true;
      lastNotifiedCloseCodeRef.current = null;
    }

    if (
      hasBeenConnectedRef.current &&
      lastCloseCode !== null &&
      lastCloseCode !== lastNotifiedCloseCodeRef.current
    ) {
      console.warn(
        `[Header] Cloud WebSocket closed unexpectedly (code=${lastCloseCode}, reason=${lastCloseReason})`
      );
      toast.error("Cloud connection lost", {
        description: `WebSocket closed ${lastCloseReason ? `(${lastCloseReason})` : ""}`,
        duration: 10000,
      });
      lastNotifiedCloseCodeRef.current = lastCloseCode;
    }
  }, [lastCloseCode, lastCloseReason, isConnected]);

  // Refresh pipelines when cloud connection status changes
  // This ensures pipeline list updates even if settings dialog is closed
  useEffect(() => {
    if (prevConnectedRef.current !== isConnected) {
      // Connection status changed - refresh pipelines to get the right list
      onPipelinesRefresh?.().catch(e =>
        console.error(
          "[Header] Failed to refresh pipelines after cloud status change:",
          e
        )
      );
    }
    prevConnectedRef.current = isConnected;
  }, [isConnected, onPipelinesRefresh]);

  const handleCloudIconClick = () => {
    setInitialTab("account");
    setSettingsOpen(true);
  };

  // React to external requests to open a specific settings/plugins tab
  useEffect(() => {
    if (openSettingsTab) {
      if (openSettingsTab === "plugins") {
        setPluginsOpen(true);
      } else {
        setInitialTab(
          openSettingsTab as
            | "general"
            | "account"
            | "api-keys"
            | "loras"
            | "osc"
            | "billing"
        );
        setSettingsOpen(true);
      }
      onSettingsTabOpened?.();
    }
  }, [openSettingsTab, onSettingsTabOpened]);

  // React to external requests to open a specific plugins dialog tab
  useEffect(() => {
    if (openPluginsTab) {
      setPluginsInitialTab(openPluginsTab);
      setPluginsOpen(true);
      onPluginsTabOpened?.();
    }
  }, [openPluginsTab, onPluginsTabOpened]);

  useEffect(() => {
    // Handle deep link actions for plugin installation
    if (window.scope?.onDeepLinkAction) {
      return window.scope.onDeepLinkAction(data => {
        if (data.action === "install-plugin" && data.package) {
          setInitialPluginPath(data.package);
          setPluginsOpen(true);
        }
      });
    }
  }, []);

  const handleSettingsClose = () => {
    setSettingsOpen(false);
    setInitialTab("general");
  };

  const handlePluginsClose = () => {
    setPluginsOpen(false);
    setInitialPluginPath("");
    setPluginsInitialTab(undefined);
  };

  return (
    <header className={`w-full bg-background px-6 py-4 ${className}`}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <h1 className="text-xl font-medium text-foreground">
            Daydream Scope
          </h1>
          {onGraphModeToggle && (
            <Button
              variant="ghost"
              size="sm"
              onClick={() => {
                onGraphModeToggle();
              }}
              className="text-xs text-muted-foreground hover:text-foreground transition-colors gap-1.5"
              title={
                graphMode
                  ? "Switch to Perform Mode"
                  : "Switch to Workflow Builder"
              }
            >
              {graphMode ? (
                <>
                  <Monitor className="h-4 w-4" />
                  Perform Mode
                </>
              ) : (
                <>
                  <Workflow className="h-4 w-4" />
                  Workflow Builder
                </>
              )}
            </Button>
          )}
        </div>
        <div className="flex items-center gap-1">
          {/* Credit balance (left of cloud button) */}
          {isSignedIn && billing.credits && (
            <span className="flex items-center gap-1.5 text-xs font-medium px-2 text-muted-foreground">
              <span className="tabular-nums">
                {billing.credits.balance.toFixed(2)}
              </span>{" "}
              credits remaining
              <TooltipProvider delayDuration={0}>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <button
                      type="button"
                      className="inline-flex text-muted-foreground hover:text-foreground transition-colors"
                      aria-label="Credit info"
                    >
                      <HelpCircle className="h-3.5 w-3.5" />
                    </button>
                  </TooltipTrigger>
                  <TooltipContent
                    side="bottom"
                    className="max-w-[260px] text-xs leading-relaxed"
                  >
                    Daydream Cloud inference requires credit purchases. For more
                    information, please refer to our{" "}
                    <a
                      href="https://daydream.live/pricing"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="underline hover:text-primary-foreground/80"
                      onClick={e => {
                        e.preventDefault();
                        openExternalUrl("https://daydream.live/pricing");
                      }}
                    >
                      Pricing page
                    </a>
                    .
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
              <button
                type="button"
                onClick={() =>
                  openExternalUrl(`${DAYDREAM_APP_BASE}/dashboard/usage`)
                }
                className="h-6 px-2 rounded-md text-[11px] font-semibold text-white bg-gradient-to-r from-[#36619D] via-[#2FBEC5] to-[#FF982E] hover:brightness-110 transition-all"
              >
                Top Up
              </button>
            </span>
          )}
          <Button
            variant="ghost"
            size="sm"
            onClick={handleCloudIconClick}
            className={`hover:opacity-80 transition-opacity h-8 gap-1.5 px-2 ${
              isConnected
                ? "text-green-500 opacity-100"
                : isConnecting
                  ? "text-amber-400 opacity-100"
                  : "text-muted-foreground opacity-80"
            }`}
            title={
              isConnected
                ? "Cloud connected"
                : isConnecting
                  ? "Connecting to cloud..."
                  : "Connect to cloud"
            }
          >
            {isConnected ? (
              <Cloud className="h-4 w-4" />
            ) : isConnecting ? (
              <Cloud className="h-4 w-4 animate-pulse" />
            ) : (
              <CloudOff className="h-4 w-4" />
            )}
            <span className="text-xs font-medium">
              {isConnected
                ? "Connected"
                : isConnecting
                  ? "Connecting..."
                  : "Connect to Cloud"}
            </span>
          </Button>
          {/* Upgrade CTA / Plan badge */}
          {!isSignedIn ? (
            <button
              type="button"
              onClick={() => redirectToSignIn()}
              className="h-7 px-3 rounded-md text-xs font-semibold text-white bg-gradient-to-r from-[#36619D] via-[#2FBEC5] to-[#FF982E] hover:brightness-110 transition-all"
            >
              Upgrade
            </button>
          ) : billing.tier === "free" ? (
            <button
              type="button"
              onClick={() => billing.openCheckout("pro")}
              className="h-7 px-3 rounded-md text-xs font-semibold text-white bg-gradient-to-r from-[#36619D] via-[#2FBEC5] to-[#FF982E] hover:brightness-110 transition-all"
            >
              Upgrade for more credits
            </button>
          ) : (
            <Button
              variant="ghost"
              size="sm"
              onClick={() => {
                setInitialTab("billing");
                setSettingsOpen(true);
              }}
              className="h-8 px-2 text-xs text-muted-foreground hover:text-foreground"
            >
              {billing.tier === "pro" ? "Pro" : "Max"}
            </Button>
          )}
          {isConnected &&
            billing.tier === "free" &&
            billing.trial &&
            !billing.trial.exhausted && (
              <span
                className={`flex items-center gap-1 text-xs font-medium px-2 ${
                  billing.trial.secondsLimit - billing.trial.secondsUsed < 300
                    ? billing.trial.secondsLimit - billing.trial.secondsUsed <
                      60
                      ? "text-red-500"
                      : "text-amber-400"
                    : "text-muted-foreground"
                }`}
              >
                <Clock className="h-3.5 w-3.5" />
                Trial:{" "}
                {formatTrialTime(
                  billing.trial.secondsLimit - billing.trial.secondsUsed
                )}
              </span>
            )}
          {/* Billing unavailable fallback */}
          {isConnected &&
            billing.billingError &&
            !billing.credits &&
            !billing.trial && (
              <span
                className="flex items-center gap-1 text-xs font-medium px-2 text-amber-400"
                title="Unable to load billing status. Usage may not be tracked."
              >
                <AlertTriangle className="h-3.5 w-3.5" />
                Billing unavailable
              </span>
            )}
          <Button
            variant="ghost"
            size="sm"
            onClick={() => {
              setPluginsInitialTab("discover");
              setPluginsOpen(true);
            }}
            className="hover:opacity-80 transition-opacity text-muted-foreground opacity-80 h-8 gap-1.5 px-2"
            title="Nodes"
          >
            <Plug className="h-4 w-4" />
            <span className="text-xs font-medium">Nodes</span>
          </Button>
          <Button
            data-tour="workflows-button"
            variant="ghost"
            size="sm"
            onClick={() => {
              setPluginsInitialTab("workflows");
              setPluginsOpen(true);
            }}
            className="hover:opacity-80 transition-opacity text-muted-foreground opacity-80 h-8 gap-1.5 px-2"
            title="Workflows"
          >
            <Workflow className="h-4 w-4" />
            <span className="text-xs font-medium">Workflows</span>
          </Button>
          <Button
            data-tour="settings-button"
            variant="ghost"
            size="sm"
            onClick={() => setSettingsOpen(true)}
            className="hover:opacity-80 transition-opacity text-muted-foreground opacity-80 h-8 gap-1.5 px-2"
            title="Settings"
          >
            <Settings className="h-4 w-4" />
            <span className="text-xs font-medium">Settings</span>
          </Button>
        </div>
      </div>

      <PluginsDialog
        open={pluginsOpen}
        onClose={handlePluginsClose}
        initialPluginPath={initialPluginPath}
        initialTab={pluginsInitialTab}
        disabled={cloudDisabled || isConnecting}
        cloudConnected={isConnected}
        onLoadWorkflow={onLoadWorkflow}
      />

      <SettingsDialog
        open={settingsOpen}
        onClose={handleSettingsClose}
        initialTab={initialTab}
        onPipelinesRefresh={onPipelinesRefresh}
        cloudDisabled={cloudDisabled}
      />

      <PaywallModal />
    </header>
  );
}
