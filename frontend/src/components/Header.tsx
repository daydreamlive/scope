import { useState, useEffect, useRef } from "react";
import {
  Settings,
  Cloud,
  CloudOff,
  Plug,
  Workflow,
  Monitor,
  Menu as MenuIcon,
  AlertTriangle,
  HelpCircle,
  ExternalLink,
} from "lucide-react";
import { Button } from "./ui/button";
import { ToggleGroup, ToggleGroupItem } from "./ui/toggle-group";
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
import { DASHBOARD_USAGE_URL } from "../lib/billing";
import { isAuthenticated, redirectToSignIn } from "../lib/auth";
import { openExternalUrl } from "../lib/openExternal";
import { useOnboarding } from "../contexts/OnboardingContext";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "./ui/dropdown-menu";

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

  // Onboarding state — used to determine if upgrade CTA should show
  const { state: onboardingState } = useOnboarding();

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

  // Track whether the user has clicked the cloud button this session
  const [hasClickedCloud, setHasClickedCloud] = useState(false);

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
    setHasClickedCloud(true);
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
          <img
            src="/icon-white.svg"
            alt="Daydream Scope"
            className="h-5 w-auto"
          />
          {onGraphModeToggle && (
            <ToggleGroup
              type="single"
              value={graphMode ? "workflow" : "perform"}
              onValueChange={value => {
                if (!value) return;
                const nextGraphMode = value === "workflow";
                if (nextGraphMode !== graphMode) onGraphModeToggle();
              }}
              className="h-8 rounded-md bg-muted/40 p-0.5 gap-0.5"
            >
              <ToggleGroupItem
                value="workflow"
                size="sm"
                className="h-7 px-3 text-xs gap-1.5 data-[state=on]:bg-background data-[state=on]:text-foreground data-[state=on]:shadow-sm text-muted-foreground"
                aria-label="Workflow Builder"
              >
                <Workflow className="h-4 w-4" />
                Workflow
              </ToggleGroupItem>
              <ToggleGroupItem
                value="perform"
                size="sm"
                className="h-7 px-3 text-xs gap-1.5 data-[state=on]:bg-background data-[state=on]:text-foreground data-[state=on]:shadow-sm text-muted-foreground"
                aria-label="Perform Mode"
              >
                <Monitor className="h-4 w-4" />
                Perform
              </ToggleGroupItem>
            </ToggleGroup>
          )}
        </div>
        <div className="flex items-center gap-1">
          <Button
            variant="ghost"
            size="sm"
            onClick={handleCloudIconClick}
            className={`hover:opacity-80 transition-opacity h-8 gap-1.5 px-2 ${
              isConnected
                ? "text-emerald-600 opacity-100"
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
          {/* Upgrade CTA / Plan badge — only show when user has cloud intent */}
          {(onboardingState.inferenceMode === "cloud" ||
            hasClickedCloud ||
            isConnected ||
            isConnecting) && (
            <>
              {!isSignedIn ? (
                <button
                  type="button"
                  onClick={() => redirectToSignIn()}
                  className="inline-flex items-center gap-1.5 h-7 px-3 rounded-md text-xs font-semibold text-white bg-gradient-to-r from-[#2FBEC5] to-[#36619D] hover:brightness-110 transition-all"
                >
                  Upgrade
                  <ExternalLink className="h-3 w-3" />
                </button>
              ) : billing.tier === "free" ? (
                <button
                  type="button"
                  onClick={() => billing.openCheckout("pro")}
                  className="inline-flex items-center gap-1.5 h-7 px-3 rounded-md text-xs font-semibold text-white bg-gradient-to-r from-[#2FBEC5] to-[#36619D] hover:brightness-110 transition-all"
                >
                  Upgrade for more credits
                  <ExternalLink className="h-3 w-3" />
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
            </>
          )}
          {/* Billing unavailable fallback */}
          {isConnected && billing.billingError && !billing.credits && (
            <span
              className="flex items-center gap-1 text-xs font-medium px-2 text-amber-400"
              title="Unable to load billing status. Usage may not be tracked."
            >
              <AlertTriangle className="h-3.5 w-3.5" />
              Billing unavailable
            </span>
          )}
          {/* Menu dropdown: credits, nodes, workflows, settings */}
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button
                variant="ghost"
                size="sm"
                className="hover:opacity-80 transition-opacity text-muted-foreground opacity-80 h-8 gap-1.5 px-2"
              >
                <MenuIcon className="h-4 w-4" />
                <span className="text-xs font-medium">Menu</span>
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-64">
              {isSignedIn && billing.credits && (
                <>
                  <div className="flex flex-col gap-2 px-2 py-1.5">
                    <span className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
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
                            Daydream Cloud inference requires credit purchases.
                            For more information, please refer to our{" "}
                            <a
                              href="https://daydream.live/pricing"
                              target="_blank"
                              rel="noopener noreferrer"
                              className="underline hover:text-primary-foreground/80"
                              onClick={e => {
                                e.preventDefault();
                                openExternalUrl(
                                  "https://daydream.live/pricing"
                                );
                              }}
                            >
                              Pricing page
                            </a>
                            .
                          </TooltipContent>
                        </Tooltip>
                      </TooltipProvider>
                    </span>
                    {(billing.tier === "pro" || billing.tier === "max") && (
                      <button
                        type="button"
                        onClick={() => openExternalUrl(DASHBOARD_USAGE_URL)}
                        className="inline-flex items-center justify-center gap-1.5 h-7 px-3 rounded-md text-xs font-semibold text-white bg-gradient-to-r from-[#2FBEC5] to-[#36619D] hover:brightness-110 transition-all w-full"
                      >
                        Top Up
                        <ExternalLink className="h-3 w-3" />
                      </button>
                    )}
                  </div>
                  <DropdownMenuSeparator />
                </>
              )}
              <DropdownMenuItem
                onClick={() => {
                  setPluginsInitialTab("discover");
                  setPluginsOpen(true);
                }}
              >
                <Plug className="h-4 w-4" />
                Nodes
              </DropdownMenuItem>
              <DropdownMenuItem
                data-tour="workflows-button"
                onClick={() => {
                  setPluginsInitialTab("workflows");
                  setPluginsOpen(true);
                }}
              >
                <Workflow className="h-4 w-4" />
                Workflows
              </DropdownMenuItem>
              <DropdownMenuItem
                data-tour="settings-button"
                onClick={() => setSettingsOpen(true)}
              >
                <Settings className="h-4 w-4" />
                Settings
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
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
