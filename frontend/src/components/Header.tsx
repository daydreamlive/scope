import { useState, useEffect } from "react";
import { Settings, BookOpenText, Bug, LogIn, LogOut } from "lucide-react";
import { Button } from "./ui/button";
import { SettingsDialog } from "./SettingsDialog";
import { ReportBugDialog } from "./ReportBugDialog";
import {
  isAuthenticated,
  redirectToSignIn,
  clearDaydreamAPIKey,
} from "../lib/auth";

interface HeaderProps {
  className?: string;
}

export function Header({ className = "" }: HeaderProps) {
  // Settings dialog state
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [initialTab, setInitialTab] = useState<"general" | "plugins">(
    "general"
  );
  const [initialPluginPath, setInitialPluginPath] = useState("");

  // Report bug dialog state
  const [reportBugOpen, setReportBugOpen] = useState(false);

  // Auth state
  const [authenticated, setAuthenticated] = useState(false);

  useEffect(() => {
    // Handle deep link actions for plugin installation
    if (window.scope?.onDeepLinkAction) {
      return window.scope.onDeepLinkAction(data => {
        if (data.action === "install-plugin" && data.package) {
          setInitialTab("plugins");
          setInitialPluginPath(data.package);
          setSettingsOpen(true);
        }
      });
    }
  }, []);

  useEffect(() => {
    // Check authentication status on mount
    setAuthenticated(isAuthenticated());

    // Listen for auth state changes
    const handleAuthChange = () => {
      setAuthenticated(isAuthenticated());
    };

    window.addEventListener("daydream-auth-change", handleAuthChange);

    return () => {
      window.removeEventListener("daydream-auth-change", handleAuthChange);
    };
  }, []);

  const handleSettingsClose = () => {
    setSettingsOpen(false);
    setInitialTab("general");
    setInitialPluginPath("");
  };

  const handleSignIn = () => {
    redirectToSignIn();
  };

  const handleSignOut = () => {
    clearDaydreamAPIKey();
    setAuthenticated(false);
  };

  return (
    <header className={`w-full bg-background px-6 py-4 ${className}`}>
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-medium text-foreground">Daydream Scope</h1>
        <div className="flex items-center gap-3">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setReportBugOpen(true)}
            className="hover:opacity-80 transition-opacity gap-1.5 text-muted-foreground opacity-60"
          >
            <Bug className="h-4 w-4" />
            <span className="text-xs">Report Bug</span>
          </Button>
          <a
            href="https://github.com/daydreamlive/scope"
            target="_blank"
            rel="noopener noreferrer"
            className="hover:opacity-80 transition-opacity"
          >
            <img
              src="/assets/github-mark-white.svg"
              alt="GitHub"
              className="h-5 w-5 opacity-60"
            />
          </a>
          <a
            href="https://discord.gg/mnfGR4Fjhp"
            target="_blank"
            rel="noopener noreferrer"
            className="hover:opacity-80 transition-opacity"
          >
            <img
              src="/assets/discord-symbol-white.svg"
              alt="Discord"
              className="h-5 w-5 opacity-60"
            />
          </a>
          <a
            href="https://docs.daydream.live/knowledge-hub/tutorials/scope"
            target="_blank"
            rel="noopener noreferrer"
            className="hover:opacity-80 transition-opacity"
          >
            <BookOpenText className="h-5 w-5 text-muted-foreground opacity-60" />
          </a>
          {authenticated ? (
            <Button
              variant="ghost"
              size="sm"
              onClick={handleSignOut}
              className="hover:opacity-80 transition-opacity gap-1.5"
            >
              <LogOut className="h-4 w-4" />
              <span className="text-xs">Sign Out</span>
            </Button>
          ) : (
            <Button
              variant="ghost"
              size="sm"
              onClick={handleSignIn}
              className="hover:opacity-80 transition-opacity gap-1.5"
            >
              <LogIn className="h-4 w-4" />
              <span className="text-xs">Sign In</span>
            </Button>
          )}
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setSettingsOpen(true)}
            className="hover:opacity-80 transition-opacity text-muted-foreground opacity-60 h-8 w-8"
            title="Settings"
          >
            <Settings className="h-5 w-5" />
          </Button>
        </div>
      </div>

      <SettingsDialog
        open={settingsOpen}
        onClose={handleSettingsClose}
        initialTab={initialTab}
        initialPluginPath={initialPluginPath}
      />

      <ReportBugDialog open={reportBugOpen} onOpenChange={setReportBugOpen} />
    </header>
  );
}
