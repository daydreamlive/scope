import { useState, useEffect } from "react";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "./ui/dialog";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./ui/tabs";
import { AccountTab } from "./settings/AccountTab";
import { ApiKeysTab } from "./settings/ApiKeysTab";
import { GeneralTab } from "./settings/GeneralTab";
import { PluginsTab } from "./settings/PluginsTab";
import { ReportBugDialog } from "./ReportBugDialog";
import { usePipelinesContext } from "@/contexts/PipelinesContext";
import { getServerInfo } from "@/lib/api";
import { usePluginOperations } from "@/hooks/usePluginOperations";

interface SettingsDialogProps {
  open: boolean;
  onClose: () => void;
  initialTab?: "general" | "account" | "api-keys" | "plugins";
  initialPluginPath?: string;
  onPipelinesRefresh?: () => Promise<unknown>;
  cloudDisabled?: boolean;
}

// Electron preload exposes scope API on window
interface ScopeAPI {
  browseDirectory?: (title: string) => Promise<string | null>;
  onDeepLinkAction?: (
    callback: (data: { action: string; package: string }) => void
  ) => () => void;
}

declare global {
  interface Window {
    scope?: ScopeAPI;
  }
}

export function SettingsDialog({
  open,
  onClose,
  initialTab = "general",
  initialPluginPath = "",
  onPipelinesRefresh,
  cloudDisabled,
}: SettingsDialogProps) {
  const { refetch: refetchPipelines } = usePipelinesContext();
  const [modelsDirectory, setModelsDirectory] = useState(
    "~/.daydream-scope/models"
  );
  const [logsDirectory, setLogsDirectory] = useState("~/.daydream-scope/logs");
  const [reportBugOpen, setReportBugOpen] = useState(false);
  const [activeTab, setActiveTab] = useState<string>(initialTab);
  const [version, setVersion] = useState<string>("");
  const [gitCommit, setGitCommit] = useState<string>("");

  const pluginOps = usePluginOperations(refetchPipelines);

  // Sync state when dialog opens with initial values from props
  useEffect(() => {
    if (open) {
      setActiveTab(initialTab);
      if (initialPluginPath) {
        pluginOps.setPluginInstallPath(initialPluginPath);
      }
    }
  }, [open, initialTab, initialPluginPath, pluginOps.setPluginInstallPath]);

  // Fetch version when dialog opens
  useEffect(() => {
    if (open) {
      getServerInfo()
        .then(info => {
          setVersion(info.version);
          setGitCommit(info.gitCommit);
        })
        .catch(err => console.error("Failed to fetch server info:", err));
    }
  }, [open]);

  // Fetch plugins when dialog opens or when switching to plugins tab
  useEffect(() => {
    if (open && activeTab === "plugins") {
      pluginOps.fetchPlugins();
    }
  }, [open, activeTab, pluginOps.fetchPlugins]);

  const handleModelsDirectoryChange = (value: string) => {
    console.log("Models directory changed:", value);
    setModelsDirectory(value);
  };

  const handleLogsDirectoryChange = (value: string) => {
    console.log("Logs directory changed:", value);
    setLogsDirectory(value);
  };

  return (
    <Dialog open={open} onOpenChange={isOpen => !isOpen && onClose()}>
      <DialogContent className="sm:max-w-[600px] p-0 gap-0">
        <DialogHeader className="sr-only">
          <DialogTitle>Settings</DialogTitle>
        </DialogHeader>
        <Tabs
          value={activeTab}
          onValueChange={setActiveTab}
          orientation="vertical"
          className="flex items-stretch"
        >
          <TabsList className="flex flex-col items-start justify-start bg-transparent gap-1 w-32 p-4">
            <TabsTrigger
              value="general"
              className="w-full justify-start px-3 py-2 hover:bg-muted/50 data-[state=active]:bg-muted"
            >
              General
            </TabsTrigger>
            <TabsTrigger
              value="account"
              className="w-full justify-start px-3 py-2 hover:bg-muted/50 data-[state=active]:bg-muted"
            >
              Account
            </TabsTrigger>
            <TabsTrigger
              value="api-keys"
              className="w-full justify-start px-3 py-2 hover:bg-muted/50 data-[state=active]:bg-muted"
            >
              API Keys
            </TabsTrigger>
            <TabsTrigger
              value="plugins"
              className="w-full justify-start px-3 py-2 hover:bg-muted/50 data-[state=active]:bg-muted"
            >
              Plugins
            </TabsTrigger>
          </TabsList>
          <div className="w-px bg-border self-stretch" />
          <div className="flex-1 min-w-0 p-4 pt-10 h-[40vh] overflow-y-auto [&::-webkit-scrollbar]:w-2 [&::-webkit-scrollbar-track]:bg-transparent [&::-webkit-scrollbar-thumb]:bg-gray-300 [&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-thumb]:transition-colors [&::-webkit-scrollbar-thumb:hover]:bg-gray-400">
            <TabsContent value="general" className="mt-0">
              <GeneralTab
                version={version}
                gitCommit={gitCommit}
                modelsDirectory={modelsDirectory}
                logsDirectory={logsDirectory}
                onModelsDirectoryChange={handleModelsDirectoryChange}
                onLogsDirectoryChange={handleLogsDirectoryChange}
                onReportBug={() => setReportBugOpen(true)}
              />
            </TabsContent>
            <TabsContent value="account" className="mt-0">
              <AccountTab
                onPipelinesRefresh={onPipelinesRefresh ?? refetchPipelines}
                cloudDisabled={cloudDisabled}
              />
            </TabsContent>
            <TabsContent value="api-keys" className="mt-0">
              <ApiKeysTab isActive={open && activeTab === "api-keys"} />
            </TabsContent>
            <TabsContent value="plugins" className="mt-0">
              <PluginsTab
                plugins={pluginOps.plugins}
                installPath={pluginOps.pluginInstallPath}
                onInstallPathChange={pluginOps.setPluginInstallPath}
                onBrowse={pluginOps.handleBrowseLocalPlugin}
                onInstall={pluginOps.handleInstallPlugin}
                onUpdate={pluginOps.handleUpdatePlugin}
                onDelete={pluginOps.handleDeletePlugin}
                onReload={pluginOps.handleReloadPlugin}
                isLoading={pluginOps.isLoadingPlugins}
                isInstalling={pluginOps.isInstalling}
              />
            </TabsContent>
          </div>
        </Tabs>
      </DialogContent>

      <ReportBugDialog
        open={reportBugOpen}
        onClose={() => setReportBugOpen(false)}
      />
    </Dialog>
  );
}
