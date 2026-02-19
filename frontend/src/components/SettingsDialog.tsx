import { useState, useEffect, useCallback, useRef } from "react";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "./ui/dialog";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./ui/tabs";
import { AccountTab } from "./settings/AccountTab";
import { ApiKeysTab } from "./settings/ApiKeysTab";
import { GeneralTab } from "./settings/GeneralTab";
import { LoRAsTab } from "./settings/LoRAsTab";
import { PluginsTab } from "./settings/PluginsTab";
import { ReportBugDialog } from "./ReportBugDialog";
import { usePipelinesContext } from "@/contexts/PipelinesContext";
import { useLoRAsContext } from "@/contexts/LoRAsContext";
import type { InstalledPlugin } from "@/types/settings";
import {
  listPlugins,
  installPlugin,
  uninstallPlugin,
  restartServer,
  waitForServer,
  getServerInfo,
  installLoRAFile,
  type FailedPluginInfo,
} from "@/lib/api";
import { toast } from "sonner";

interface SettingsDialogProps {
  open: boolean;
  onClose: () => void;
  initialTab?: "general" | "account" | "api-keys" | "plugins" | "loras";
  initialPluginPath?: string;
  onPipelinesRefresh?: () => Promise<unknown>;
  cloudDisabled?: boolean;
}

const isLocalPath = (spec: string): boolean => {
  const s = spec.trim();
  return (
    s.startsWith("/") ||
    /^[A-Za-z]:[\\/]/.test(s) ||
    s.startsWith("./") ||
    s.startsWith(".\\") ||
    s.startsWith("../") ||
    s.startsWith("..\\") ||
    s.startsWith("~/")
  );
};

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
  const {
    loraFiles,
    isLoading: isLoadingLoRAs,
    refresh: refreshLoRAs,
  } = useLoRAsContext();
  const [modelsDirectory, setModelsDirectory] = useState(
    "~/.daydream-scope/models"
  );
  const [logsDirectory, setLogsDirectory] = useState("~/.daydream-scope/logs");
  const [reportBugOpen, setReportBugOpen] = useState(false);
  const [pluginInstallPath, setPluginInstallPath] = useState(initialPluginPath);
  const [plugins, setPlugins] = useState<InstalledPlugin[]>([]);
  const [failedPlugins, setFailedPlugins] = useState<FailedPluginInfo[]>([]);
  const [isLoadingPlugins, setIsLoadingPlugins] = useState(false);
  const [isInstalling, setIsInstalling] = useState(false);
  const [activeTab, setActiveTab] = useState<string>(initialTab);
  // LoRA install state (files come from context)
  const [loraInstallUrl, setLoraInstallUrl] = useState("");
  const [isInstallingLoRA, setIsInstallingLoRA] = useState(false);
  const [version, setVersion] = useState<string>("");
  const [gitCommit, setGitCommit] = useState<string>("");
  // Track install/update/uninstall operations to suppress spurious error toasts
  const isModifyingPluginsRef = useRef(false);

  // Sync state when dialog opens with initial values from props
  useEffect(() => {
    if (open) {
      setActiveTab(initialTab);
      if (initialPluginPath) {
        setPluginInstallPath(initialPluginPath);
      }
    }
  }, [open, initialTab, initialPluginPath]);

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

  const fetchPlugins = useCallback(async () => {
    setIsLoadingPlugins(true);
    try {
      const response = await listPlugins();
      setPlugins(
        response.plugins.map(p => ({
          name: p.name,
          version: p.version,
          author: p.author,
          description: p.description,
          source: p.source,
          editable: p.editable,
          latest_version: p.latest_version,
          update_available: p.update_available,
          package_spec: p.package_spec,
        }))
      );
      setFailedPlugins(response.failed_plugins ?? []);
    } catch (error) {
      console.error("Failed to fetch plugins:", error);
      // Don't show error toast during install/update/uninstall operations
      // as the server may be busy or restarting
      if (!isModifyingPluginsRef.current) {
        toast.error("Failed to load plugins");
      }
    } finally {
      setIsLoadingPlugins(false);
    }
  }, []);

  // Fetch plugins when dialog opens or when switching to plugins tab
  useEffect(() => {
    if (open && activeTab === "plugins") {
      fetchPlugins();
    }
  }, [open, activeTab, fetchPlugins]);

  // Refresh LoRAs when switching to LoRAs tab
  useEffect(() => {
    if (open && activeTab === "loras") {
      refreshLoRAs();
    }
  }, [open, activeTab, refreshLoRAs]);

  const handleModelsDirectoryChange = (value: string) => {
    console.log("Models directory changed:", value);
    setModelsDirectory(value);
  };

  const handleLogsDirectoryChange = (value: string) => {
    console.log("Logs directory changed:", value);
    setLogsDirectory(value);
  };

  const handleBrowseLocalPlugin = async () => {
    if (window.scope?.browseDirectory) {
      const path = await window.scope.browseDirectory(
        "Select Plugin Directory"
      );
      if (path) setPluginInstallPath(path);
    }
  };

  const handleInstallPlugin = async (packageSpec: string) => {
    setIsInstalling(true);
    isModifyingPluginsRef.current = true;
    try {
      const response = await installPlugin({
        package: packageSpec,
        editable: isLocalPath(packageSpec),
      });
      if (response.success) {
        const pluginName = response.plugin?.name || packageSpec;
        toast.success(`Installed ${pluginName}. Restarting server...`);
        setPluginInstallPath("");

        // Optimistically add plugin to local state
        if (response.plugin) {
          setPlugins(prev => [
            ...prev,
            {
              name: response.plugin!.name,
              version: response.plugin!.version,
              author: response.plugin!.author,
              description: response.plugin!.description,
              source: response.plugin!.source,
              editable: response.plugin!.editable,
              latest_version: response.plugin!.latest_version,
              update_available: response.plugin!.update_available,
              package_spec: response.plugin!.package_spec,
            },
          ]);
        }

        // Trigger restart and wait for server to come back
        const oldStartTime = await restartServer();
        await waitForServer(oldStartTime);
        toast.success("Server restarted");

        // Sync with server after restart
        await fetchPlugins();
        await refetchPipelines();
      } else {
        toast.error(response.message);
      }
    } catch (error) {
      console.error("Failed to install plugin:", error);
      toast.error(
        error instanceof Error ? error.message : "Failed to install plugin"
      );
    } finally {
      setIsInstalling(false);
      isModifyingPluginsRef.current = false;
    }
  };

  const handleUpdatePlugin = async (
    pluginName: string,
    packageSpec: string
  ) => {
    setIsInstalling(true);
    isModifyingPluginsRef.current = true;
    try {
      const response = await installPlugin({
        package: packageSpec,
        upgrade: true,
      });
      if (response.success) {
        toast.success(`Updated ${pluginName}. Restarting server...`);

        // Optimistically update plugin in local state
        if (response.plugin) {
          setPlugins(prev =>
            prev.map(p =>
              p.name === pluginName
                ? {
                    name: response.plugin!.name,
                    version: response.plugin!.version,
                    author: response.plugin!.author,
                    description: response.plugin!.description,
                    source: response.plugin!.source,
                    editable: response.plugin!.editable,
                    latest_version: response.plugin!.latest_version,
                    update_available: response.plugin!.update_available,
                    package_spec: response.plugin!.package_spec,
                  }
                : p
            )
          );
        }

        // Trigger restart and wait for server to come back
        const oldStartTime = await restartServer();
        await waitForServer(oldStartTime);
        toast.success("Server restarted");

        // Sync with server after restart
        await fetchPlugins();
        await refetchPipelines();
      } else {
        toast.error(response.message);
      }
    } catch (error) {
      console.error("Failed to update plugin:", error);
      toast.error(
        error instanceof Error ? error.message : "Failed to update plugin"
      );
    } finally {
      setIsInstalling(false);
      isModifyingPluginsRef.current = false;
    }
  };

  const handleDeletePlugin = async (pluginName: string) => {
    isModifyingPluginsRef.current = true;
    try {
      const response = await uninstallPlugin(pluginName);
      if (response.success) {
        toast.success(`Uninstalled ${pluginName}. Restarting server...`);

        // Optimistically remove plugin from local state
        setPlugins(prev => prev.filter(p => p.name !== pluginName));
        setFailedPlugins(prev =>
          prev.filter(fp => fp.package_name !== pluginName)
        );

        // Trigger restart and wait for server to come back
        const oldStartTime = await restartServer();
        await waitForServer(oldStartTime);
        toast.success("Server restarted");

        // Sync with server after restart
        await fetchPlugins();
        await refetchPipelines();
      } else {
        toast.error(response.message);
      }
    } catch (error) {
      console.error("Failed to uninstall plugin:", error);
      toast.error(
        error instanceof Error ? error.message : "Failed to uninstall plugin"
      );
    } finally {
      isModifyingPluginsRef.current = false;
    }
  };

  const handleReloadPlugin = async (pluginName: string) => {
    isModifyingPluginsRef.current = true;
    try {
      toast.info(`Reloading ${pluginName}. Restarting server...`);
      const oldStartTime = await restartServer();
      await waitForServer(oldStartTime);
      toast.success("Server restarted");
      await fetchPlugins();
      await refetchPipelines();
    } catch (error) {
      toast.error(
        error instanceof Error ? error.message : "Failed to reload plugin"
      );
    } finally {
      isModifyingPluginsRef.current = false;
    }
  };

  const handleInstallLoRA = async (url: string) => {
    setIsInstallingLoRA(true);
    const filename = url.split("/").pop()?.split("?")[0] || "LoRA file";
    try {
      const installPromise = installLoRAFile({ url });
      toast.promise(installPromise, {
        loading: `Installing ${filename}...`,
        success: response => response.message,
        error: err => err.message || "Install failed",
      });
      await installPromise;
      setLoraInstallUrl("");
      await refreshLoRAs();
    } catch (error) {
      console.error("Failed to install LoRA:", error);
    } finally {
      setIsInstallingLoRA(false);
    }
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
            <TabsTrigger
              value="loras"
              className="w-full justify-start px-3 py-2 hover:bg-muted/50 data-[state=active]:bg-muted"
            >
              LoRAs
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
                plugins={plugins}
                failedPlugins={failedPlugins}
                installPath={pluginInstallPath}
                onInstallPathChange={setPluginInstallPath}
                onBrowse={handleBrowseLocalPlugin}
                onInstall={handleInstallPlugin}
                onUpdate={handleUpdatePlugin}
                onDelete={handleDeletePlugin}
                onReload={handleReloadPlugin}
                isLoading={isLoadingPlugins}
                isInstalling={isInstalling}
              />
            </TabsContent>
            <TabsContent value="loras" className="mt-0">
              <LoRAsTab
                loraFiles={loraFiles}
                installUrl={loraInstallUrl}
                onInstallUrlChange={setLoraInstallUrl}
                onInstall={handleInstallLoRA}
                onRefresh={refreshLoRAs}
                isLoading={isLoadingLoRAs}
                isInstalling={isInstallingLoRA}
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
