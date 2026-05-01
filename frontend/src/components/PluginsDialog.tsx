import { useState, useEffect, useRef, useMemo } from "react";
import { Dialog, DialogContent } from "./ui/dialog";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./ui/tabs";
import { PluginsTab } from "./settings/PluginsTab";
import { DiscoverTab } from "./settings/DiscoverTab";
import { WorkflowsTab } from "./settings/WorkflowsTab";
import { usePipelinesContext } from "@/contexts/PipelinesContext";
import { usePluginsContext } from "@/contexts/PluginsContext";
import type { InstalledPlugin } from "@/types/settings";
import {
  installPlugin,
  uninstallPlugin,
  restartServer,
  waitForServer,
} from "@/lib/api";
import { toast } from "sonner";

interface PluginsDialogProps {
  open: boolean;
  onClose: () => void;
  initialPluginPath?: string;
  /** Open directly to a specific tab (e.g. "workflows"). */
  initialTab?: string;
  disabled?: boolean;
  cloudConnected?: boolean;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  onLoadWorkflow?: (workflowData: Record<string, any>) => void;
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

export function PluginsDialog({
  open,
  onClose,
  initialPluginPath = "",
  initialTab,
  disabled = false,
  cloudConnected = false,
  onLoadWorkflow,
}: PluginsDialogProps) {
  const { refetch: refetchPipelines } = usePipelinesContext();
  const {
    plugins: pluginInfos,
    failedPlugins,
    isLoading: isLoadingPlugins,
    refresh: refreshPlugins,
  } = usePluginsContext();
  const [pluginInstallPath, setPluginInstallPath] = useState(initialPluginPath);
  const [isInstalling, setIsInstalling] = useState(false);
  const [activeTab, setActiveTab] = useState("installed");
  const isModifyingPluginsRef = useRef(false);

  const plugins: InstalledPlugin[] = useMemo(
    () =>
      pluginInfos.map(p => ({
        name: p.name,
        version: p.version,
        author: p.author,
        description: p.description,
        source: p.source,
        editable: p.editable,
        latest_version: p.latest_version,
        update_available: p.update_available,
        package_spec: p.package_spec,
        bundled: p.bundled,
        kind: p.kind,
        origin: p.origin,
      })),
    [pluginInfos]
  );

  useEffect(() => {
    if (open && initialPluginPath) {
      setPluginInstallPath(initialPluginPath);
    }
  }, [open, initialPluginPath]);

  useEffect(() => {
    if (open && initialTab) {
      setActiveTab(initialTab);
    }
  }, [open, initialTab]);

  useEffect(() => {
    if (open) {
      refreshPlugins();
    }
  }, [open, refreshPlugins]);

  const handleBrowseLocalPlugin = async () => {
    if (window.scope?.browseDirectory) {
      const path = await window.scope.browseDirectory(
        "Select Plugin Directory"
      );
      if (path) setPluginInstallPath(path);
    }
  };

  // Restart the server and refresh plugins+pipelines. Refresh runs even
  // if the restart wait times out (e.g. the cloud's restart took longer
  // than the local poll budget) — the install/uninstall/reload itself
  // already succeeded server-side, so the UI should pick up the new
  // state regardless. Returns a label for the toast.
  const restartAndRefresh = async (
    successLabel: string,
    fallbackLabel: string
  ): Promise<string> => {
    let label = successLabel;
    try {
      const oldStartTime = await restartServer();
      await waitForServer(oldStartTime);
    } catch (e) {
      console.warn("Server restart wait did not complete:", e);
      label = fallbackLabel;
    }
    await refreshPlugins();
    await refetchPipelines();
    return label;
  };

  const handleInstallPlugin = async (packageSpec: string) => {
    setIsInstalling(true);
    isModifyingPluginsRef.current = true;
    const toastId = toast.loading("Installing node...");
    try {
      const response = await installPlugin({
        package: packageSpec,
        editable: isLocalPath(packageSpec),
      });
      if (response.success) {
        const pluginName = response.plugin?.name || packageSpec;
        toast.loading(`Installed ${pluginName}. Restarting server...`, {
          id: toastId,
        });
        setPluginInstallPath("");

        const label = await restartAndRefresh(
          "Server restarted",
          `Installed ${pluginName}`
        );
        toast.success(label, { id: toastId });
      } else {
        toast.error(response.message, { id: toastId });
      }
    } catch (error) {
      console.error("Failed to install plugin:", error);
      toast.error(
        error instanceof Error ? error.message : "Failed to install node",
        { id: toastId }
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
    const toastId = toast.loading(`Updating ${pluginName}...`);
    try {
      const response = await installPlugin({
        package: packageSpec,
        upgrade: true,
      });
      if (response.success) {
        toast.loading(`Updated ${pluginName}. Restarting server...`, {
          id: toastId,
        });
        const label = await restartAndRefresh(
          "Server restarted",
          `Updated ${pluginName}`
        );
        toast.success(label, { id: toastId });
      } else {
        toast.error(response.message, { id: toastId });
      }
    } catch (error) {
      console.error("Failed to update plugin:", error);
      toast.error(
        error instanceof Error ? error.message : "Failed to update node",
        { id: toastId }
      );
    } finally {
      setIsInstalling(false);
      isModifyingPluginsRef.current = false;
    }
  };

  const handleDeletePlugin = async (pluginName: string) => {
    isModifyingPluginsRef.current = true;
    const toastId = toast.loading(`Uninstalling ${pluginName}...`);
    try {
      const response = await uninstallPlugin(pluginName);
      if (response.success) {
        toast.loading(`Uninstalled ${pluginName}. Restarting server...`, {
          id: toastId,
        });
        const label = await restartAndRefresh(
          "Server restarted",
          `Uninstalled ${pluginName}`
        );
        toast.success(label, { id: toastId });
      } else {
        toast.error(response.message, { id: toastId });
      }
    } catch (error) {
      console.error("Failed to uninstall plugin:", error);
      toast.error(
        error instanceof Error ? error.message : "Failed to uninstall node",
        { id: toastId }
      );
    } finally {
      isModifyingPluginsRef.current = false;
    }
  };

  const handleReloadPlugin = async (pluginName: string) => {
    isModifyingPluginsRef.current = true;
    try {
      toast.info(`Reloading ${pluginName}. Restarting server...`);
      const label = await restartAndRefresh(
        "Server restarted",
        `Reloaded ${pluginName}`
      );
      toast.success(label);
    } catch (error) {
      toast.error(
        error instanceof Error ? error.message : "Failed to reload node"
      );
    } finally {
      isModifyingPluginsRef.current = false;
    }
  };

  return (
    <Dialog open={open} onOpenChange={isOpen => !isOpen && onClose()}>
      <DialogContent className="sm:max-w-[600px] lg:max-w-[800px] xl:max-w-[960px] p-0 gap-0">
        <Tabs
          value={activeTab}
          onValueChange={setActiveTab}
          orientation="vertical"
          className="flex items-stretch"
        >
          <TabsList className="flex flex-col items-start justify-start bg-transparent gap-1 w-32 p-4">
            <TabsTrigger
              value="installed"
              className="w-full justify-start px-3 py-2 hover:bg-muted/50 data-[state=active]:bg-muted"
            >
              Installed
            </TabsTrigger>
            <TabsTrigger
              value="workflows"
              className="w-full justify-start px-3 py-2 hover:bg-muted/50 data-[state=active]:bg-muted"
            >
              Workflows
            </TabsTrigger>
            <TabsTrigger
              value="discover"
              className="w-full justify-start px-3 py-2 hover:bg-muted/50 data-[state=active]:bg-muted"
            >
              Nodes
            </TabsTrigger>
          </TabsList>
          <div className="w-px bg-border self-stretch" />
          <div className="flex-1 min-w-0 p-4 pt-10 h-[80vh] lg:h-[80vh] xl:h-[80vh] overflow-y-auto [&::-webkit-scrollbar]:w-2 [&::-webkit-scrollbar-track]:bg-transparent [&::-webkit-scrollbar-thumb]:bg-gray-300 [&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-thumb]:transition-colors [&::-webkit-scrollbar-thumb:hover]:bg-gray-400">
            <TabsContent value="installed" className="mt-0">
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
                disabled={disabled}
                hideInstall={cloudConnected}
              />
            </TabsContent>
            <TabsContent value="workflows" className="mt-0">
              <WorkflowsTab
                onLoad={data => {
                  onLoadWorkflow?.(data);
                  onClose();
                }}
              />
            </TabsContent>
            <TabsContent value="discover" className="mt-0">
              <DiscoverTab
                onInstall={handleInstallPlugin}
                installedRepoUrls={plugins
                  .map(p => p.package_spec)
                  .filter((s): s is string => !!s)}
                isInstalling={isInstalling}
                disabled={disabled}
                cloudConnected={cloudConnected}
              />
            </TabsContent>
          </div>
        </Tabs>
      </DialogContent>
    </Dialog>
  );
}
