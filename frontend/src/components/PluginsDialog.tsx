import { useState, useEffect, useCallback, useRef } from "react";
import { Dialog, DialogContent } from "./ui/dialog";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./ui/tabs";
import { PluginsTab } from "./settings/PluginsTab";
import { DiscoverTab } from "./settings/DiscoverTab";
import { usePipelinesContext } from "@/contexts/PipelinesContext";
import type { InstalledPlugin } from "@/types/settings";
import {
  listPlugins,
  installPlugin,
  uninstallPlugin,
  restartServer,
  waitForServer,
  type FailedPluginInfo,
} from "@/lib/api";
import { toast } from "sonner";

interface PluginsDialogProps {
  open: boolean;
  onClose: () => void;
  initialPluginPath?: string;
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
}: PluginsDialogProps) {
  const { refetch: refetchPipelines } = usePipelinesContext();
  const [pluginInstallPath, setPluginInstallPath] = useState(initialPluginPath);
  const [plugins, setPlugins] = useState<InstalledPlugin[]>([]);
  const [failedPlugins, setFailedPlugins] = useState<FailedPluginInfo[]>([]);
  const [isLoadingPlugins, setIsLoadingPlugins] = useState(false);
  const [isInstalling, setIsInstalling] = useState(false);
  const [activeTab, setActiveTab] = useState("installed");
  const isModifyingPluginsRef = useRef(false);

  useEffect(() => {
    if (open && initialPluginPath) {
      setPluginInstallPath(initialPluginPath);
    }
  }, [open, initialPluginPath]);

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
      if (!isModifyingPluginsRef.current) {
        toast.error("Failed to load plugins");
      }
    } finally {
      setIsLoadingPlugins(false);
    }
  }, []);

  useEffect(() => {
    if (open) {
      fetchPlugins();
    }
  }, [open, fetchPlugins]);

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

        const oldStartTime = await restartServer();
        await waitForServer(oldStartTime);
        toast.success("Server restarted");

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

        const oldStartTime = await restartServer();
        await waitForServer(oldStartTime);
        toast.success("Server restarted");

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

        setPlugins(prev => prev.filter(p => p.name !== pluginName));
        setFailedPlugins(prev =>
          prev.filter(fp => fp.package_name !== pluginName)
        );

        const oldStartTime = await restartServer();
        await waitForServer(oldStartTime);
        toast.success("Server restarted");

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
              value="discover"
              className="w-full justify-start px-3 py-2 hover:bg-muted/50 data-[state=active]:bg-muted"
            >
              Discover
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
              />
            </TabsContent>
            <TabsContent value="discover" className="mt-0">
              <DiscoverTab
                onInstall={handleInstallPlugin}
                installedRepoUrls={plugins
                  .map(p => p.package_spec)
                  .filter((s): s is string => !!s)}
                isInstalling={isInstalling}
              />
            </TabsContent>
          </div>
        </Tabs>
      </DialogContent>
    </Dialog>
  );
}
