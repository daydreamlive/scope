import { useState, useCallback, useRef } from "react";
import type { InstalledPlugin } from "../types/settings";
import type { PluginInfo } from "../lib/api";
import {
  listPlugins,
  installPlugin,
  uninstallPlugin,
  restartServer,
  waitForServer,
} from "../lib/api";
import { toast } from "sonner";

const toInstalledPlugin = (p: PluginInfo): InstalledPlugin => ({
  name: p.name,
  version: p.version,
  author: p.author,
  description: p.description,
  source: p.source,
  editable: p.editable,
  latest_version: p.latest_version,
  update_available: p.update_available,
  package_spec: p.package_spec,
});

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

export function usePluginOperations(
  refetchPipelines: () => Promise<unknown>
) {
  const [plugins, setPlugins] = useState<InstalledPlugin[]>([]);
  const [isLoadingPlugins, setIsLoadingPlugins] = useState(false);
  const [isInstalling, setIsInstalling] = useState(false);
  const [pluginInstallPath, setPluginInstallPath] = useState("");
  const isModifyingRef = useRef(false);

  const fetchPlugins = useCallback(async () => {
    setIsLoadingPlugins(true);
    try {
      const response = await listPlugins();
      setPlugins(response.plugins.map(toInstalledPlugin));
    } catch (error) {
      console.error("Failed to fetch plugins:", error);
      if (!isModifyingRef.current) {
        toast.error("Failed to load plugins");
      }
    } finally {
      setIsLoadingPlugins(false);
    }
  }, []);

  const restartAndSync = async () => {
    const oldStartTime = await restartServer();
    await waitForServer(oldStartTime);
    toast.success("Server restarted");
    await fetchPlugins();
    await refetchPipelines();
  };

  const handleInstallPlugin = async (packageSpec: string) => {
    setIsInstalling(true);
    isModifyingRef.current = true;
    try {
      const response = await installPlugin({
        package: packageSpec,
        editable: isLocalPath(packageSpec),
      });
      if (response.success) {
        toast.success(
          `Installed ${response.plugin?.name || packageSpec}. Restarting server...`
        );
        setPluginInstallPath("");
        if (response.plugin) {
          setPlugins(prev => [...prev, toInstalledPlugin(response.plugin!)]);
        }
        await restartAndSync();
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
      isModifyingRef.current = false;
    }
  };

  const handleUpdatePlugin = async (
    pluginName: string,
    packageSpec: string
  ) => {
    setIsInstalling(true);
    isModifyingRef.current = true;
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
              p.name === pluginName ? toInstalledPlugin(response.plugin!) : p
            )
          );
        }
        await restartAndSync();
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
      isModifyingRef.current = false;
    }
  };

  const handleDeletePlugin = async (pluginName: string) => {
    isModifyingRef.current = true;
    try {
      const response = await uninstallPlugin(pluginName);
      if (response.success) {
        toast.success(`Uninstalled ${pluginName}. Restarting server...`);
        setPlugins(prev => prev.filter(p => p.name !== pluginName));
        await restartAndSync();
      } else {
        toast.error(response.message);
      }
    } catch (error) {
      console.error("Failed to uninstall plugin:", error);
      toast.error(
        error instanceof Error ? error.message : "Failed to uninstall plugin"
      );
    } finally {
      isModifyingRef.current = false;
    }
  };

  const handleReloadPlugin = async (pluginName: string) => {
    isModifyingRef.current = true;
    try {
      toast.info(`Reloading ${pluginName}. Restarting server...`);
      await restartAndSync();
    } catch (error) {
      toast.error(
        error instanceof Error ? error.message : "Failed to reload plugin"
      );
    } finally {
      isModifyingRef.current = false;
    }
  };

  const handleBrowseLocalPlugin = async () => {
    if (window.scope?.browseDirectory) {
      const path = await window.scope.browseDirectory(
        "Select Plugin Directory"
      );
      if (path) setPluginInstallPath(path);
    }
  };

  return {
    plugins,
    isLoadingPlugins,
    isInstalling,
    pluginInstallPath,
    setPluginInstallPath,
    fetchPlugins,
    handleInstallPlugin,
    handleUpdatePlugin,
    handleDeletePlugin,
    handleReloadPlugin,
    handleBrowseLocalPlugin,
  };
}
