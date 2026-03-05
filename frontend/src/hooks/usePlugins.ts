import { useState, useCallback, useEffect } from "react";
import { listPlugins } from "@/lib/api";
import type { PluginInfo, FailedPluginInfo } from "@/lib/api";

export interface UsePluginsReturn {
  plugins: PluginInfo[];
  failedPlugins: FailedPluginInfo[];
  isLoading: boolean;
  refresh: () => Promise<void>;
}

export function usePlugins(): UsePluginsReturn {
  const [plugins, setPlugins] = useState<PluginInfo[]>([]);
  const [failedPlugins, setFailedPlugins] = useState<FailedPluginInfo[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  const refresh = useCallback(async () => {
    setIsLoading(true);
    try {
      const response = await listPlugins();
      setPlugins(response.plugins);
      setFailedPlugins(response.failed_plugins ?? []);
    } catch (error) {
      console.error("Failed to load plugins:", error);
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Initial load
  useEffect(() => {
    refresh();
  }, [refresh]);

  return {
    plugins,
    failedPlugins,
    isLoading,
    refresh,
  };
}
