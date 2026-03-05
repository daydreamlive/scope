import { useState, useCallback, useEffect } from "react";
import { getServerInfo } from "@/lib/api";

export interface UseServerInfoReturn {
  version: string | null;
  gitCommit: string | null;
  isLoading: boolean;
  refresh: () => Promise<void>;
}

export function useServerInfo(): UseServerInfoReturn {
  const [version, setVersion] = useState<string | null>(null);
  const [gitCommit, setGitCommit] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const refresh = useCallback(async () => {
    setIsLoading(true);
    try {
      const info = await getServerInfo();
      setVersion(info.version);
      setGitCommit(info.gitCommit);
    } catch (error) {
      console.error("Failed to fetch server info:", error);
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Initial load
  useEffect(() => {
    refresh();
  }, [refresh]);

  return {
    version,
    gitCommit,
    isLoading,
    refresh,
  };
}
