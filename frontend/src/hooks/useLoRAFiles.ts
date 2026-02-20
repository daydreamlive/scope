import { useState, useCallback, useRef, useEffect } from "react";
import { useApi } from "./useApi";
import { useCloudStatus } from "./useCloudStatus";
import type { LoRAFileInfo } from "@/lib/api";

export interface UseLoRAFilesReturn {
  loraFiles: LoRAFileInfo[];
  isLoading: boolean;
  refresh: () => Promise<void>;
}

export function useLoRAFiles(): UseLoRAFilesReturn {
  const { listLoRAFiles } = useApi();
  const { isConnected: isCloudConnected } = useCloudStatus();
  const [loraFiles, setLoraFiles] = useState<LoRAFileInfo[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const prevCloudConnectedRef = useRef<boolean | null>(null);

  const refresh = useCallback(async () => {
    setIsLoading(true);
    try {
      const response = await listLoRAFiles();
      setLoraFiles(response.lora_files);
    } catch (error) {
      console.error("Failed to load LoRA files:", error);
    } finally {
      setIsLoading(false);
    }
  }, [listLoRAFiles]);

  // Initial load
  useEffect(() => {
    refresh();
  }, [refresh]);

  // Refresh when cloud connection state changes
  useEffect(() => {
    if (prevCloudConnectedRef.current === null) {
      prevCloudConnectedRef.current = isCloudConnected;
      return;
    }

    if (prevCloudConnectedRef.current !== isCloudConnected) {
      refresh();
    }

    prevCloudConnectedRef.current = isCloudConnected;
  }, [isCloudConnected, refresh]);

  return {
    loraFiles,
    isLoading,
    refresh,
  };
}
