import { useQuery } from "@tanstack/react-query";
import { useApi } from "../useApi";
import { queryKeys } from "./queryKeys";
import type { HardwareInfoResponse } from "../../lib/api";

export function useHardwareInfo() {
  const { getHardwareInfo, isCloudMode, isReady } = useApi();

  return useQuery<HardwareInfoResponse>({
    queryKey: queryKeys.hardwareInfo(),
    queryFn: getHardwareInfo,
    staleTime: Infinity,
    enabled: !isCloudMode || isReady,
  });
}
