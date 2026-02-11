import { useQuery } from "@tanstack/react-query";
import { useApi } from "../useApi";
import { queryKeys } from "./queryKeys";
import type { PipelineSchemasResponse } from "../../lib/api";

export function usePipelineSchemas() {
  const { getPipelineSchemas, isCloudMode, isReady } = useApi();

  return useQuery<PipelineSchemasResponse>({
    queryKey: queryKeys.pipelineSchemas(isCloudMode),
    queryFn: getPipelineSchemas,
    staleTime: Infinity,
    enabled: !isCloudMode || isReady,
  });
}
