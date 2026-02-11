export const queryKeys = {
  pipelineSchemas: (isCloud: boolean) => ["pipelineSchemas", isCloud] as const,
  pipelineStatus: () => ["pipelineStatus"] as const,
  hardwareInfo: () => ["hardwareInfo"] as const,
  cloudStatus: () => ["cloudStatus"] as const,
  modelStatus: (id: string) => ["modelStatus", id] as const,
  assets: (type?: string) => ["assets", type] as const,
  loraFiles: () => ["loraFiles"] as const,
  plugins: () => ["plugins"] as const,
  apiKeys: () => ["apiKeys"] as const,
};
