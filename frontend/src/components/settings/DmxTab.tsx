import { useState, useEffect, useCallback } from "react";
import {
  BookOpenText,
  Loader2,
  Plus,
  Trash2,
  Play,
  AlertCircle,
} from "lucide-react";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import { Switch } from "../ui/switch";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
  SelectGroup,
  SelectLabel,
} from "../ui/select";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "../ui/dialog";

import { openExternalUrl } from "@/lib/openExternal";
import { toast } from "sonner";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../ui/tabs";

interface DmxStatus {
  enabled: boolean;
  listening: boolean;
  port: number | null;
  input_active: boolean;
  input_mapping_count: number;
  output_enabled: boolean;
  output_mapping_count: number;
  output_merge_mode: string;
}

interface DmxConfig {
  input_mappings: DmxInputMapping[];
  output_mappings: DmxOutputMapping[];
  input_universe: number;
  input_start_channel: number;
  output_universe: number;
  output_enabled: boolean;
  output_merge_mode: string;
}

interface DmxInputMapping {
  id: string;
  universe: number;
  channel: number;
  param_key: string;
  category: string;
  min_value: number;
  max_value: number;
  enabled: boolean;
}

interface DmxOutputMapping {
  id: string;
  universe: number;
  channel: number;
  source_key: string;
  category: string;
  min_value: number;
  max_value: number;
  enabled: boolean;
}

interface ParameterInfo {
  key: string;
  type: string;
  description: string;
  min?: number;
  max?: number;
}

interface AnalysisSource {
  key: string;
  category: string;
  description: string;
  min: number;
  max: number;
}

interface DmxTabProps {
  isActive: boolean;
}

export function DmxTab({ isActive }: DmxTabProps) {
  const [status, setStatus] = useState<DmxStatus | null>(null);
  const [config, setConfig] = useState<DmxConfig | null>(null);
  const [parameters, setParameters] = useState<Record<string, ParameterInfo[]>>(
    {}
  );
  const [analysisSources, setAnalysisSources] = useState<AnalysisSource[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [direction, setDirection] = useState<"in" | "out">("in");

  // Add mapping dialog state
  const [isAddDialogOpen, setIsAddDialogOpen] = useState(false);
  const [newInputMapping, setNewInputMapping] = useState({
    universe: 0,
    channel: 1,
    param_key: "",
    category: "generation",
    min_value: 0,
    max_value: 1,
  });
  const [newOutputMapping, setNewOutputMapping] = useState({
    universe: 0,
    channel: 1,
    source_key: "",
    category: "analysis",
    min_value: 0,
    max_value: 1,
  });

  const fetchStatus = useCallback(async () => {
    try {
      const res = await fetch("/api/v1/dmx/status");
      if (res.ok) {
        setStatus(await res.json());
      }
    } catch (err) {
      console.error("Failed to fetch DMX status:", err);
    }
  }, []);

  const fetchConfig = useCallback(async () => {
    try {
      const res = await fetch("/api/v1/dmx/config");
      if (res.ok) {
        setConfig(await res.json());
      }
    } catch (err) {
      console.error("Failed to fetch DMX config:", err);
    }
  }, []);

  const fetchParameters = useCallback(async () => {
    try {
      const res = await fetch("/api/v1/dmx/parameters");
      if (res.ok) {
        setParameters(await res.json());
      }
    } catch (err) {
      console.error("Failed to fetch DMX parameters:", err);
    }
  }, []);

  const fetchAnalysisSources = useCallback(async () => {
    try {
      const res = await fetch("/api/v1/dmx/analysis-sources");
      if (res.ok) {
        const data = await res.json();
        setAnalysisSources(data.sources || []);
      }
    } catch (err) {
      console.error("Failed to fetch analysis sources:", err);
    }
  }, []);

  useEffect(() => {
    if (isActive) {
      setIsLoading(true);
      Promise.all([
        fetchStatus(),
        fetchConfig(),
        fetchParameters(),
        fetchAnalysisSources(),
      ]).finally(() => setIsLoading(false));
    }
  }, [
    isActive,
    fetchStatus,
    fetchConfig,
    fetchParameters,
    fetchAnalysisSources,
  ]);

  const handleUpdateConfig = async (updates: Partial<DmxConfig>) => {
    try {
      const res = await fetch("/api/v1/dmx/config", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(updates),
      });
      if (res.ok) {
        const updatedConfig = await res.json();
        setConfig(updatedConfig);
        fetchStatus();
      }
    } catch (err) {
      toast.error("Failed to update DMX config");
      console.error(err);
    }
  };

  const handleAddInputMapping = async () => {
    if (!newInputMapping.param_key) {
      toast.error("Please select a parameter");
      return;
    }

    try {
      const res = await fetch("/api/v1/dmx/input-mappings", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ...newInputMapping,
          id: `in-${newInputMapping.universe}-${newInputMapping.channel}-${Date.now()}`,
          enabled: true,
        }),
      });

      if (res.ok) {
        toast.success("Input mapping added");
        setIsAddDialogOpen(false);
        setNewInputMapping({
          universe: config?.input_universe ?? 0,
          channel: 1,
          param_key: "",
          category: "generation",
          min_value: 0,
          max_value: 1,
        });
        fetchConfig();
        fetchStatus();
      } else {
        const err = await res.json();
        toast.error(err.detail || "Failed to add mapping");
      }
    } catch (err) {
      toast.error("Failed to add mapping");
      console.error(err);
    }
  };

  const handleAddOutputMapping = async () => {
    if (!newOutputMapping.source_key) {
      toast.error("Please select a source");
      return;
    }

    try {
      const res = await fetch("/api/v1/dmx/output-mappings", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ...newOutputMapping,
          id: `out-${newOutputMapping.universe}-${newOutputMapping.channel}-${Date.now()}`,
          enabled: true,
        }),
      });

      if (res.ok) {
        toast.success("Output mapping added");
        setIsAddDialogOpen(false);
        setNewOutputMapping({
          universe: config?.output_universe ?? 0,
          channel: 1,
          source_key: "",
          category: "analysis",
          min_value: 0,
          max_value: 1,
        });
        fetchConfig();
        fetchStatus();
      } else {
        const err = await res.json();
        toast.error(err.detail || "Failed to add mapping");
      }
    } catch (err) {
      toast.error("Failed to add mapping");
      console.error(err);
    }
  };

  const handleDeleteInputMapping = async (id: string) => {
    try {
      const res = await fetch(
        `/api/v1/dmx/input-mappings/${encodeURIComponent(id)}`,
        { method: "DELETE" }
      );
      if (res.ok) {
        toast.success("Mapping removed");
        fetchConfig();
        fetchStatus();
      }
    } catch (err) {
      toast.error("Failed to remove mapping");
      console.error(err);
    }
  };

  const handleDeleteOutputMapping = async (id: string) => {
    try {
      const res = await fetch(
        `/api/v1/dmx/output-mappings/${encodeURIComponent(id)}`,
        { method: "DELETE" }
      );
      if (res.ok) {
        toast.success("Mapping removed");
        fetchConfig();
        fetchStatus();
      }
    } catch (err) {
      toast.error("Failed to remove mapping");
      console.error(err);
    }
  };

  const handleTestOutput = async () => {
    try {
      const res = await fetch("/api/v1/dmx/test-output", { method: "POST" });
      if (res.ok) {
        toast.success("Test ramp started (2 seconds)");
      } else {
        const err = await res.json();
        toast.error(err.detail || "Failed to start test");
      }
    } catch (err) {
      toast.error("Failed to start test");
      console.error(err);
    }
  };

  const handleOpenDocs = () => {
    openExternalUrl("https://docs.daydream.live/scope/guides/dmx");
  };

  // Group input mappings by category
  const groupedInputMappings = config?.input_mappings.reduce(
    (acc, mapping) => {
      const cat = mapping.category || "generation";
      if (!acc[cat]) acc[cat] = [];
      acc[cat].push(mapping);
      return acc;
    },
    {} as Record<string, DmxInputMapping[]>
  );

  // Group output mappings by category
  const groupedOutputMappings = config?.output_mappings.reduce(
    (acc, mapping) => {
      const cat = mapping.category || "analysis";
      if (!acc[cat]) acc[cat] = [];
      acc[cat].push(mapping);
      return acc;
    },
    {} as Record<string, DmxOutputMapping[]>
  );

  const categoryLabels: Record<string, string> = {
    generation: "Generation",
    lora: "LoRA",
    color: "Color",
    analysis: "Analysis",
  };

  return (
    <div className="space-y-4">
      {/* Status Section */}
      <div className="rounded-lg bg-muted/50 p-4 space-y-3">
        <div className="flex items-center gap-4">
          <span className="text-sm font-medium text-foreground w-32">
            Status
          </span>
          <div className="flex-1 flex items-center justify-end gap-2">
            {isLoading ? (
              <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
            ) : status?.listening ? (
              <>
                <span
                  className={`h-2 w-2 rounded-full ${status.input_active ? "bg-green-500 animate-pulse" : "bg-yellow-500"}`}
                />
                <span className="text-sm text-green-500">
                  Listening on UDP port {status.port}
                </span>
              </>
            ) : (
              <span className="text-sm text-muted-foreground">
                Not listening
              </span>
            )}
          </div>
        </div>

        {!status?.input_active && status?.listening && (
          <div className="flex items-start gap-2 p-2 rounded-md bg-yellow-500/10 border border-yellow-500/20">
            <AlertCircle className="h-4 w-4 mt-0.5 text-yellow-600" />
            <span className="text-xs text-yellow-600">
              No Art-Net signal on Universe {config?.input_universe ?? 0}
            </span>
          </div>
        )}

        <div className="flex items-center gap-4">
          <span className="text-sm font-medium text-foreground w-32">
            Reference
          </span>
          <div className="flex-1 flex items-center justify-end">
            <Button
              variant="outline"
              size="sm"
              onClick={handleOpenDocs}
              className="gap-1.5"
            >
              <BookOpenText className="h-4 w-4" />
              Open DMX Docs
            </Button>
          </div>
        </div>

        <div className="text-xs text-muted-foreground leading-relaxed">
          Send Art-Net DMX data to UDP port{" "}
          <code className="bg-muted px-1 py-0.5 rounded text-xs">
            {status?.port ?? 6454}
          </code>{" "}
          to control pipeline parameters. Configure channel mappings below.
        </div>
      </div>

      {/* Direction Tabs */}
      <Tabs
        value={direction}
        onValueChange={v => setDirection(v as "in" | "out")}
      >
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="in">
            DMX In
            {(status?.input_mapping_count ?? 0) > 0 && (
              <span className="ml-1.5 text-xs text-muted-foreground">
                ({status?.input_mapping_count})
              </span>
            )}
          </TabsTrigger>
          <TabsTrigger value="out">
            DMX Out
            {(status?.output_mapping_count ?? 0) > 0 && (
              <span className="ml-1.5 text-xs text-muted-foreground">
                ({status?.output_mapping_count})
              </span>
            )}
          </TabsTrigger>
        </TabsList>

        {/* DMX In Tab */}
        <TabsContent value="in" className="space-y-4 mt-4">
          {/* Input Settings */}
          <div className="rounded-lg border p-3 space-y-3">
            <div className="flex items-center gap-4">
              <span className="text-sm font-medium w-28">Transport</span>
              <Select value="artnet" disabled>
                <SelectTrigger className="flex-1 h-8">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="artnet">Art-Net</SelectItem>
                  <SelectItem value="sacn" disabled>
                    sACN (Coming soon)
                  </SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="flex items-center gap-4">
              <span className="text-sm font-medium w-28">Universe</span>
              <Input
                type="number"
                min={0}
                max={32767}
                value={config?.input_universe ?? 0}
                onChange={e =>
                  handleUpdateConfig({
                    input_universe: parseInt(e.target.value) || 0,
                  })
                }
                className="flex-1 h-8"
              />
            </div>

            <div className="flex items-center gap-4">
              <span className="text-sm font-medium w-28">Start Channel</span>
              <Input
                type="number"
                min={1}
                max={512}
                value={config?.input_start_channel ?? 1}
                onChange={e =>
                  handleUpdateConfig({
                    input_start_channel: parseInt(e.target.value) || 1,
                  })
                }
                className="flex-1 h-8"
              />
            </div>
          </div>

          {/* Input Mappings */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-medium">Channel Mappings</h3>
              <Button
                variant="outline"
                size="sm"
                onClick={() => {
                  setNewInputMapping(m => ({
                    ...m,
                    universe: config?.input_universe ?? 0,
                  }));
                  setIsAddDialogOpen(true);
                }}
                className="gap-1.5"
              >
                <Plus className="h-4 w-4" />
                Add Mapping
              </Button>
            </div>

            {(!config?.input_mappings ||
              config.input_mappings.length === 0) && (
              <div className="text-sm text-muted-foreground text-center py-6 border border-dashed rounded-lg">
                Add a mapping to connect DMX channels to Scope parameters.
              </div>
            )}

            {/* Grouped mappings */}
            {groupedInputMappings &&
              Object.entries(groupedInputMappings).map(
                ([category, mappings]) => (
                  <div key={category} className="space-y-2">
                    <div className="flex items-center gap-2 p-2">
                      <span className="text-sm font-medium">
                        {categoryLabels[category] || category}
                      </span>
                      <span className="text-xs text-muted-foreground">
                        ({mappings.length})
                      </span>
                    </div>
                    <div className="space-y-2">
                      {mappings.map(mapping => (
                        <div
                          key={mapping.id}
                          className="flex items-center gap-3 p-3 rounded-lg bg-muted/30 border"
                        >
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2">
                              <code className="text-xs bg-muted px-1.5 py-0.5 rounded">
                                U{mapping.universe} Ch{mapping.channel}
                              </code>
                              <span className="text-sm">→</span>
                              <code className="text-xs bg-accent/20 text-accent-foreground px-1.5 py-0.5 rounded truncate">
                                {mapping.param_key}
                              </code>
                            </div>
                            <p className="text-xs text-muted-foreground mt-1">
                              Range: {mapping.min_value} → {mapping.max_value}
                            </p>
                          </div>
                          <Button
                            variant="ghost"
                            size="icon"
                            className="h-8 w-8 text-muted-foreground hover:text-destructive"
                            onClick={() => handleDeleteInputMapping(mapping.id)}
                          >
                            <Trash2 className="h-4 w-4" />
                          </Button>
                        </div>
                      ))}
                    </div>
                  </div>
                )
              )}
          </div>
        </TabsContent>

        {/* DMX Out Tab */}
        <TabsContent value="out" className="space-y-4 mt-4">
          {/* Output Settings */}
          <div className="rounded-lg border p-3 space-y-3">
            <div className="flex items-center gap-4">
              <span className="text-sm font-medium w-28">Output Enable</span>
              <div className="flex-1 flex items-center justify-end gap-2">
                <span className="text-xs text-muted-foreground">
                  {config?.output_enabled ? "Sending" : "Disabled"}
                </span>
                <Switch
                  checked={config?.output_enabled ?? false}
                  onCheckedChange={checked =>
                    handleUpdateConfig({ output_enabled: checked })
                  }
                  className="data-[state=unchecked]:bg-zinc-600 data-[state=checked]:bg-green-500"
                />
              </div>
            </div>

            {!config?.output_enabled && (
              <div className="text-xs text-muted-foreground p-2 bg-muted/50 rounded">
                DMX output is disabled. Enable to send values to fixtures.
              </div>
            )}

            <div className="flex items-center gap-4">
              <span className="text-sm font-medium w-28">Universe</span>
              <Input
                type="number"
                min={0}
                max={32767}
                value={config?.output_universe ?? 0}
                onChange={e =>
                  handleUpdateConfig({
                    output_universe: parseInt(e.target.value) || 0,
                  })
                }
                className="flex-1 h-8"
              />
            </div>

            <div className="flex items-center gap-4">
              <span className="text-sm font-medium w-28">Merge Mode</span>
              <Select
                value={config?.output_merge_mode ?? "htp"}
                onValueChange={value =>
                  handleUpdateConfig({ output_merge_mode: value })
                }
              >
                <SelectTrigger className="flex-1 h-8">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="htp">
                    HTP (Highest Takes Precedence)
                  </SelectItem>
                  <SelectItem value="ltp">
                    LTP (Latest Takes Precedence)
                  </SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="flex items-center gap-4">
              <span className="text-sm font-medium w-28">Test</span>
              <Button
                variant="outline"
                size="sm"
                onClick={handleTestOutput}
                disabled={
                  !config?.output_enabled ||
                  !config?.output_mappings?.length
                }
                className="gap-1.5"
              >
                <Play className="h-4 w-4" />
                Test Ramp (2s)
              </Button>
            </div>
          </div>

          {/* Output Mappings */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-medium">Source Mappings</h3>
              <Button
                variant="outline"
                size="sm"
                onClick={() => {
                  setNewOutputMapping(m => ({
                    ...m,
                    universe: config?.output_universe ?? 0,
                  }));
                  setIsAddDialogOpen(true);
                }}
                className="gap-1.5"
              >
                <Plus className="h-4 w-4" />
                Add Mapping
              </Button>
            </div>

            {(!config?.output_mappings ||
              config.output_mappings.length === 0) && (
              <div className="text-sm text-muted-foreground text-center py-6 border border-dashed rounded-lg">
                {analysisSources.length === 0
                  ? "Add an analysis node to generate output data."
                  : "Add a mapping to send analysis values to DMX fixtures."}
              </div>
            )}

            {/* Grouped output mappings */}
            {groupedOutputMappings &&
              Object.entries(groupedOutputMappings).map(
                ([category, mappings]) => (
                  <div key={category} className="space-y-2">
                    <div className="flex items-center gap-2 p-2">
                      <span className="text-sm font-medium">
                        {categoryLabels[category] || category}
                      </span>
                      <span className="text-xs text-muted-foreground">
                        ({mappings.length})
                      </span>
                    </div>
                    <div className="space-y-2">
                      {mappings.map(mapping => (
                        <div
                          key={mapping.id}
                          className="flex items-center gap-3 p-3 rounded-lg bg-muted/30 border"
                        >
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2">
                              <code className="text-xs bg-accent/20 text-accent-foreground px-1.5 py-0.5 rounded truncate">
                                {mapping.source_key}
                              </code>
                              <span className="text-sm">→</span>
                              <code className="text-xs bg-muted px-1.5 py-0.5 rounded">
                                U{mapping.universe} Ch{mapping.channel}
                              </code>
                            </div>
                            <p className="text-xs text-muted-foreground mt-1">
                              Range: {mapping.min_value} → {mapping.max_value}
                            </p>
                          </div>
                          <Button
                            variant="ghost"
                            size="icon"
                            className="h-8 w-8 text-muted-foreground hover:text-destructive"
                            onClick={() =>
                              handleDeleteOutputMapping(mapping.id)
                            }
                          >
                            <Trash2 className="h-4 w-4" />
                          </Button>
                        </div>
                      ))}
                    </div>
                  </div>
                )
              )}
          </div>
        </TabsContent>
      </Tabs>

      {/* Add Mapping Dialog */}
      <Dialog open={isAddDialogOpen} onOpenChange={setIsAddDialogOpen}>
        <DialogContent className="sm:max-w-[425px]">
          <DialogHeader>
            <DialogTitle>
              Add {direction === "in" ? "Input" : "Output"} Mapping
            </DialogTitle>
          </DialogHeader>

          {direction === "in" ? (
            <div className="grid gap-4 py-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <label className="text-sm font-medium">Universe</label>
                  <Input
                    type="number"
                    min={0}
                    max={32767}
                    value={newInputMapping.universe}
                    onChange={e =>
                      setNewInputMapping(m => ({
                        ...m,
                        universe: parseInt(e.target.value) || 0,
                      }))
                    }
                  />
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium">Channel (1-512)</label>
                  <Input
                    type="number"
                    min={1}
                    max={512}
                    value={newInputMapping.channel}
                    onChange={e =>
                      setNewInputMapping(m => ({
                        ...m,
                        channel: parseInt(e.target.value) || 1,
                      }))
                    }
                  />
                </div>
              </div>

              <div className="space-y-2">
                <label className="text-sm font-medium">Parameter</label>
                <Select
                  value={newInputMapping.param_key}
                  onValueChange={value => {
                    // Find the parameter to get its range
                    let foundParam: ParameterInfo | undefined;
                    let foundCategory = "generation";
                    for (const [cat, params] of Object.entries(parameters)) {
                      const param = params.find(p => p.key === value);
                      if (param) {
                        foundParam = param;
                        foundCategory = cat;
                        break;
                      }
                    }
                    setNewInputMapping(m => ({
                      ...m,
                      param_key: value,
                      category: foundCategory,
                      min_value: foundParam?.min ?? 0,
                      max_value: foundParam?.max ?? 1,
                    }));
                  }}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Select a parameter" />
                  </SelectTrigger>
                  <SelectContent className="max-h-[300px]">
                    {Object.entries(parameters).map(([category, params]) =>
                      params.length > 0 ? (
                        <SelectGroup key={category}>
                          <SelectLabel>
                            {categoryLabels[category] || category}
                          </SelectLabel>
                          {params.map(param => (
                            <SelectItem key={param.key} value={param.key}>
                              <span className="font-mono text-xs">
                                {param.key}
                              </span>
                            </SelectItem>
                          ))}
                        </SelectGroup>
                      ) : null
                    )}
                  </SelectContent>
                </Select>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <label className="text-sm font-medium">Min Value</label>
                  <Input
                    type="number"
                    step="0.01"
                    value={newInputMapping.min_value}
                    onChange={e =>
                      setNewInputMapping(m => ({
                        ...m,
                        min_value: parseFloat(e.target.value) || 0,
                      }))
                    }
                  />
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium">Max Value</label>
                  <Input
                    type="number"
                    step="0.01"
                    value={newInputMapping.max_value}
                    onChange={e =>
                      setNewInputMapping(m => ({
                        ...m,
                        max_value: parseFloat(e.target.value) || 1,
                      }))
                    }
                  />
                </div>
              </div>

              <p className="text-xs text-muted-foreground">
                DMX value 0 → Min Value, DMX value 255 → Max Value
              </p>
            </div>
          ) : (
            <div className="grid gap-4 py-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <label className="text-sm font-medium">Universe</label>
                  <Input
                    type="number"
                    min={0}
                    max={32767}
                    value={newOutputMapping.universe}
                    onChange={e =>
                      setNewOutputMapping(m => ({
                        ...m,
                        universe: parseInt(e.target.value) || 0,
                      }))
                    }
                  />
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium">Channel (1-512)</label>
                  <Input
                    type="number"
                    min={1}
                    max={512}
                    value={newOutputMapping.channel}
                    onChange={e =>
                      setNewOutputMapping(m => ({
                        ...m,
                        channel: parseInt(e.target.value) || 1,
                      }))
                    }
                  />
                </div>
              </div>

              <div className="space-y-2">
                <label className="text-sm font-medium">Source</label>
                <Select
                  value={newOutputMapping.source_key}
                  onValueChange={value => {
                    const source = analysisSources.find(s => s.key === value);
                    setNewOutputMapping(m => ({
                      ...m,
                      source_key: value,
                      category: source?.category || "analysis",
                      min_value: source?.min ?? 0,
                      max_value: source?.max ?? 1,
                    }));
                  }}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Select a source" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectGroup>
                      <SelectLabel>Analysis</SelectLabel>
                      {analysisSources.map(source => (
                        <SelectItem key={source.key} value={source.key}>
                          <span className="font-mono text-xs">
                            {source.key}
                          </span>
                        </SelectItem>
                      ))}
                    </SelectGroup>
                  </SelectContent>
                </Select>
                {newOutputMapping.source_key && (
                  <p className="text-xs text-muted-foreground">
                    {
                      analysisSources.find(
                        s => s.key === newOutputMapping.source_key
                      )?.description
                    }
                  </p>
                )}
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <label className="text-sm font-medium">Min Value</label>
                  <Input
                    type="number"
                    step="0.01"
                    value={newOutputMapping.min_value}
                    onChange={e =>
                      setNewOutputMapping(m => ({
                        ...m,
                        min_value: parseFloat(e.target.value) || 0,
                      }))
                    }
                  />
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium">Max Value</label>
                  <Input
                    type="number"
                    step="0.01"
                    value={newOutputMapping.max_value}
                    onChange={e =>
                      setNewOutputMapping(m => ({
                        ...m,
                        max_value: parseFloat(e.target.value) || 1,
                      }))
                    }
                  />
                </div>
              </div>

              <p className="text-xs text-muted-foreground">
                Source Min Value → DMX 0, Source Max Value → DMX 255
              </p>
            </div>
          )}

          <DialogFooter>
            <Button variant="outline" onClick={() => setIsAddDialogOpen(false)}>
              Cancel
            </Button>
            <Button
              onClick={
                direction === "in"
                  ? handleAddInputMapping
                  : handleAddOutputMapping
              }
            >
              Add Mapping
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
