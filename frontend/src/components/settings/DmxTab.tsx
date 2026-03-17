import { useState, useEffect, useCallback, useRef } from "react";
import { Loader2, Plus, Trash2, Download, Upload, Save } from "lucide-react";
import { Button } from "../ui/button";
import { Switch } from "../ui/switch";
import { Input } from "../ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../ui/select";
import { toast } from "sonner";
import {
  getDmxStatus,
  getDmxPaths,
  getDmxConfig,
  saveDmxConfig,
  updateDmxSettings,
  applyDmxPort,
  type DmxStatusResponse,
  type DmxMapping,
  type DmxPathEntry,
  type DmxPathsResponse,
  type DmxConfigResponse,
} from "@/lib/api";

interface DmxTabProps {
  isActive: boolean;
}

interface FlatParam {
  key: string;
  type: string;
  description: string;
  min: number;
  max: number;
  group: string;
}

function flattenPaths(pathsResponse: DmxPathsResponse | null): FlatParam[] {
  if (!pathsResponse) return [];
  const result: FlatParam[] = [];
  const addGroup = (
    groups: Record<string, DmxPathEntry[]>,
    _section: string
  ) => {
    for (const [group, entries] of Object.entries(groups)) {
      for (const entry of entries) {
        result.push({
          key: entry.key,
          type: entry.type,
          description: entry.description,
          min: entry.min ?? 0,
          max: entry.max ?? 1,
          group,
        });
      }
    }
  };
  addGroup(pathsResponse.active, "Active");
  addGroup(pathsResponse.available, "Available");
  return result;
}

export function DmxTab({ isActive }: DmxTabProps) {
  const [status, setStatus] = useState<DmxStatusResponse | null>(null);
  const [, setConfig] = useState<DmxConfigResponse | null>(null);
  const [paths, setPaths] = useState<DmxPathsResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [isApplyingPort, setIsApplyingPort] = useState(false);
  const [dirty, setDirty] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const dirtyRef = useRef(dirty);

  const [localMappings, setLocalMappings] = useState<DmxMapping[]>([]);
  const [localPort, setLocalPort] = useState<string>("6454");

  const flatParams = flattenPaths(paths);

  useEffect(() => {
    dirtyRef.current = dirty;
  }, [dirty]);

  const fetchAll = useCallback(async () => {
    setIsLoading(true);
    try {
      const [s, c, p] = await Promise.all([
        getDmxStatus(),
        getDmxConfig(),
        getDmxPaths(),
      ]);
      setStatus(s);
      setConfig(c);
      setPaths(p);
      if (!dirtyRef.current) {
        setLocalMappings(c.mappings);
        setLocalPort(String(c.preferred_port));
        setDirty(false);
      }
    } catch (err) {
      console.error("Failed to fetch DMX state:", err);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    if (isActive) {
      fetchAll();
    }
  }, [isActive, fetchAll]);

  const handleSave = async () => {
    setIsSaving(true);
    try {
      const port = parseInt(localPort, 10);
      const newConfig = await saveDmxConfig({
        preferred_port: isNaN(port) ? 6454 : port,
        log_all_messages: status?.log_all_messages ?? false,
        mappings: localMappings,
      });
      setConfig(newConfig);
      setDirty(false);
      toast.success("DMX configuration saved");

      const s = await getDmxStatus();
      setStatus(s);
    } catch (err) {
      toast.error("Failed to save DMX config");
      console.error(err);
    } finally {
      setIsSaving(false);
    }
  };

  const handleToggleEnabled = async (checked: boolean) => {
    try {
      const updated = await updateDmxSettings({ enabled: checked });
      setStatus(prev => (prev ? { ...prev, ...updated } : prev));
    } catch (err) {
      toast.error("Failed to toggle DMX");
      console.error(err);
    }
  };

  const handleToggleLogging = async (checked: boolean) => {
    try {
      const updated = await updateDmxSettings({ log_all_messages: checked });
      setStatus(prev => (prev ? { ...prev, ...updated } : prev));
      setDirty(true);
    } catch (err) {
      toast.error("Failed to update DMX logging");
      console.error(err);
    }
  };

  const handleApplyPort = async () => {
    const port = parseInt(localPort, 10);
    if (isNaN(port) || port < 1024 || port > 65535) {
      toast.error("Enter a valid port (1024–65535)");
      return;
    }
    setIsApplyingPort(true);
    try {
      const updated = await applyDmxPort(port);
      setStatus(prev => (prev ? { ...prev, ...updated } : prev));
      if (updated.listening) {
        setDirty(false);
        toast.success(`DMX now listening on port ${updated.port ?? port}`);
      } else {
        toast.error("Failed to start listening on the requested port");
      }
    } catch (err) {
      toast.error("Failed to apply port");
      console.error(err);
    } finally {
      setIsApplyingPort(false);
    }
  };

  const addMapping = () => {
    setLocalMappings(prev => [...prev, { universe: 0, channel: 0, key: "" }]);
    setDirty(true);
  };

  const removeMapping = (index: number) => {
    setLocalMappings(prev => prev.filter((_, i) => i !== index));
    setDirty(true);
  };

  const updateMapping = (
    index: number,
    field: keyof DmxMapping,
    value: string | number
  ) => {
    setLocalMappings(prev =>
      prev.map((m, i) => (i === index ? { ...m, [field]: value } : m))
    );
    setDirty(true);
  };

  const handleExport = () => {
    const exportData: DmxConfigResponse = {
      enabled: status?.enabled ?? false,
      preferred_port: parseInt(localPort, 10) || 6454,
      log_all_messages: status?.log_all_messages ?? false,
      mappings: localMappings,
    };
    const blob = new Blob([JSON.stringify(exportData, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "scope-dmx-config.json";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
    toast.success("DMX config exported");
  };

  const handleImport = async (file: File) => {
    try {
      const text = await file.text();
      const imported = JSON.parse(text);
      if (!Array.isArray(imported.mappings)) {
        toast.error("Invalid DMX config file");
        return;
      }
      const mappings: DmxMapping[] = imported.mappings.filter(
        (m: DmxMapping) =>
          typeof m.universe === "number" &&
          typeof m.channel === "number" &&
          typeof m.key === "string" &&
          m.key.length > 0
      );
      setLocalMappings(mappings);
      if (typeof imported.preferred_port === "number") {
        setLocalPort(String(imported.preferred_port));
      }
      setDirty(true);
      toast.success(`Imported ${mappings.length} mapping(s)`);
    } catch {
      toast.error("Failed to parse DMX config file");
    }
  };

  const getParamInfo = (key: string): FlatParam | undefined =>
    flatParams.find(p => p.key === key);

  // Group params for the select dropdown (dedupe by key so the same param
  // from multiple pipelines doesn't render multiple SelectItems and produce
  // duplicated text like "zoomzoomzoom" in the trigger)
  const groupedParams = (() => {
    const groups: Record<string, FlatParam[]> = {};
    const seenKeys = new Set<string>();
    for (const p of flatParams) {
      if (seenKeys.has(p.key)) continue;
      seenKeys.add(p.key);
      if (!groups[p.group]) groups[p.group] = [];
      groups[p.group].push(p);
    }
    return groups;
  })();

  return (
    <div className="space-y-4">
      <div className="rounded-lg bg-muted/50 p-4 space-y-4">
        {/* Enable / Disable */}
        <div className="flex items-center gap-4">
          <span className="text-sm font-medium text-foreground w-32">
            DMX Input
          </span>
          <div className="flex-1 flex items-center justify-end gap-2">
            <span className="text-xs text-muted-foreground">
              {status?.enabled ? "Enabled" : "Disabled"}
            </span>
            <Switch
              aria-label="Enable DMX input"
              checked={status?.enabled ?? false}
              onCheckedChange={handleToggleEnabled}
              disabled={isLoading}
              className="data-[state=unchecked]:bg-zinc-600 data-[state=checked]:bg-green-500"
            />
          </div>
        </div>

        {/* Status */}
        <div className="flex items-center gap-4">
          <span className="text-sm font-medium text-foreground w-32">
            Status
          </span>
          <div className="flex-1 flex items-center justify-end">
            {isLoading ? (
              <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
            ) : !status?.enabled ? (
              <span className="text-sm text-muted-foreground">Disabled</span>
            ) : status?.listening ? (
              <span className="text-sm text-green-500">
                Listening on UDP port {status.port}
              </span>
            ) : (
              <span className="text-sm text-muted-foreground">
                Not listening
              </span>
            )}
          </div>
        </div>

        {/* Preferred Port */}
        <div className="flex items-center gap-4">
          <span className="text-sm font-medium text-foreground w-32">
            Preferred Port
          </span>
          <div className="flex-1 flex items-center justify-end gap-2">
            <Input
              type="number"
              value={localPort}
              onChange={e => {
                setLocalPort(e.target.value);
                setDirty(true);
              }}
              className="w-24 h-8 text-sm font-mono text-right"
              min={1024}
              max={65535}
            />
            <Button
              variant="outline"
              size="sm"
              onClick={handleApplyPort}
              disabled={isApplyingPort || isLoading || !status?.enabled}
              className="h-8"
            >
              {isApplyingPort ? (
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
              ) : (
                "Apply"
              )}
            </Button>
          </div>
        </div>

        {/* Log all messages toggle */}
        <div className="flex items-center gap-4">
          <span className="text-sm font-medium text-foreground w-32">
            Log Messages
          </span>
          <div className="flex-1 flex items-center justify-end gap-2">
            <span className="text-xs text-muted-foreground">
              {status?.log_all_messages ? "All" : "Errors only"}
            </span>
            <Switch
              aria-label="Log all DMX messages"
              checked={status?.log_all_messages ?? false}
              onCheckedChange={handleToggleLogging}
              disabled={isLoading || !status?.enabled}
              className="data-[state=unchecked]:bg-zinc-600 data-[state=checked]:bg-green-500"
            />
          </div>
        </div>

        {/* Quick-start hint */}
        <div className="text-xs text-muted-foreground leading-relaxed">
          Send Art-Net DMX packets to UDP port{" "}
          <code className="bg-muted px-1 py-0.5 rounded text-xs">
            {status?.port ?? localPort}
          </code>{" "}
          to control pipeline parameters in real time. Map DMX channels to
          numeric parameters below.
        </div>
      </div>

      {/* Mappings */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <span className="text-sm font-medium">Channel Mappings</span>
          <Button
            variant="outline"
            size="sm"
            onClick={addMapping}
            className="gap-1.5 h-7"
          >
            <Plus className="h-3.5 w-3.5" />
            Add
          </Button>
        </div>

        {localMappings.length === 0 ? (
          <div className="text-xs text-muted-foreground text-center py-4">
            No mappings configured. Click Add to map a DMX channel to a
            parameter.
          </div>
        ) : (
          <div className="space-y-2 max-h-[200px] overflow-y-auto [&::-webkit-scrollbar]:w-2 [&::-webkit-scrollbar-track]:bg-transparent [&::-webkit-scrollbar-thumb]:bg-gray-300 [&::-webkit-scrollbar-thumb]:rounded-full">
            {/* Header */}
            <div className="grid grid-cols-[50px_60px_1fr_24px] gap-2 text-[10px] text-muted-foreground px-1">
              <span>Uni</span>
              <span>Ch</span>
              <span>Parameter</span>
              <span />
            </div>
            {localMappings.map((mapping, index) => {
              const info = getParamInfo(mapping.key);
              return (
                <div
                  key={index}
                  className="grid grid-cols-[50px_60px_1fr_24px] gap-2 items-center"
                >
                  <Input
                    type="number"
                    value={mapping.universe}
                    onChange={e =>
                      updateMapping(
                        index,
                        "universe",
                        parseInt(e.target.value, 10) || 0
                      )
                    }
                    className="h-7 text-xs px-1.5"
                    min={0}
                    max={32767}
                  />
                  <Input
                    type="number"
                    value={mapping.channel}
                    onChange={e =>
                      updateMapping(
                        index,
                        "channel",
                        parseInt(e.target.value, 10) || 0
                      )
                    }
                    className="h-7 text-xs px-1.5"
                    min={0}
                    max={511}
                  />
                  <div className="flex flex-col gap-0.5">
                    <Select
                      value={mapping.key || "__none__"}
                      onValueChange={v =>
                        updateMapping(index, "key", v === "__none__" ? "" : v)
                      }
                    >
                      <SelectTrigger className="h-7 text-xs">
                        <SelectValue placeholder="Select parameter" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="__none__" disabled>
                          Select parameter...
                        </SelectItem>
                        {Object.entries(groupedParams).map(
                          ([group, params]) => (
                            <div key={group}>
                              <div className="px-2 py-1 text-[10px] font-semibold text-muted-foreground uppercase tracking-wider">
                                {group}
                              </div>
                              {params.map(p => (
                                <SelectItem key={p.key} value={p.key}>
                                  {p.key}
                                </SelectItem>
                              ))}
                            </div>
                          )
                        )}
                      </SelectContent>
                    </Select>
                    {info && (
                      <span className="text-[10px] text-muted-foreground pl-1">
                        {info.description} ({info.min} – {info.max})
                      </span>
                    )}
                  </div>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-6 w-6 text-destructive hover:text-destructive hover:bg-destructive/10"
                    onClick={() => removeMapping(index)}
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                  </Button>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Actions */}
      <div className="flex items-center gap-2 pt-2 border-t border-border">
        <Button
          variant="default"
          size="sm"
          onClick={handleSave}
          disabled={!dirty || isSaving}
          className="gap-1.5"
        >
          <Save className="h-3.5 w-3.5" />
          {isSaving ? "Saving..." : "Save"}
        </Button>
        <Button
          variant="outline"
          size="sm"
          onClick={handleExport}
          className="gap-1.5"
        >
          <Download className="h-3.5 w-3.5" />
          Export
        </Button>
        <Button
          variant="outline"
          size="sm"
          onClick={() => fileInputRef.current?.click()}
          className="gap-1.5"
        >
          <Upload className="h-3.5 w-3.5" />
          Import
        </Button>
        <input
          ref={fileInputRef}
          type="file"
          accept=".json"
          className="hidden"
          onChange={e => {
            const file = e.target.files?.[0];
            if (file) handleImport(file);
            e.target.value = "";
          }}
        />
      </div>
    </div>
  );
}
