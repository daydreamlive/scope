import { useState, useEffect, useCallback } from "react";
import { BookOpenText, Loader2, Plus, Trash2 } from "lucide-react";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import { Label } from "../ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
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

interface DmxStatus {
  enabled: boolean;
  listening: boolean;
  port: number | null;
  mapping_count: number;
}

interface DmxMapping {
  id: string;
  universe: number;
  channel: number;
  param_key: string;
  min_value: number;
  max_value: number;
  enabled: boolean;
}

interface OscPath {
  key: string;
  type: string;
  description: string;
  min?: number;
  max?: number;
}

interface DmxTabProps {
  isActive: boolean;
}

export function DmxTab({ isActive }: DmxTabProps) {
  const [status, setStatus] = useState<DmxStatus | null>(null);
  const [mappings, setMappings] = useState<DmxMapping[]>([]);
  const [availableParams, setAvailableParams] = useState<OscPath[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isAddDialogOpen, setIsAddDialogOpen] = useState(false);

  // New mapping form state
  const [newMapping, setNewMapping] = useState({
    universe: 0,
    channel: 1,
    param_key: "",
    min_value: 0,
    max_value: 1,
  });

  const fetchStatus = useCallback(async () => {
    setIsLoading(true);
    try {
      const res = await fetch("/api/v1/dmx/status");
      if (res.ok) {
        setStatus(await res.json());
      }
    } catch (err) {
      console.error("Failed to fetch DMX status:", err);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const fetchMappings = useCallback(async () => {
    try {
      const res = await fetch("/api/v1/dmx/mappings");
      if (res.ok) {
        const data = await res.json();
        setMappings(data.mappings || []);
      }
    } catch (err) {
      console.error("Failed to fetch DMX mappings:", err);
    }
  }, []);

  const fetchAvailableParams = useCallback(async () => {
    try {
      // Reuse OSC paths endpoint - same params are available for DMX
      const res = await fetch("/api/v1/osc/paths");
      if (res.ok) {
        const data = await res.json();
        const paths: OscPath[] = [];
        // Collect all paths from active and available groups
        for (const group of Object.values(data.active || {})) {
          paths.push(...(group as OscPath[]));
        }
        for (const group of Object.values(data.available || {})) {
          paths.push(...(group as OscPath[]));
        }
        setAvailableParams(paths);
      }
    } catch (err) {
      console.error("Failed to fetch available params:", err);
    }
  }, []);

  useEffect(() => {
    if (isActive) {
      fetchStatus();
      fetchMappings();
      fetchAvailableParams();
    }
  }, [isActive, fetchStatus, fetchMappings, fetchAvailableParams]);

  const handleAddMapping = async () => {
    if (!newMapping.param_key) {
      toast.error("Please select a parameter");
      return;
    }

    try {
      const res = await fetch("/api/v1/dmx/mappings", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ...newMapping,
          id: `${newMapping.universe}-${newMapping.channel}-${Date.now()}`,
          enabled: true,
        }),
      });

      if (res.ok) {
        toast.success("Mapping added");
        setIsAddDialogOpen(false);
        setNewMapping({
          universe: 0,
          channel: 1,
          param_key: "",
          min_value: 0,
          max_value: 1,
        });
        fetchMappings();
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

  const handleDeleteMapping = async (id: string) => {
    try {
      const res = await fetch(`/api/v1/dmx/mappings/${encodeURIComponent(id)}`, {
        method: "DELETE",
      });

      if (res.ok) {
        toast.success("Mapping removed");
        fetchMappings();
        fetchStatus();
      } else {
        toast.error("Failed to remove mapping");
      }
    } catch (err) {
      toast.error("Failed to remove mapping");
      console.error(err);
    }
  };

  const handleOpenDocs = () => {
    openExternalUrl(`${window.location.origin}/api/v1/dmx/docs`);
  };

  // Get param info for display
  const getParamInfo = (key: string) => {
    return availableParams.find(p => p.key === key);
  };

  return (
    <div className="space-y-4">
      <div className="rounded-lg bg-muted/50 p-4 space-y-4">
        {/* Status */}
        <div className="flex items-center gap-4">
          <span className="text-sm font-medium text-foreground w-32">
            Status
          </span>
          <div className="flex-1 flex items-center justify-end">
            {isLoading ? (
              <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
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

        {/* Port */}
        {status?.port && (
          <div className="flex items-center gap-4">
            <span className="text-sm font-medium text-foreground w-32">
              Art-Net Port
            </span>
            <div className="flex-1 flex items-center justify-end">
              <span className="text-sm text-muted-foreground font-mono">
                {status.port}
              </span>
            </div>
          </div>
        )}

        {/* Mappings count */}
        <div className="flex items-center gap-4">
          <span className="text-sm font-medium text-foreground w-32">
            Active Mappings
          </span>
          <div className="flex-1 flex items-center justify-end">
            <span className="text-sm text-muted-foreground">
              {status?.mapping_count ?? 0}
            </span>
          </div>
        </div>

        {/* Docs */}
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

        {/* Quick-start hint */}
        <div className="text-xs text-muted-foreground leading-relaxed">
          Send Art-Net DMX data to UDP port{" "}
          <code className="bg-muted px-1 py-0.5 rounded text-xs">
            {status?.port ?? 6454}
          </code>{" "}
          to control pipeline parameters. Map DMX channels to parameters below.
        </div>
      </div>

      {/* Mappings Section */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-medium">Channel Mappings</h3>
          <Button
            variant="outline"
            size="sm"
            onClick={() => setIsAddDialogOpen(true)}
            className="gap-1.5"
          >
            <Plus className="h-4 w-4" />
            Add Mapping
          </Button>
        </div>

        {mappings.length === 0 ? (
          <div className="text-sm text-muted-foreground text-center py-4 border border-dashed rounded-lg">
            No mappings configured. Add a mapping to control parameters via DMX.
          </div>
        ) : (
          <div className="space-y-2">
            {mappings.map(mapping => {
              const paramInfo = getParamInfo(mapping.param_key);
              return (
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
                      <code className="text-xs bg-accent/20 text-accent-foreground px-1.5 py-0.5 rounded">
                        {mapping.param_key}
                      </code>
                    </div>
                    {paramInfo && (
                      <p className="text-xs text-muted-foreground mt-1 truncate">
                        {paramInfo.description}
                      </p>
                    )}
                    <p className="text-xs text-muted-foreground">
                      Range: {mapping.min_value} → {mapping.max_value}
                    </p>
                  </div>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8 text-muted-foreground hover:text-destructive"
                    onClick={() => handleDeleteMapping(mapping.id)}
                  >
                    <Trash2 className="h-4 w-4" />
                  </Button>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Add Mapping Dialog */}
      <Dialog open={isAddDialogOpen} onOpenChange={setIsAddDialogOpen}>
        <DialogContent className="sm:max-w-[425px]">
          <DialogHeader>
            <DialogTitle>Add DMX Mapping</DialogTitle>
          </DialogHeader>
          <div className="grid gap-4 py-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="universe">Universe</Label>
                <Input
                  id="universe"
                  type="number"
                  min={0}
                  max={32767}
                  value={newMapping.universe}
                  onChange={e =>
                    setNewMapping(m => ({
                      ...m,
                      universe: parseInt(e.target.value) || 0,
                    }))
                  }
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="channel">Channel (1-512)</Label>
                <Input
                  id="channel"
                  type="number"
                  min={1}
                  max={512}
                  value={newMapping.channel}
                  onChange={e =>
                    setNewMapping(m => ({
                      ...m,
                      channel: parseInt(e.target.value) || 1,
                    }))
                  }
                />
              </div>
            </div>

            <div className="space-y-2">
              <Label htmlFor="param">Parameter</Label>
              <Select
                value={newMapping.param_key}
                onValueChange={value => {
                  const param = availableParams.find(p => p.key === value);
                  setNewMapping(m => ({
                    ...m,
                    param_key: value,
                    min_value: param?.min ?? 0,
                    max_value: param?.max ?? 1,
                  }));
                }}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select a parameter" />
                </SelectTrigger>
                <SelectContent className="max-h-[200px]">
                  {availableParams.map(param => (
                    <SelectItem key={param.key} value={param.key}>
                      <span className="font-mono text-xs">{param.key}</span>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              {newMapping.param_key && (
                <p className="text-xs text-muted-foreground">
                  {getParamInfo(newMapping.param_key)?.description}
                </p>
              )}
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="min_value">Min Value</Label>
                <Input
                  id="min_value"
                  type="number"
                  step="0.01"
                  value={newMapping.min_value}
                  onChange={e =>
                    setNewMapping(m => ({
                      ...m,
                      min_value: parseFloat(e.target.value) || 0,
                    }))
                  }
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="max_value">Max Value</Label>
                <Input
                  id="max_value"
                  type="number"
                  step="0.01"
                  value={newMapping.max_value}
                  onChange={e =>
                    setNewMapping(m => ({
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
          <DialogFooter>
            <Button variant="outline" onClick={() => setIsAddDialogOpen(false)}>
              Cancel
            </Button>
            <Button onClick={handleAddMapping}>Add Mapping</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
