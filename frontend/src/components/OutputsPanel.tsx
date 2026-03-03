import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Input } from "./ui/input";
import { Toggle } from "./ui/toggle";
import { LabelWithTooltip } from "./ui/label-with-tooltip";
import { PARAMETER_METADATA } from "../data/parameterMetadata";
import type { SettingsState } from "../types";

interface OutputsPanelProps {
  className?: string;
  outputSinks?: SettingsState["outputSinks"];
  onOutputSinkChange?: (
    sinkType: string,
    config: { enabled: boolean; name: string }
  ) => void;
  spoutAvailable?: boolean;
  ndiAvailable?: boolean;
  syphonAvailable?: boolean;
  isStreaming?: boolean;
}

export function OutputsPanel({
  className = "",
  outputSinks,
  onOutputSinkChange,
  spoutAvailable = false,
  ndiAvailable = false,
  syphonAvailable = false,
  isStreaming = false,
}: OutputsPanelProps) {
  const hasAvailableOutputs = spoutAvailable || ndiAvailable || syphonAvailable;
  if (!hasAvailableOutputs) return null;

  return (
    <Card className={className}>
      <CardHeader className="px-4 py-3">
        <CardTitle className="text-base font-medium">Outputs</CardTitle>
      </CardHeader>
      <CardContent className="px-4 pb-4 pt-0 space-y-3">
        {/* Spout Output */}
        {spoutAvailable && (
          <div className="space-y-3">
            <div className="flex items-center justify-between gap-2">
              <LabelWithTooltip
                label="Spout Output"
                tooltip="Send video to Spout-compatible apps (Windows) like TouchDesigner, Resolume, OBS."
                className="text-sm font-medium"
              />
              <Toggle
                pressed={outputSinks?.spout?.enabled ?? false}
                onPressedChange={enabled => {
                  onOutputSinkChange?.("spout", {
                    enabled,
                    name: outputSinks?.spout?.name ?? "ScopeOut",
                  });
                }}
                variant="outline"
                size="sm"
                className="h-7"
              >
                {outputSinks?.spout?.enabled ? "ON" : "OFF"}
              </Toggle>
            </div>

            {outputSinks?.spout?.enabled && (
              <div className="flex items-center gap-3">
                <LabelWithTooltip
                  label="Sender Name"
                  tooltip="The name visible to Spout receivers."
                  className="text-xs text-muted-foreground whitespace-nowrap"
                />
                <Input
                  type="text"
                  value={outputSinks?.spout?.name ?? "ScopeOut"}
                  onChange={e => {
                    onOutputSinkChange?.("spout", {
                      enabled: outputSinks?.spout?.enabled ?? false,
                      name: e.target.value,
                    });
                  }}
                  disabled={isStreaming}
                  className="h-8 text-sm flex-1"
                  placeholder="ScopeOut"
                />
              </div>
            )}
          </div>
        )}

        {/* NDI Output */}
        {ndiAvailable && (
          <div className="space-y-3">
            <div className="flex items-center justify-between gap-2">
              <LabelWithTooltip
                label={PARAMETER_METADATA.ndiSender.label}
                tooltip={PARAMETER_METADATA.ndiSender.tooltip}
                className="text-sm font-medium"
              />
              <Toggle
                pressed={outputSinks?.ndi?.enabled ?? false}
                onPressedChange={enabled => {
                  onOutputSinkChange?.("ndi", {
                    enabled,
                    name: outputSinks?.ndi?.name ?? "Scope",
                  });
                }}
                variant="outline"
                size="sm"
                className="h-7"
              >
                {outputSinks?.ndi?.enabled ? "ON" : "OFF"}
              </Toggle>
            </div>

            {outputSinks?.ndi?.enabled && (
              <div className="flex items-center gap-3">
                <LabelWithTooltip
                  label="Sender Name"
                  tooltip="The name visible to NDI receivers on the network."
                  className="text-xs text-muted-foreground whitespace-nowrap"
                />
                <Input
                  type="text"
                  value={outputSinks?.ndi?.name ?? "Scope"}
                  onChange={e => {
                    onOutputSinkChange?.("ndi", {
                      enabled: outputSinks?.ndi?.enabled ?? false,
                      name: e.target.value,
                    });
                  }}
                  disabled={isStreaming}
                  className="h-8 text-sm flex-1"
                  placeholder="Scope"
                />
              </div>
            )}
          </div>
        )}

        {/* Syphon Output */}
        {syphonAvailable && (
          <div className="space-y-3">
            <div className="flex items-center justify-between gap-2">
              <LabelWithTooltip
                label={PARAMETER_METADATA.syphonSender.label}
                tooltip={PARAMETER_METADATA.syphonSender.tooltip}
                className="text-sm font-medium"
              />
              <Toggle
                pressed={outputSinks?.syphon?.enabled ?? false}
                onPressedChange={enabled => {
                  onOutputSinkChange?.("syphon", {
                    enabled,
                    name: outputSinks?.syphon?.name ?? "Scope",
                  });
                }}
                variant="outline"
                size="sm"
                className="h-7"
              >
                {outputSinks?.syphon?.enabled ? "ON" : "OFF"}
              </Toggle>
            </div>

            {outputSinks?.syphon?.enabled && (
              <div className="flex items-center gap-3">
                <LabelWithTooltip
                  label="Sender Name"
                  tooltip="The name visible to Syphon receivers on this Mac."
                  className="text-xs text-muted-foreground whitespace-nowrap"
                />
                <Input
                  type="text"
                  value={outputSinks?.syphon?.name ?? "Scope"}
                  onChange={e => {
                    onOutputSinkChange?.("syphon", {
                      enabled: outputSinks?.syphon?.enabled ?? false,
                      name: e.target.value,
                    });
                  }}
                  disabled={isStreaming}
                  className="h-8 text-sm flex-1"
                  placeholder="Scope"
                />
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
