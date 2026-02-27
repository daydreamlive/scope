import { useMIDI } from "../contexts/MIDIContext";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "./ui/dialog";
import { Button } from "./ui/button";
import { Trash2 } from "lucide-react";
import type { MIDIMapping } from "../types/midi";

interface MIDIMappingsModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

function formatControllerInfo(mapping: MIDIMapping): string {
  if (mapping.source.midi_cc !== undefined) {
    return `CC ${mapping.source.midi_cc} (Ch ${mapping.source.channel})`;
  }
  if (mapping.source.midi_note !== undefined) {
    return `Note ${mapping.source.midi_note} (Ch ${mapping.source.channel})`;
  }
  return "Not mapped";
}

function formatTargetInfo(mapping: MIDIMapping): string {
  if (mapping.target.action) {
    const actionNames: Record<string, string> = {
      switch_prompt: "Switch Prompt",
      switch_prompt_0: "Switch Prompt 0",
      switch_prompt_1: "Switch Prompt 1",
      switch_prompt_2: "Switch Prompt 2",
      switch_prompt_3: "Switch Prompt 3",
      reset_cache: "Reset Cache",
      toggle_pause: "Toggle Pause",
      add_denoising_step: "Add Denoising Step",
      remove_denoising_step: "Remove Denoising Step",
    };
    return actionNames[mapping.target.action] || mapping.target.action;
  }
  if (mapping.target.parameter) {
    const paramName = mapping.target.parameter;
    if (mapping.target.arrayIndex !== undefined) {
      return `${paramName}[${mapping.target.arrayIndex}]`;
    }
    return paramName;
  }
  return "Unknown";
}

function formatMappingType(type: string): string {
  return type.charAt(0).toUpperCase() + type.slice(1);
}

export function MIDIMappingsModal({
  open,
  onOpenChange,
}: MIDIMappingsModalProps) {
  const { mappingProfile, deleteMapping, clearAllMappings } = useMIDI();
  const mappings = mappingProfile.mappings;

  const handleDelete = (index: number) => {
    deleteMapping(index);
  };

  const handleClearAll = () => {
    if (mappings.length === 0) return;
    if (confirm("Are you sure you want to clear all mappings?")) {
      clearAllMappings();
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl max-h-[80vh] flex flex-col">
        <DialogHeader>
          <DialogTitle>MIDI Mappings</DialogTitle>
          <DialogDescription>
            View and manage your MIDI controller mappings. Delete individual
            mappings or clear all at once.
          </DialogDescription>
        </DialogHeader>

        <div className="flex-1 overflow-y-auto min-h-0">
          {mappings.length === 0 ? (
            <div className="text-center py-8 text-muted-foreground">
              No mappings configured. Use "Edit Mapping" to create mappings.
            </div>
          ) : (
            <div className="space-y-2">
              {mappings.map((mapping, index) => (
                <div
                  key={index}
                  className="flex items-center justify-between p-3 border rounded-md hover:bg-accent/50 transition-colors"
                >
                  <div className="flex-1 grid grid-cols-4 gap-4 items-center">
                    <div className="font-medium text-sm">
                      {formatControllerInfo(mapping)}
                    </div>
                    <div className="text-sm text-muted-foreground">
                      {formatTargetInfo(mapping)}
                    </div>
                    <div className="text-xs text-muted-foreground">
                      {formatMappingType(mapping.type)}
                      {mapping.range && (
                        <span className="ml-1">
                          ({mapping.range.min} - {mapping.range.max})
                        </span>
                      )}
                    </div>
                    <div className="flex justify-end">
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-8 w-8 text-destructive hover:text-destructive hover:bg-destructive/10"
                        onClick={() => handleDelete(index)}
                        title="Delete mapping"
                      >
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Close
          </Button>
          {mappings.length > 0 && (
            <Button variant="destructive" onClick={handleClearAll}>
              Clear All Mappings
            </Button>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
