import { useState, useCallback } from "react";
import type { TimelinePrompt } from "../components/PromptTimeline";
import type { SettingsState } from "../types";
import { generateRandomColor } from "../utils/promptColors";

const DEFAULT_VISIBLE_END_TIME = 20;

interface UseTimelineImportExportParams {
  prompts: TimelinePrompt[];
  settings?: SettingsState;
  onPromptsChange: (prompts: TimelinePrompt[]) => void;
  onSettingsImport?: (settings: Partial<SettingsState>) => void;
  onTimeChange?: (time: number) => void;
  onPromptSubmit?: (prompt: string) => void;
  resetTimelineUI: () => void;
  setVisibleStartTime: (time: number) => void;
  setVisibleEndTime: (time: number) => void;
}

export function useTimelineImportExport({
  prompts,
  settings,
  onPromptsChange,
  onSettingsImport,
  onTimeChange,
  onPromptSubmit,
  resetTimelineUI,
  setVisibleStartTime,
  setVisibleEndTime,
}: UseTimelineImportExportParams) {
  const [showExportDialog, setShowExportDialog] = useState(false);

  const handleSaveTimeline = useCallback(() => {
    const exportPrompts = prompts
      .filter(prompt => prompt.startTime !== prompt.endTime)
      .map(prompt => {
        const { id, text, isLive, color, ...exportPrompt } = prompt;

        if (!exportPrompt.prompts && text) {
          exportPrompt.prompts = [{ text, weight: 100 }];
        }

        void id;
        void text;
        void isLive;
        void color;
        return exportPrompt;
      });

    const timelineData = {
      prompts: exportPrompts,
      settings: settings
        ? {
            pipelineId: settings.pipelineId,
            inputMode: settings.inputMode,
            resolution: settings.resolution,
            denoisingSteps: settings.denoisingSteps,
            noiseScale: settings.noiseScale,
            noiseController: settings.noiseController,
            manageCache: settings.manageCache,
            quantization: settings.quantization,
            kvCacheAttentionBias: settings.kvCacheAttentionBias,
            loras: settings.loras,
            loraMergeStrategy: settings.loraMergeStrategy,
            schemaFieldOverrides: settings.schemaFieldOverrides,
          }
        : undefined,
      version: "2.1",
      exportedAt: new Date().toISOString(),
    };

    const dataStr = JSON.stringify(timelineData, null, 2);
    const dataBlob = new Blob([dataStr], { type: "application/json" });
    const url = URL.createObjectURL(dataBlob);

    const link = document.createElement("a");
    link.href = url;
    link.download = `timeline-${new Date().toISOString().split("T")[0]}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }, [prompts, settings]);

  const handleExport = useCallback(() => {
    setShowExportDialog(true);
  }, []);

  const handleImport = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      if (!file) return;

      const reader = new FileReader();
      reader.onload = e => {
        try {
          const content = e.target?.result as string;
          const timelineData = JSON.parse(content);

          if (timelineData.prompts && Array.isArray(timelineData.prompts)) {
            resetTimelineUI();

            const importedPrompts = timelineData.prompts.map(
              (prompt: Partial<TimelinePrompt>, index: number) => ({
                ...prompt,
                id: prompt.id || `imported-${Date.now()}-${index}`,
                text:
                  prompt.text ||
                  (prompt.prompts && prompt.prompts.length > 0
                    ? prompt.prompts
                        .map((p: { text: string; weight: number }) => p.text)
                        .join(", ")
                    : ""),
                isLive: prompt.isLive || false,
                color: prompt.color || generateRandomColor(),
              })
            );
            onPromptsChange(importedPrompts);

            if (importedPrompts.length > 0) {
              const maxEndTime = Math.max(
                ...importedPrompts.map((p: TimelinePrompt) => p.endTime || 0)
              );
              const newVisibleEndTime = Math.max(
                maxEndTime + 2,
                DEFAULT_VISIBLE_END_TIME
              );
              setVisibleStartTime(0);
              setVisibleEndTime(newVisibleEndTime);
            }

            if (timelineData.settings && onSettingsImport) {
              onSettingsImport(timelineData.settings);
            }

            onTimeChange?.(0);

            if (importedPrompts.length > 0 && onPromptSubmit) {
              const firstPrompt = importedPrompts[0];
              onPromptSubmit(firstPrompt.text);
            }
          } else {
            alert("Invalid timeline file format");
          }
        } catch (error) {
          alert("Error reading timeline file");
          console.error("Import error:", error);
        }
      };
      reader.readAsText(file);

      event.target.value = "";
    },
    [
      onPromptsChange,
      onSettingsImport,
      resetTimelineUI,
      onTimeChange,
      onPromptSubmit,
      setVisibleStartTime,
      setVisibleEndTime,
    ]
  );

  return {
    showExportDialog,
    setShowExportDialog,
    handleExport,
    handleImport,
    handleSaveTimeline,
  };
}
