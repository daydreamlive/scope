import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Badge } from "./ui/badge";
import { ArrowUp } from "lucide-react";
import type { VideoSourceMode } from "../hooks/useVideoSource";
import type { ExtensionMode } from "../types";
import { PromptInput } from "./PromptInput";
import { TimelinePromptEditor } from "./TimelinePromptEditor";
import { ImageManager } from "./ImageManager";
import { Button } from "./ui/button";
import {
  type ConfigSchemaLike,
  type PrimitiveFieldType,
  COMPLEX_COMPONENTS,
  parseInputFields,
} from "../lib/schemaSettings";
import { SchemaPrimitiveField } from "./PrimitiveFields";
import { useAppStore } from "../stores";
import { useShallow } from "zustand/react/shallow";
import { usePipelinesContext } from "../contexts/PipelinesContext";
import { useStreamContext } from "../contexts/StreamContext";
import { VideoInputSection } from "./input/VideoInputSection";
import { ExtensionFramesSection } from "./input/ExtensionFramesSection";

interface InputAndControlsPanelProps {
  className?: string;
  localStream: MediaStream | null;
  isInitializing: boolean;
  error: string | null;
  mode: VideoSourceMode;
  onVideoFileUpload?: (file: File) => Promise<boolean>;
}

export function InputAndControlsPanel({
  className = "",
  localStream,
  isInitializing,
  error,
  mode,
  onVideoFileUpload,
}: InputAndControlsPanelProps) {
  const {
    settings,
    prompts,
    onPromptsChange,
    interpolationMethod,
    onInterpolationMethodChange,
    temporalInterpolationMethod,
    onTemporalInterpolationMethodChange,
    transitionSteps,
    onTransitionStepsChange,
    isLive,
    selectedTimelinePrompt,
    isTimelinePlaying,
    currentTime,
    timelinePrompts,
    isDownloading,
  } = useAppStore(
    useShallow(s => ({
      settings: s.settings,
      prompts: s.promptItems,
      onPromptsChange: s.setPromptItems,
      interpolationMethod: s.interpolationMethod,
      onInterpolationMethodChange: s.setInterpolationMethod,
      temporalInterpolationMethod: s.temporalInterpolationMethod,
      onTemporalInterpolationMethodChange: s.setTemporalInterpolationMethod,
      transitionSteps: s.transitionSteps,
      onTransitionStepsChange: s.setTransitionSteps,
      isLive: s.isLive,
      selectedTimelinePrompt: s.selectedTimelinePrompt,
      isTimelinePlaying: s.isTimelinePlaying,
      currentTime: s.timelineCurrentTime,
      timelinePrompts: s.timelinePrompts,
      isDownloading: s.isDownloading,
    }))
  );

  const { pipelines } = usePipelinesContext();
  const {
    actions,
    isStreaming,
    isConnecting,
    isPipelineLoading,
    spoutAvailable,
  } = useStreamContext();

  // Derive values from settings/pipelines
  const pipelineId = settings.pipelineId;
  const inputMode = settings.inputMode || "text";
  const isVideoPaused = settings.paused ?? false;
  const spoutReceiverName = settings.spoutReceiver?.name ?? "";
  const vaceEnabled =
    settings.vaceEnabled ??
    (pipelines?.[pipelineId]?.supportsVACE && inputMode !== "video");
  const refImages = settings.refImages || [];
  const pipeline = pipelines?.[pipelineId];
  const supportsImages = pipeline?.supportsImages ?? false;
  const firstFrameImage = settings.firstFrameImage;
  const lastFrameImage = settings.lastFrameImage;
  const extensionMode = settings.extensionMode || "firstframe";
  const configSchema = pipeline?.configSchema as
    | ConfigSchemaLike
    | undefined;
  const schemaFieldOverrides = settings.schemaFieldOverrides ?? {};
  const isMultiMode = (pipeline?.supportedModes?.length ?? 0) > 1;

  const isAtEndOfTimeline = () => {
    if (timelinePrompts.length === 0) return true;
    const lastPrompt = timelinePrompts[timelinePrompts.length - 1];
    return currentTime >= lastPrompt.endTime;
  };

  return (
    <Card className={`h-full flex flex-col ${className}`}>
      <CardHeader className="flex-shrink-0">
        <CardTitle className="text-base font-medium">
          Input & Controls
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4 overflow-y-auto flex-1 [&::-webkit-scrollbar]:w-2 [&::-webkit-scrollbar-track]:bg-transparent [&::-webkit-scrollbar-thumb]:bg-gray-300 [&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-thumb]:transition-colors [&::-webkit-scrollbar-thumb:hover]:bg-gray-400">
        <VideoInputSection
          inputMode={inputMode}
          isMultiMode={isMultiMode}
          mode={mode}
          spoutAvailable={spoutAvailable}
          spoutReceiverName={spoutReceiverName}
          localStream={localStream}
          isInitializing={isInitializing}
          error={error}
          isStreaming={isStreaming}
          isConnecting={isConnecting}
          onInputModeChange={actions.handleInputModeChange}
          onModeChange={actions.handleModeChange}
          onSpoutReceiverChange={actions.handleSpoutReceiverChange}
          onVideoFileUpload={onVideoFileUpload}
        />

        {/* Reference Images - show when VACE enabled OR when pipeline supports images without VACE */}
        {(vaceEnabled || (supportsImages && !pipeline?.supportsVACE)) && (
          <div>
            <ImageManager
              images={refImages}
              onImagesChange={actions.handleRefImagesChange || (() => {})}
              disabled={isDownloading}
              maxImages={1}
              singleColumn={false}
              label={
                vaceEnabled && pipeline?.supportsVACE
                  ? "Reference Images"
                  : "Images"
              }
              tooltip={
                vaceEnabled && pipeline?.supportsVACE
                  ? "Select reference images for VACE conditioning. Images will guide the video generation style and content."
                  : "Select images to send to the pipeline for conditioning."
              }
            />
            {actions.handleSendHints &&
              refImages &&
              refImages.length > 0 && (
                <div className="flex items-center justify-end mt-2">
                  <Button
                    onMouseDown={e => {
                      e.preventDefault();
                      actions.handleSendHints(refImages.filter(img => img));
                    }}
                    disabled={isDownloading || !isStreaming}
                    size="sm"
                    className="rounded-full w-8 h-8 p-0 bg-black hover:bg-gray-800 text-white disabled:opacity-50 disabled:cursor-not-allowed"
                    title={
                      !isStreaming
                        ? "Start streaming to send hints"
                        : "Submit all reference images"
                    }
                  >
                    <ArrowUp className="h-4 w-4" />
                  </Button>
                </div>
              )}
          </div>
        )}

        {/* FFLF Extension Frames - only show when VACE is enabled */}
        {vaceEnabled && (
          <ExtensionFramesSection
            firstFrameImage={firstFrameImage}
            lastFrameImage={lastFrameImage}
            extensionMode={extensionMode as ExtensionMode}
            isDownloading={isDownloading}
            isStreaming={isStreaming}
            onFirstFrameImageChange={actions.handleFirstFrameImageChange}
            onLastFrameImageChange={actions.handleLastFrameImageChange}
            onExtensionModeChange={actions.handleExtensionModeChange}
            onSendExtensionFrames={actions.handleSendExtensionFrames}
          />
        )}

        <div>
          {(() => {
            const isEditMode = selectedTimelinePrompt && isVideoPaused;

            if (pipeline?.supportsPrompts === false) {
              return null;
            }

            return (
              <div>
                <div className="flex items-center justify-between mb-2">
                  <h3 className="text-sm font-medium">Prompts</h3>
                  {isEditMode && (
                    <Badge variant="secondary" className="text-xs">
                      Editing
                    </Badge>
                  )}
                </div>

                {selectedTimelinePrompt ? (
                  <TimelinePromptEditor
                    prompt={selectedTimelinePrompt}
                    onPromptUpdate={actions.handleTimelinePromptUpdate}
                    disabled={false}
                    interpolationMethod={interpolationMethod}
                    onInterpolationMethodChange={onInterpolationMethodChange}
                    promptIndex={timelinePrompts.findIndex(
                      p => p.id === selectedTimelinePrompt.id
                    )}
                    defaultTemporalInterpolationMethod={
                      pipeline?.defaultTemporalInterpolationMethod
                    }
                    defaultSpatialInterpolationMethod={
                      pipeline?.defaultSpatialInterpolationMethod
                    }
                  />
                ) : (
                  <PromptInput
                    prompts={prompts}
                    onPromptsChange={onPromptsChange}
                    onPromptsSubmit={onPromptsChange}
                    onTransitionSubmit={actions.handleTransitionSubmit}
                    disabled={
                      (isTimelinePlaying &&
                        !isVideoPaused &&
                        !isAtEndOfTimeline()) ||
                      (!selectedTimelinePrompt &&
                        isVideoPaused &&
                        !isAtEndOfTimeline())
                    }
                    interpolationMethod={interpolationMethod}
                    onInterpolationMethodChange={onInterpolationMethodChange}
                    temporalInterpolationMethod={temporalInterpolationMethod}
                    onTemporalInterpolationMethodChange={
                      onTemporalInterpolationMethodChange
                    }
                    isLive={isLive}
                    onLivePromptSubmit={actions.handleLivePromptSubmit}
                    isStreaming={isStreaming}
                    transitionSteps={transitionSteps}
                    onTransitionStepsChange={onTransitionStepsChange}
                    timelinePrompts={timelinePrompts}
                    defaultTemporalInterpolationMethod={
                      pipeline?.defaultTemporalInterpolationMethod
                    }
                    defaultSpatialInterpolationMethod={
                      pipeline?.defaultSpatialInterpolationMethod
                    }
                  />
                )}
              </div>
            );
          })()}
        </div>

        {/* Schema-driven input fields */}
        {configSchema &&
          (() => {
            const parsedInputFields = parseInputFields(configSchema, inputMode);
            if (parsedInputFields.length === 0) return null;
            const enumValuesByRef: Record<string, string[]> = {};
            if (configSchema?.$defs) {
              for (const [defName, def] of Object.entries(
                configSchema.$defs as Record<string, { enum?: unknown[] }>
              )) {
                if (def?.enum && Array.isArray(def.enum)) {
                  enumValuesByRef[defName] = def.enum as string[];
                }
              }
            }
            return (
              <div className="space-y-2">
                {parsedInputFields.map(({ key, prop, ui, fieldType }) => {
                  const comp = ui.component;
                  const isRuntimeParam = ui.is_load_param === false;
                  const disabled =
                    (isStreaming && !isRuntimeParam) || isPipelineLoading;
                  const value = schemaFieldOverrides?.[key] ?? prop.default;
                  const setValue = (v: unknown) =>
                    actions.handleSchemaFieldOverrideChange?.(
                      key,
                      v,
                      isRuntimeParam
                    );
                  if (comp === "image") {
                    const path = value == null ? null : String(value);
                    return (
                      <div key={key} className="space-y-1">
                        {ui.label != null && (
                          <span className="text-xs text-muted-foreground">
                            {ui.label}
                          </span>
                        )}
                        <ImageManager
                          images={path ? [path] : []}
                          onImagesChange={images =>
                            actions.handleSchemaFieldOverrideChange?.(
                              key,
                              images[0] ?? null,
                              isRuntimeParam
                            )
                          }
                          disabled={disabled}
                          maxImages={1}
                          hideLabel
                        />
                      </div>
                    );
                  }
                  if (
                    comp &&
                    (COMPLEX_COMPONENTS as readonly string[]).includes(comp)
                  ) {
                    return null;
                  }
                  const enumValues =
                    fieldType === "enum" && typeof prop.$ref === "string"
                      ? enumValuesByRef[prop.$ref.split("/").pop() ?? ""]
                      : undefined;
                  const primitiveType: PrimitiveFieldType | undefined =
                    typeof fieldType === "string" &&
                    !(COMPLEX_COMPONENTS as readonly string[]).includes(
                      fieldType
                    )
                      ? (fieldType as PrimitiveFieldType)
                      : undefined;
                  return (
                    <SchemaPrimitiveField
                      key={key}
                      fieldKey={key}
                      prop={prop}
                      value={value}
                      onChange={setValue}
                      disabled={disabled}
                      label={ui.label}
                      fieldType={primitiveType}
                      enumValues={enumValues}
                    />
                  );
                })}
              </div>
            );
          })()}
      </CardContent>
    </Card>
  );
}
