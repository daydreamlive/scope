import { useEffect, useRef, useCallback, useState, useMemo } from "react";
import { Handle, Position } from "@xyflow/react";
import type { NodeProps, Node } from "@xyflow/react";
import type { FlowNodeData } from "../../../lib/graphUtils";
import {
  getInputSourceSources,
  getInputSourceResolution,
  type DiscoveredSource,
  type InputSourceType,
} from "../../../lib/api";
import { useNodeData } from "../hooks/node/useNodeData";
import { useNodeCollapse } from "../hooks/node/useNodeCollapse";
import {
  NodeCard,
  NodeHeader,
  NodeParamRow,
  NodePillSelect,
  NodePillInput,
  NodePillSearchableSelect,
  collapsedHandleStyle,
} from "../ui";

type SourceNodeType = Node<FlowNodeData, "source">;

const HEADER_H = 28;
const BODY_PAD = 6;
const SELECT_ROW_H = 20;

// Frontend-only modes that produce a local MediaStream (not backed by an
// InputSource). Every other value in the dropdown is a server-side
// source_id from /api/v1/input-sources.
const LOCAL_MODES = [
  { value: "video", label: "File" },
  { value: "camera", label: "Camera" },
];

// Server-side modes that already have dedicated UI blocks below. Anything
// else (youtube and future plugin sources) falls through to the generic
// renderer driven by the source's declared ``params``.
const KNOWN_SERVER_MODES = new Set(["spout", "ndi", "syphon"]);

// "video_file" is represented in the dropdown as the frontend-only "video"
// mode, so we hide it to avoid a duplicate entry.
const HIDDEN_SERVER_MODES = new Set(["video_file"]);

type InputKind = "url" | "discovered" | "asset";

function getSourceNameParam(src: InputSourceType | undefined) {
  return src?.params.find(p => p.name === "source_name");
}

function getInputKind(src: InputSourceType | undefined): InputKind | null {
  const p = getSourceNameParam(src);
  const ui = p?.ui as Record<string, unknown> | undefined;
  const kind = ui?.input_kind;
  if (kind === "url" || kind === "discovered" || kind === "asset") return kind;
  return null;
}

export function SourceNode({ id, data, selected }: NodeProps<SourceNodeType>) {
  const { updateData } = useNodeData(id);
  const { collapsed, toggleCollapse } = useNodeCollapse();
  const sourceMode = data.sourceMode || "video";
  const sourceName = data.sourceName || "";
  const sourceFlipVertical = data.sourceFlipVertical === true;
  const localStream = data.localStream as MediaStream | null | undefined;
  const onVideoFileUpload = data.onVideoFileUpload as
    | ((file: File) => Promise<boolean>)
    | undefined;
  const onSourceModeChange = data.onSourceModeChange as
    | ((mode: string) => void)
    | undefined;
  const spoutAvailable = data.spoutAvailable ?? false;
  const ndiAvailable = data.ndiAvailable ?? false;
  const syphonAvailable = data.syphonAvailable ?? false;
  const availableInputSourcesRaw = data.availableInputSources as
    | InputSourceType[]
    | undefined;
  const availableInputSources = useMemo<InputSourceType[]>(
    () => availableInputSourcesRaw ?? [],
    [availableInputSourcesRaw]
  );
  const onSpoutSourceChange = data.onSpoutSourceChange as
    | ((name: string) => void)
    | undefined;
  const onNdiSourceChange = data.onNdiSourceChange as
    | ((identifier: string) => void)
    | undefined;
  const onSyphonSourceChange = data.onSyphonSourceChange as
    | ((identifier: string) => void)
    | undefined;
  const onCycleSampleVideo = data.onCycleSampleVideo as
    | (() => void)
    | undefined;
  const onInitSampleVideo = data.onInitSampleVideo as (() => void) | undefined;
  const videoRef = useRef<HTMLVideoElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [ndiSources, setNdiSources] = useState<DiscoveredSource[]>([]);
  const [isDiscoveringNdi, setIsDiscoveringNdi] = useState(false);
  const [syphonSources, setSyphonSources] = useState<DiscoveredSource[]>([]);
  const [isDiscoveringSyphon, setIsDiscoveringSyphon] = useState(false);

  // Generic discovery + probe state for plugin-declared sources
  const [pluginSources, setPluginSources] = useState<DiscoveredSource[]>([]);
  const [isDiscoveringPlugin, setIsDiscoveringPlugin] = useState(false);
  const [probeStatus, setProbeStatus] = useState<{
    state: "idle" | "probing" | "ok" | "error";
    message: string;
  }>({ state: "idle", message: "" });

  const currentSourceDef = useMemo(
    () => availableInputSources.find(s => s.source_id === sourceMode),
    [availableInputSources, sourceMode]
  );
  const currentSourceNameParam = getSourceNameParam(currentSourceDef);
  const currentInputKind = getInputKind(currentSourceDef);

  useEffect(() => {
    if (videoRef.current && localStream instanceof MediaStream) {
      videoRef.current.srcObject = localStream;
    }
  }, [localStream]);

  // Auto-load the first sample video (test.mp4) when in file mode without a
  // stream. Without this, freshly added source nodes — or nodes loaded after
  // the global useVideoSource fallback was cleared (e.g. switching to
  // Spout/NDI/Syphon globally) — would show "No video loaded" until the user
  // manually clicked the cycle button. The init handler is idempotent.
  useEffect(() => {
    if (sourceMode === "video" && !localStream && onInitSampleVideo) {
      onInitSampleVideo();
    }
  }, [sourceMode, localStream, onInitSampleVideo]);

  // Discover NDI sources
  useEffect(() => {
    if (sourceMode === "ndi" && ndiAvailable && ndiSources.length === 0) {
      discoverNdiSources();
    }
  }, [sourceMode, ndiAvailable]); // eslint-disable-line react-hooks/exhaustive-deps

  const discoverNdiSources = useCallback(async () => {
    if (!ndiAvailable) return;
    setIsDiscoveringNdi(true);
    try {
      const result = await getInputSourceSources("ndi", 5000);
      setNdiSources(result.sources);
    } catch (e) {
      console.error("Failed to discover NDI sources:", e);
      setNdiSources([]);
    } finally {
      setIsDiscoveringNdi(false);
    }
  }, [ndiAvailable]);

  // Discover Syphon sources
  useEffect(() => {
    if (
      sourceMode === "syphon" &&
      syphonAvailable &&
      syphonSources.length === 0
    ) {
      discoverSyphonSources();
    }
  }, [sourceMode, syphonAvailable]); // eslint-disable-line react-hooks/exhaustive-deps

  const discoverSyphonSources = useCallback(async () => {
    if (!syphonAvailable) return;
    setIsDiscoveringSyphon(true);
    try {
      const result = await getInputSourceSources("syphon");
      setSyphonSources(result.sources);
    } catch (e) {
      console.error("Failed to discover Syphon sources:", e);
      setSyphonSources([]);
    } finally {
      setIsDiscoveringSyphon(false);
    }
  }, [syphonAvailable]);

  // Generic discovery for plugin-declared "discovered" sources.
  const discoverPluginSources = useCallback(async () => {
    if (!currentSourceDef) return;
    setIsDiscoveringPlugin(true);
    try {
      const result = await getInputSourceSources(currentSourceDef.source_id);
      setPluginSources(result.sources);
    } catch (e) {
      console.error(
        `Failed to discover ${currentSourceDef.source_id} sources:`,
        e
      );
      setPluginSources([]);
    } finally {
      setIsDiscoveringPlugin(false);
    }
  }, [currentSourceDef]);

  // Auto-discover when switching to a plugin "discovered" source.
  useEffect(() => {
    if (
      currentSourceDef &&
      !KNOWN_SERVER_MODES.has(sourceMode) &&
      currentInputKind === "discovered"
    ) {
      discoverPluginSources();
    }
  }, [sourceMode, currentSourceDef, currentInputKind, discoverPluginSources]);

  // Probe URL-based plugin sources (debounced).
  useEffect(() => {
    if (KNOWN_SERVER_MODES.has(sourceMode)) return;
    if (!currentSourceDef) return;
    const ui = currentSourceNameParam?.ui as
      | Record<string, unknown>
      | undefined;
    if (!ui?.probe || currentInputKind !== "url") return;

    if (!sourceName.trim()) {
      setProbeStatus({ state: "idle", message: "" });
      return;
    }

    setProbeStatus({ state: "probing", message: "Checking…" });
    const handle = setTimeout(async () => {
      try {
        const { width, height } = await getInputSourceResolution(
          currentSourceDef.source_id,
          sourceName,
          15000
        );
        setProbeStatus({
          state: "ok",
          message: `Ready (${width}×${height})`,
        });
      } catch (e) {
        const msg =
          e instanceof Error ? e.message : "Could not read this source";
        // Map HTTP errors to friendly messages
        let friendly = msg;
        if (/\b400\b/.test(msg)) friendly = "Invalid URL";
        else if (/\b404\b/.test(msg)) friendly = "Video is unavailable";
        else if (/\b408\b/.test(msg)) friendly = "Could not read this source";
        setProbeStatus({ state: "error", message: friendly });
      }
    }, 600);
    return () => clearTimeout(handle);
  }, [
    sourceName,
    sourceMode,
    currentSourceDef,
    currentSourceNameParam,
    currentInputKind,
  ]);

  const handleSourceModeChange = (newMode: string) => {
    updateData({
      sourceMode: newMode,
      // Reset source_name when leaving any server-side mode that used it
      ...(newMode === "video" || newMode === "camera"
        ? { sourceName: undefined }
        : {}),
    });
    setProbeStatus({ state: "idle", message: "" });
    onSourceModeChange?.(newMode);
  };

  const handleSpoutNameChange = (value: string | number) => {
    const name = String(value);
    updateData({ sourceName: name });
    onSpoutSourceChange?.(name);
  };

  const handleNdiSourceChange_ = (identifier: string) => {
    updateData({ sourceName: identifier });
    onNdiSourceChange?.(identifier);
  };

  const handleSyphonSourceChange_ = (identifier: string) => {
    updateData({ sourceName: identifier });
    onSyphonSourceChange?.(identifier);
  };

  const handleSyphonFlipVerticalChange = (
    e: React.ChangeEvent<HTMLInputElement>
  ) => {
    updateData({ sourceFlipVertical: e.target.checked });
  };

  const handlePluginSourceNameChange = (value: string | number) => {
    updateData({ sourceName: String(value) });
  };

  const handleFileClick = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  const handleFileChange = useCallback(
    async (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (!file || !onVideoFileUpload) return;
      await onVideoFileUpload(file);
      if (fileInputRef.current) fileInputRef.current.value = "";
    },
    [onVideoFileUpload]
  );

  const showPreview = sourceMode === "video" || sourceMode === "camera";
  const showFilePicker = sourceMode === "video";
  const handleY = HEADER_H + BODY_PAD + SELECT_ROW_H / 2;

  // Dropdown options: frontend-only modes + all available server-side
  // sources (excluding hidden ones like video_file). Availability is
  // honored for the three known server-side modes; plugin sources are
  // included if their backend reports ``available``.
  const sourceModeOptions = useMemo(() => {
    const opts: { value: string; label: string }[] = [...LOCAL_MODES];
    for (const s of availableInputSources) {
      if (HIDDEN_SERVER_MODES.has(s.source_id)) continue;
      if (!s.available) {
        if (s.source_id === "spout" && !spoutAvailable) continue;
        if (s.source_id === "ndi" && !ndiAvailable) continue;
        if (s.source_id === "syphon" && !syphonAvailable) continue;
        continue;
      }
      // Preserve the legacy capitalized labels for the three known modes
      let label = s.source_name;
      if (s.source_id === "spout") label = "Spout";
      if (s.source_id === "ndi") label = "NDI";
      if (s.source_id === "syphon") label = "Syphon";
      opts.push({ value: s.source_id, label });
    }
    return opts;
  }, [availableInputSources, spoutAvailable, ndiAvailable, syphonAvailable]);

  const ndiOptions = ndiSources.map(s => ({
    value: s.identifier,
    label: s.name,
  }));
  const syphonOptions = syphonSources.map(s => ({
    value: s.identifier,
    label: s.name,
  }));
  const pluginOptions = pluginSources.map(s => ({
    value: s.identifier,
    label: s.name,
  }));

  const isPluginMode =
    !showPreview &&
    !KNOWN_SERVER_MODES.has(sourceMode) &&
    currentSourceDef !== undefined;

  return (
    <NodeCard selected={selected} collapsed={collapsed} autoMinHeight={false}>
      <NodeHeader
        title={data.customTitle || "Source"}
        onTitleChange={newTitle => updateData({ customTitle: newTitle })}
        collapsed={collapsed}
        onCollapseToggle={toggleCollapse}
      />
      {!collapsed && (
        <div className="px-2 py-1.5 flex flex-col gap-1.5 flex-1 min-h-0">
          <div className="px-2">
            <NodeParamRow label="Source" className="justify-start gap-2">
              <NodePillSelect
                value={sourceMode}
                onChange={handleSourceModeChange}
                options={sourceModeOptions}
              />
            </NodeParamRow>
          </div>

          {sourceMode === "spout" && (
            <div className="px-2">
              <NodeParamRow label="Sender">
                <NodePillInput
                  type="text"
                  value={sourceName}
                  onChange={handleSpoutNameChange}
                  placeholder="TDSyphonSpoutOut"
                  disabled={!spoutAvailable}
                />
              </NodeParamRow>
            </div>
          )}

          {sourceMode === "ndi" && (
            <div className="px-2 flex flex-col gap-1.5">
              <NodeParamRow label="Source">
                <div className="flex items-center gap-1">
                  <NodePillSearchableSelect
                    value={sourceName}
                    onChange={handleNdiSourceChange_}
                    options={ndiOptions}
                    placeholder={
                      isDiscoveringNdi
                        ? "Discovering..."
                        : ndiOptions.length === 0
                          ? "No sources"
                          : "Select source"
                    }
                    disabled={isDiscoveringNdi || !ndiAvailable}
                    className="flex-1"
                  />
                  <button
                    type="button"
                    onClick={discoverNdiSources}
                    disabled={isDiscoveringNdi || !ndiAvailable}
                    className="w-5 h-5 flex items-center justify-center text-[#fafafa] hover:text-blue-400 transition-colors disabled:opacity-50"
                    title="Refresh NDI sources"
                  >
                    <svg
                      className={`h-3 w-3 ${isDiscoveringNdi ? "animate-spin" : ""}`}
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    >
                      <path d="M21.5 2v6h-6M2.5 22v-6h6M2 11.5a10 10 0 0 1 18.8-4.3M22 12.5a10 10 0 0 1-18.8 4.2" />
                    </svg>
                  </button>
                </div>
              </NodeParamRow>
            </div>
          )}

          {sourceMode === "syphon" && (
            <div className="px-2 flex flex-col gap-1.5">
              <NodeParamRow label="Source" className="justify-start gap-2">
                <div className="flex items-center gap-1">
                  <NodePillSearchableSelect
                    value={sourceName}
                    onChange={handleSyphonSourceChange_}
                    options={syphonOptions}
                    placeholder={
                      isDiscoveringSyphon
                        ? "Discovering..."
                        : syphonOptions.length === 0
                          ? "No sources"
                          : "Select source"
                    }
                    disabled={isDiscoveringSyphon || !syphonAvailable}
                    className="flex-1"
                  />
                  <button
                    type="button"
                    onClick={discoverSyphonSources}
                    disabled={isDiscoveringSyphon || !syphonAvailable}
                    className="w-5 h-5 flex items-center justify-center text-[#fafafa] hover:text-blue-400 transition-colors disabled:opacity-50"
                    title="Refresh Syphon sources"
                  >
                    <svg
                      className={`h-3 w-3 ${isDiscoveringSyphon ? "animate-spin" : ""}`}
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    >
                      <path d="M21.5 2v6h-6M2.5 22v-6h6M2 11.5a10 10 0 0 1 18.8-4.3M22 12.5a10 10 0 0 1-18.8 4.2" />
                    </svg>
                  </button>
                </div>
              </NodeParamRow>
              <NodeParamRow label="Flip Y" className="justify-start gap-2">
                <label className="flex items-center gap-2 text-[10px] text-[#fafafa]">
                  <input
                    type="checkbox"
                    checked={sourceFlipVertical}
                    onChange={handleSyphonFlipVerticalChange}
                    className="h-3 w-3 accent-white"
                    disabled={data.isStreaming === true}
                  />
                  <span className="text-left text-[#8c8c8d]">
                    Fix flipped inputs
                  </span>
                </label>
              </NodeParamRow>
            </div>
          )}

          {isPluginMode && (
            <div className="px-2 flex flex-col gap-1.5">
              {currentInputKind === "url" && (
                <>
                  <NodeParamRow label="URL">
                    <NodePillInput
                      type="text"
                      value={sourceName}
                      onChange={handlePluginSourceNameChange}
                      placeholder={
                        (currentSourceNameParam?.ui?.[
                          "placeholder"
                        ] as string) || ""
                      }
                    />
                  </NodeParamRow>
                  {probeStatus.state !== "idle" && (
                    <div
                      className={`text-[10px] px-2 ${
                        probeStatus.state === "error"
                          ? "text-red-400"
                          : probeStatus.state === "ok"
                            ? "text-green-400"
                            : "text-[#8c8c8d]"
                      }`}
                    >
                      {probeStatus.message}
                    </div>
                  )}
                </>
              )}
              {currentInputKind === "discovered" && (
                <NodeParamRow label="Source">
                  <div className="flex items-center gap-1">
                    <NodePillSearchableSelect
                      value={sourceName}
                      onChange={handlePluginSourceNameChange}
                      options={pluginOptions}
                      placeholder={
                        isDiscoveringPlugin
                          ? "Discovering..."
                          : pluginOptions.length === 0
                            ? "No sources"
                            : "Select source"
                      }
                      disabled={isDiscoveringPlugin}
                      className="flex-1"
                    />
                    <button
                      type="button"
                      onClick={discoverPluginSources}
                      disabled={isDiscoveringPlugin}
                      className="w-5 h-5 flex items-center justify-center text-[#fafafa] hover:text-blue-400 transition-colors disabled:opacity-50"
                      title={`Refresh ${currentSourceDef?.source_name ?? ""} sources`}
                    >
                      <svg
                        className={`h-3 w-3 ${isDiscoveringPlugin ? "animate-spin" : ""}`}
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      >
                        <path d="M21.5 2v6h-6M2.5 22v-6h6M2 11.5a10 10 0 0 1 18.8-4.3M22 12.5a10 10 0 0 1-18.8 4.2" />
                      </svg>
                    </button>
                  </div>
                </NodeParamRow>
              )}
              {currentSourceNameParam?.ui?.["help"] ? (
                <div className="px-2 text-[10px] text-[#8c8c8d]">
                  {String(currentSourceNameParam.ui["help"])}
                </div>
              ) : null}
            </div>
          )}

          {showPreview && (
            <div className="relative rounded-md overflow-hidden bg-black/50 flex-1 min-h-[60px]">
              {localStream ? (
                <video
                  ref={videoRef}
                  className="w-full h-full object-cover"
                  autoPlay
                  muted
                  playsInline
                />
              ) : (
                <div className="flex items-center justify-center h-full text-[10px] text-[#8c8c8d]">
                  {sourceMode === "camera"
                    ? "Camera preview"
                    : "No video loaded"}
                </div>
              )}
              {showFilePicker && (
                <div className="absolute bottom-1 right-1 flex gap-0.5">
                  <button
                    type="button"
                    onClick={() => onCycleSampleVideo?.()}
                    className="w-5 h-5 flex items-center justify-center bg-[#2a2a2a]/80 hover:bg-[#2a2a2a] text-[#fafafa] rounded border border-[rgba(119,119,119,0.35)] transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
                    title="Cycle sample video"
                  >
                    <svg
                      className="h-2.5 w-2.5"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    >
                      <path d="M16 3l4 4-4 4" />
                      <path d="M20 7H4" />
                      <path d="M8 21l-4-4 4-4" />
                      <path d="M4 17h16" />
                    </svg>
                  </button>
                  <button
                    type="button"
                    onClick={handleFileClick}
                    className="w-5 h-5 flex items-center justify-center bg-[#2a2a2a]/80 hover:bg-[#2a2a2a] text-[#fafafa] rounded border border-[rgba(119,119,119,0.35)] transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
                    title="Upload video file"
                  >
                    <svg
                      className="h-2.5 w-2.5"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    >
                      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                      <polyline points="17 8 12 3 7 8" />
                      <line x1="12" y1="3" x2="12" y2="15" />
                    </svg>
                  </button>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="video/*"
                    className="hidden"
                    onChange={handleFileChange}
                  />
                </div>
              )}
            </div>
          )}

          {!showPreview &&
            sourceMode !== "spout" &&
            sourceMode !== "ndi" &&
            sourceMode !== "syphon" &&
            !isPluginMode && (
              <div className="flex items-center justify-center rounded-md bg-black/30 text-[10px] text-[#8c8c8d] flex-1 min-h-[40px]">
                Waiting for input...
              </div>
            )}
        </div>
      )}
      <Handle
        type="source"
        position={Position.Right}
        id="stream:video"
        className="!w-2.5 !h-2.5 !border-0"
        style={
          collapsed
            ? collapsedHandleStyle("right")
            : { top: handleY, right: 0, backgroundColor: "#ffffff" }
        }
      />
    </NodeCard>
  );
}
