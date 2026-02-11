import { describe, it, expect, beforeEach } from "vitest";
import { useAppStore } from "../index";

// Reset store state before each test
beforeEach(() => {
  useAppStore.setState(useAppStore.getInitialState());
});

describe("settingsSlice", () => {
  it("has correct default settings", () => {
    const { settings } = useAppStore.getState();
    expect(settings.pipelineId).toBe("longlive");
    expect(settings.resolution).toEqual({ height: 320, width: 576 });
    expect(settings.manageCache).toBe(true);
    expect(settings.inputMode).toBe("text");
  });

  it("updates settings with partial patch", () => {
    useAppStore.getState().updateSettings({ pipelineId: "newpipe" });
    expect(useAppStore.getState().settings.pipelineId).toBe("newpipe");
    // Other settings unchanged
    expect(useAppStore.getState().settings.manageCache).toBe(true);
  });

  it("merges multiple setting updates", () => {
    useAppStore.getState().updateSettings({ noiseScale: 0.5 });
    useAppStore.getState().updateSettings({ noiseController: true });
    const { settings } = useAppStore.getState();
    expect(settings.noiseScale).toBe(0.5);
    expect(settings.noiseController).toBe(true);
  });
});

describe("promptSlice", () => {
  it("has default prompt items", () => {
    const { promptItems } = useAppStore.getState();
    expect(promptItems).toEqual([{ text: "", weight: 100 }]);
  });

  it("sets prompt items", () => {
    useAppStore.getState().setPromptItems([
      { text: "hello", weight: 50 },
      { text: "world", weight: 50 },
    ]);
    expect(useAppStore.getState().promptItems).toHaveLength(2);
  });

  it("sets interpolation method", () => {
    useAppStore.getState().setInterpolationMethod("slerp");
    expect(useAppStore.getState().interpolationMethod).toBe("slerp");
  });

  it("sets transition steps", () => {
    useAppStore.getState().setTransitionSteps(8);
    expect(useAppStore.getState().transitionSteps).toBe(8);
  });
});

describe("timelineSlice", () => {
  it("defaults to not playing", () => {
    expect(useAppStore.getState().isTimelinePlaying).toBe(false);
    expect(useAppStore.getState().isLive).toBe(false);
  });

  it("sets timeline prompts", () => {
    const prompts = [
      { id: "1", text: "test", startTime: 0, endTime: 5 },
    ];
    useAppStore.getState().setTimelinePrompts(prompts);
    expect(useAppStore.getState().timelinePrompts).toEqual(prompts);
  });

  it("sets timeline current time", () => {
    useAppStore.getState().setTimelineCurrentTime(3.5);
    expect(useAppStore.getState().timelineCurrentTime).toBe(3.5);
  });
});

describe("videoSlice", () => {
  it("defaults to fit mode", () => {
    expect(useAppStore.getState().videoScaleMode).toBe("fit");
  });

  it("sets video scale mode", () => {
    useAppStore.getState().setVideoScaleMode("native");
    expect(useAppStore.getState().videoScaleMode).toBe("native");
  });

  it("sets recording state", () => {
    useAppStore.getState().setIsRecording(true);
    expect(useAppStore.getState().isRecording).toBe(true);
  });

  it("sets custom video resolution", () => {
    useAppStore.getState().setCustomVideoResolution({ width: 1920, height: 1080 });
    expect(useAppStore.getState().customVideoResolution).toEqual({
      width: 1920,
      height: 1080,
    });
  });
});

describe("downloadSlice", () => {
  it("defaults to not downloading", () => {
    expect(useAppStore.getState().isDownloading).toBe(false);
    expect(useAppStore.getState().showDownloadDialog).toBe(false);
  });

  it("manages download state", () => {
    useAppStore.getState().setIsDownloading(true);
    useAppStore.getState().setCurrentDownloadPipeline("longlive");
    useAppStore.getState().setDownloadProgress({
      is_downloading: true,
      percentage: 50,
      current_artifact: "model.safetensors",
    });

    const state = useAppStore.getState();
    expect(state.isDownloading).toBe(true);
    expect(state.currentDownloadPipeline).toBe("longlive");
    expect(state.downloadProgress?.percentage).toBe(50);
  });

  it("sets pipelines needing models", () => {
    useAppStore.getState().setPipelinesNeedingModels(["a", "b"]);
    expect(useAppStore.getState().pipelinesNeedingModels).toEqual(["a", "b"]);
  });
});

describe("uiSlice", () => {
  it("sets open settings tab", () => {
    useAppStore.getState().setOpenSettingsTab("advanced");
    expect(useAppStore.getState().openSettingsTab).toBe("advanced");
  });

  it("sets cloud connecting state", () => {
    useAppStore.getState().setIsCloudConnecting(true);
    expect(useAppStore.getState().isCloudConnecting).toBe(true);
  });
});

describe("cross-slice independence", () => {
  it("updating one slice does not affect others", () => {
    useAppStore.getState().updateSettings({ pipelineId: "changed" });
    useAppStore.getState().setIsRecording(true);

    const state = useAppStore.getState();
    expect(state.settings.pipelineId).toBe("changed");
    expect(state.isRecording).toBe(true);
    expect(state.promptItems).toEqual([{ text: "", weight: 100 }]); // unchanged
    expect(state.isDownloading).toBe(false); // unchanged
  });
});
