import { describe, it, expect } from "vitest";
import {
  getResolutionScaleFactor,
  adjustResolutionForPipeline,
} from "../utils";

describe("getResolutionScaleFactor", () => {
  it("returns 16 for longlive pipeline", () => {
    expect(getResolutionScaleFactor("longlive")).toBe(16);
  });

  it("returns 16 for streamdiffusionv2 pipeline", () => {
    expect(getResolutionScaleFactor("streamdiffusionv2")).toBe(16);
  });

  it("returns null for unknown pipeline", () => {
    expect(getResolutionScaleFactor("unknown-pipeline")).toBeNull();
  });
});

describe("adjustResolutionForPipeline", () => {
  it("returns unchanged resolution for unknown pipeline", () => {
    const resolution = { height: 513, width: 513 };
    const result = adjustResolutionForPipeline("unknown", resolution);
    expect(result.wasAdjusted).toBe(false);
    expect(result.resolution).toBe(resolution);
  });

  it("adjusts resolution to be divisible by 16 for longlive", () => {
    const result = adjustResolutionForPipeline("longlive", {
      height: 321,
      width: 577,
    });
    expect(result.wasAdjusted).toBe(true);
    expect(result.resolution.height % 16).toBe(0);
    expect(result.resolution.width % 16).toBe(0);
  });

  it("rounds to nearest multiple of 16", () => {
    const result = adjustResolutionForPipeline("longlive", {
      height: 320,
      width: 576,
    });
    expect(result.wasAdjusted).toBe(false);
    expect(result.resolution).toEqual({ height: 320, width: 576 });
  });

  it("rounds 328 down to 320 (nearest 16)", () => {
    const result = adjustResolutionForPipeline("longlive", {
      height: 328,
      width: 576,
    });
    // Math.round(328/16)*16 = Math.round(20.5)*16 = 21*16 = 336
    expect(result.resolution.height).toBe(336);
    expect(result.wasAdjusted).toBe(true);
  });
});
