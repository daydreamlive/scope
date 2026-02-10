import { describe, it, expect } from "vitest";
import {
  redistributeWeightsOnAdd,
  redistributeWeightsOnRemove,
} from "../promptWeights";

describe("redistributeWeightsOnAdd", () => {
  it("distributes weights equally when adding a prompt", () => {
    const existing = [{ text: "hello", weight: 100 }];
    const result = redistributeWeightsOnAdd(existing);

    expect(result).toHaveLength(2);
    expect(result[0].weight).toBe(50);
    expect(result[1].weight).toBe(50);
    expect(result[1].text).toBe("");
  });

  it("redistributes across 3 prompts", () => {
    const existing = [
      { text: "a", weight: 50 },
      { text: "b", weight: 50 },
    ];
    const result = redistributeWeightsOnAdd(existing);

    expect(result).toHaveLength(3);
    result.forEach(p => {
      expect(p.weight).toBeCloseTo(100 / 3);
    });
  });

  it("applies newPromptData to the added prompt", () => {
    const existing = [{ text: "a", weight: 100 }];
    const result = redistributeWeightsOnAdd(existing, { text: "new" });

    expect(result[1].text).toBe("new");
  });
});

describe("redistributeWeightsOnRemove", () => {
  it("redistributes weights proportionally after removal", () => {
    const prompts = [
      { text: "a", weight: 60 },
      { text: "b", weight: 20 },
      { text: "c", weight: 20 },
    ];
    const result = redistributeWeightsOnRemove(prompts, 0);

    expect(result).toHaveLength(2);
    // b and c had equal weights, so they should split 100 equally
    expect(result[0].weight).toBe(50);
    expect(result[1].weight).toBe(50);
  });

  it("handles removal when remaining weights sum to 0", () => {
    const prompts = [
      { text: "a", weight: 100 },
      { text: "b", weight: 0 },
      { text: "c", weight: 0 },
    ];
    const result = redistributeWeightsOnRemove(prompts, 0);

    expect(result).toHaveLength(2);
    // Both had 0 weight, so distribute evenly
    expect(result[0].weight).toBe(50);
    expect(result[1].weight).toBe(50);
  });

  it("total weight sums to 100 after removal", () => {
    const prompts = [
      { text: "a", weight: 30 },
      { text: "b", weight: 45 },
      { text: "c", weight: 25 },
    ];
    const result = redistributeWeightsOnRemove(prompts, 1);
    const total = result.reduce((sum, p) => sum + p.weight, 0);

    expect(total).toBeCloseTo(100);
  });
});
