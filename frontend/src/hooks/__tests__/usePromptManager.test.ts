import { describe, it, expect, vi } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { usePromptManager } from "../usePromptManager";

describe("usePromptManager", () => {
  describe("uncontrolled mode", () => {
    it("initializes with provided prompts", () => {
      const { result } = renderHook(() =>
        usePromptManager({
          initialPrompts: [{ text: "hello", weight: 100 }],
        })
      );

      expect(result.current.prompts).toEqual([{ text: "hello", weight: 100 }]);
    });

    it("adds a prompt with default weight", () => {
      const { result } = renderHook(() =>
        usePromptManager({
          initialPrompts: [{ text: "hello", weight: 100 }],
        })
      );

      act(() => {
        result.current.handleAddPrompt();
      });

      expect(result.current.prompts).toHaveLength(2);
      expect(result.current.prompts[1]).toEqual({ text: "", weight: 100 });
    });

    it("respects maxPrompts limit", () => {
      const { result } = renderHook(() =>
        usePromptManager({
          initialPrompts: [
            { text: "a", weight: 25 },
            { text: "b", weight: 25 },
            { text: "c", weight: 25 },
            { text: "d", weight: 25 },
          ],
          maxPrompts: 4,
        })
      );

      act(() => {
        result.current.handleAddPrompt();
      });

      expect(result.current.prompts).toHaveLength(4);
    });

    it("removes a prompt but keeps at least one", () => {
      const { result } = renderHook(() =>
        usePromptManager({
          initialPrompts: [
            { text: "a", weight: 50 },
            { text: "b", weight: 50 },
          ],
        })
      );

      act(() => {
        result.current.handleRemovePrompt(0);
      });

      expect(result.current.prompts).toHaveLength(1);
      expect(result.current.prompts[0].text).toBe("b");
    });

    it("does not remove the last prompt", () => {
      const { result } = renderHook(() =>
        usePromptManager({
          initialPrompts: [{ text: "only", weight: 100 }],
        })
      );

      act(() => {
        result.current.handleRemovePrompt(0);
      });

      expect(result.current.prompts).toHaveLength(1);
    });

    it("updates prompt text", () => {
      const { result } = renderHook(() =>
        usePromptManager({
          initialPrompts: [{ text: "old", weight: 100 }],
        })
      );

      act(() => {
        result.current.handlePromptTextChange(0, "new");
      });

      expect(result.current.prompts[0].text).toBe("new");
    });
  });

  describe("weight management", () => {
    it("redistributes weights when changing one prompt", () => {
      const { result } = renderHook(() =>
        usePromptManager({
          initialPrompts: [
            { text: "a", weight: 50 },
            { text: "b", weight: 50 },
          ],
        })
      );

      act(() => {
        result.current.handleWeightChange(0, 80);
      });

      expect(result.current.prompts[0].weight).toBe(80);
      expect(result.current.prompts[1].weight).toBe(20);
    });

    it("calculates normalized weights", () => {
      const { result } = renderHook(() =>
        usePromptManager({
          initialPrompts: [
            { text: "a", weight: 75 },
            { text: "b", weight: 25 },
          ],
        })
      );

      expect(result.current.normalizedWeights[0]).toBe(75);
      expect(result.current.normalizedWeights[1]).toBe(25);
      expect(result.current.totalWeight).toBe(100);
    });
  });

  describe("controlled mode", () => {
    it("uses controlled prompts when provided", () => {
      const controlled = [{ text: "controlled", weight: 100 }];
      const { result } = renderHook(() =>
        usePromptManager({ prompts: controlled })
      );

      expect(result.current.prompts).toBe(controlled);
    });

    it("calls onPromptsChange when updating", () => {
      const onChange = vi.fn();
      const controlled = [{ text: "a", weight: 100 }];
      const { result } = renderHook(() =>
        usePromptManager({ prompts: controlled, onPromptsChange: onChange })
      );

      act(() => {
        result.current.handlePromptTextChange(0, "updated");
      });

      expect(onChange).toHaveBeenCalledWith([{ text: "updated", weight: 100 }]);
    });
  });
});
