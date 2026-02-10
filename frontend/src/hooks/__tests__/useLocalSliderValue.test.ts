import { describe, it, expect, vi } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useLocalSliderValue } from "../useLocalSliderValue";

describe("useLocalSliderValue", () => {
  it("initializes with the provided value", () => {
    const { result } = renderHook(() => useLocalSliderValue(0.5));
    expect(result.current.localValue).toBe(0.5);
  });

  it("updates local value on handleValueChange", () => {
    const { result } = renderHook(() => useLocalSliderValue(0.5));

    act(() => {
      result.current.handleValueChange(0.8);
    });

    expect(result.current.localValue).toBe(0.8);
  });

  it("calls onChange on handleValueCommit", () => {
    const onChange = vi.fn();
    const { result } = renderHook(() => useLocalSliderValue(0.5, onChange));

    act(() => {
      result.current.handleValueCommit(0.8);
    });

    expect(onChange).toHaveBeenCalledWith(0.8);
  });

  it("syncs with external value changes", () => {
    const { result, rerender } = renderHook(
      ({ value }) => useLocalSliderValue(value),
      { initialProps: { value: 0.5 } }
    );

    expect(result.current.localValue).toBe(0.5);

    rerender({ value: 0.9 });
    expect(result.current.localValue).toBe(0.9);
  });

  it("formats value to specified decimal places", () => {
    const { result } = renderHook(() => useLocalSliderValue(0.5, undefined, 1));
    expect(result.current.formatValue(0.456)).toBe(0.5);
  });

  it("defaults to 2 decimal places", () => {
    const { result } = renderHook(() => useLocalSliderValue(0.5));
    expect(result.current.formatValue(0.4567)).toBe(0.46);
  });
});
