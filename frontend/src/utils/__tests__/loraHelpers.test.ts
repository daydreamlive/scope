import { describe, it, expect, vi } from "vitest";
import { sendLoRAScaleUpdates } from "../loraHelpers";

describe("sendLoRAScaleUpdates", () => {
  it("sends scale updates for loaded adapters only", () => {
    const sendUpdate = vi.fn();
    const loras = [
      { path: "lora-a.safetensors", scale: 0.8 },
      { path: "lora-b.safetensors", scale: 0.5 },
      { path: "lora-c.safetensors", scale: 1.0 },
    ];
    const loaded = [
      { path: "lora-a.safetensors", scale: 1.0 },
      { path: "lora-c.safetensors", scale: 1.0 },
    ];

    sendLoRAScaleUpdates(loras, loaded, sendUpdate);

    expect(sendUpdate).toHaveBeenCalledWith({
      lora_scales: [
        { path: "lora-a.safetensors", scale: 0.8 },
        { path: "lora-c.safetensors", scale: 1.0 },
      ],
    });
  });

  it("does not call sendUpdate when loras is undefined", () => {
    const sendUpdate = vi.fn();
    sendLoRAScaleUpdates(undefined, [{ path: "a", scale: 1 }], sendUpdate);
    expect(sendUpdate).not.toHaveBeenCalled();
  });

  it("does not call sendUpdate when loadedAdapters is undefined", () => {
    const sendUpdate = vi.fn();
    sendLoRAScaleUpdates([{ path: "a", scale: 1 }], undefined, sendUpdate);
    expect(sendUpdate).not.toHaveBeenCalled();
  });

  it("does not call sendUpdate when no loaded adapters match", () => {
    const sendUpdate = vi.fn();
    sendLoRAScaleUpdates(
      [{ path: "a", scale: 1 }],
      [{ path: "b", scale: 1 }],
      sendUpdate
    );
    expect(sendUpdate).not.toHaveBeenCalled();
  });
});
