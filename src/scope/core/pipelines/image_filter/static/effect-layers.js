/**
 * Effect Layers Web Component
 *
 * A custom element that provides a layer-based effect configuration UI
 * with preset support.
 */

// Preset definitions - moved from pipeline.py
const PRESETS = {
  custom: { name: "Custom", params: {} },
  // Cinematic
  cinematic: {
    name: "Cinematic",
    params: { contrast: 0.15, saturation: -0.1, temperature: 0.1, shadows: 0.1, highlights: -0.1, vignette: 0.3, grain: 0.02 }
  },
  blockbuster: {
    name: "Blockbuster",
    params: { contrast: 0.25, saturation: 0.1, temperature: -0.15, tint: 0.05, shadows: 0.15, highlights: -0.05, sharpen: 0.3, vignette: 0.4 }
  },
  noir: {
    name: "Noir",
    params: { saturation: -1.0, contrast: 0.4, gamma: 0.9, shadows: -0.1, highlights: 0.1, vignette: 0.5, grain: 0.05 }
  },
  scifi: {
    name: "Sci-Fi",
    params: { contrast: 0.2, saturation: -0.2, temperature: -0.3, tint: 0.1, highlights: 0.1, sharpen: 0.4, vignette: 0.2 }
  },
  horror: {
    name: "Horror",
    params: { contrast: 0.3, saturation: -0.3, exposure: -0.2, gamma: 0.85, temperature: -0.1, tint: 0.1, shadows: -0.2, vignette: 0.6, grain: 0.04 }
  },
  // Film
  kodachrome: {
    name: "Kodachrome",
    params: { contrast: 0.15, saturation: 0.2, vibrance: 0.15, temperature: 0.15, shadows: 0.1, highlights: -0.05, grain: 0.02 }
  },
  polaroid: {
    name: "Polaroid",
    params: { contrast: -0.1, saturation: -0.15, exposure: 0.1, temperature: 0.2, tint: -0.05, highlights: 0.15, vignette: 0.25, grain: 0.03 }
  },
  vintage_70s: {
    name: "Vintage 70s",
    params: { contrast: -0.1, saturation: -0.2, exposure: 0.05, temperature: 0.25, tint: 0.1, gamma: 1.1, vignette: 0.35, grain: 0.06, sepia: 0.15 }
  },
  cross_process: {
    name: "Cross Process",
    params: { contrast: 0.3, saturation: 0.25, temperature: -0.2, tint: 0.15, shadows: 0.2, highlights: -0.1, vignette: 0.2 }
  },
  // Portrait
  soft_portrait: {
    name: "Soft Portrait",
    params: { contrast: -0.1, saturation: -0.05, exposure: 0.1, highlights: 0.1, blur: 0.5, sharpen: 0.2 }
  },
  warm_portrait: {
    name: "Warm Portrait",
    params: { saturation: 0.05, temperature: 0.2, tint: 0.05, highlights: 0.1, vignette: 0.15 }
  },
  cool_portrait: {
    name: "Cool Portrait",
    params: { saturation: -0.1, temperature: -0.15, tint: -0.05, highlights: 0.15, vignette: 0.15 }
  },
  // Landscape
  vivid_nature: {
    name: "Vivid Nature",
    params: { contrast: 0.15, saturation: 0.3, vibrance: 0.25, highlights: -0.1, shadows: 0.1, sharpen: 0.3 }
  },
  golden_hour: {
    name: "Golden Hour",
    params: { contrast: 0.1, saturation: 0.15, exposure: 0.1, temperature: 0.35, tint: 0.05, highlights: 0.1, vignette: 0.2 }
  },
  misty_morning: {
    name: "Misty Morning",
    params: { contrast: -0.2, saturation: -0.25, exposure: 0.15, gamma: 1.15, temperature: -0.1, highlights: 0.2, blur: 0.3 }
  },
  // Black & White
  bw_high_contrast: {
    name: "B&W High Contrast",
    params: { saturation: -1.0, contrast: 0.5, gamma: 0.85, shadows: -0.15, highlights: 0.15 }
  },
  bw_film_noir: {
    name: "B&W Film Noir",
    params: { saturation: -1.0, contrast: 0.35, gamma: 0.9, vignette: 0.5, grain: 0.04 }
  },
  bw_soft: {
    name: "B&W Soft",
    params: { saturation: -1.0, contrast: -0.1, gamma: 1.1, highlights: 0.1 }
  },
  // Mood
  dreamy: {
    name: "Dreamy",
    params: { contrast: -0.15, saturation: 0.1, exposure: 0.15, gamma: 1.15, highlights: 0.2, blur: 1.0, vignette: 0.2 }
  },
  moody_blue: {
    name: "Moody Blue",
    params: { contrast: 0.15, saturation: -0.15, temperature: -0.3, tint: 0.1, shadows: -0.1, vignette: 0.35 }
  },
  warm_sunset: {
    name: "Warm Sunset",
    params: { contrast: 0.1, saturation: 0.2, exposure: 0.05, temperature: 0.4, tint: 0.1, highlights: 0.1, vignette: 0.25 }
  },
  // Creative
  neon_nights: {
    name: "Neon Nights",
    params: { contrast: 0.35, saturation: 0.5, vibrance: 0.3, temperature: -0.2, tint: 0.2, shadows: -0.1, sharpen: 0.3, vignette: 0.4, chromatic_aberration: 3.0, scanlines: 0.2 }
  },
  retro_gaming: {
    name: "Retro Gaming",
    params: { contrast: 0.2, saturation: 0.3, posterize_levels: 8, pixelate: 4, sharpen: 0.5, scanlines: 0.3 }
  },
  duotone_pop: {
    name: "Duotone Pop",
    params: { contrast: 0.25, saturation: -0.5, temperature: 0.3, tint: -0.2, vignette: 0.2 }
  },
  vhs_tape: {
    name: "VHS Tape",
    params: { contrast: 0.1, saturation: -0.2, temperature: 0.1, blur: 0.5, chromatic_aberration: 5.0, rgb_shift: 0.3, scanlines: 0.4, grain: 0.08, noise: 0.1, vignette: 0.25 }
  },
  glitch_art: {
    name: "Glitch Art",
    params: { contrast: 0.2, saturation: 0.2, glitch: 0.7, chromatic_aberration: 8.0, rgb_shift: 0.5, posterize_levels: 16 }
  },
  cyberpunk: {
    name: "Cyberpunk",
    params: { contrast: 0.4, saturation: 0.3, temperature: -0.3, tint: 0.3, gamma: 0.9, shadows: -0.15, sharpen: 0.4, chromatic_aberration: 4.0, scanlines: 0.15, vignette: 0.5, grain: 0.02 }
  },
  comic_book: {
    name: "Comic Book",
    params: { contrast: 0.4, saturation: 0.3, posterize_levels: 6, edge_detect: 0.3, sharpen: 0.6 }
  },
  sketch: {
    name: "Sketch",
    params: { saturation: -1.0, contrast: 0.3, edge_detect: 0.8, invert: 0.5, gamma: 1.2 }
  },
  thermal: {
    name: "Thermal",
    params: { invert: 1.0, hue_shift: 0.6, saturation: 0.5, contrast: 0.4, posterize_levels: 10, temperature: 0.3 }
  },
};

const AVAILABLE_EFFECTS = [
  // Basic adjustments
  { id: "brightness", name: "Brightness", params: [{ name: "amount", min: -1, max: 1, default: 0, step: 0.01 }] },
  { id: "contrast", name: "Contrast", params: [{ name: "amount", min: -1, max: 1, default: 0, step: 0.01 }] },
  { id: "saturation", name: "Saturation", params: [{ name: "amount", min: -1, max: 1, default: 0, step: 0.01 }] },
  { id: "exposure", name: "Exposure", params: [{ name: "amount", min: -2, max: 2, default: 0, step: 0.01 }] },
  { id: "gamma", name: "Gamma", params: [{ name: "amount", min: 0.2, max: 3, default: 1, step: 0.01 }] },
  // Color
  { id: "hue_shift", name: "Hue Shift", params: [{ name: "amount", min: -1, max: 1, default: 0, step: 0.01 }] },
  { id: "temperature", name: "Temperature", params: [{ name: "amount", min: -1, max: 1, default: 0, step: 0.01 }] },
  { id: "tint", name: "Tint", params: [{ name: "amount", min: -1, max: 1, default: 0, step: 0.01 }] },
  { id: "vibrance", name: "Vibrance", params: [{ name: "amount", min: -1, max: 1, default: 0, step: 0.01 }] },
  // Tone
  { id: "highlights", name: "Highlights", params: [{ name: "amount", min: -1, max: 1, default: 0, step: 0.01 }] },
  { id: "shadows", name: "Shadows", params: [{ name: "amount", min: -1, max: 1, default: 0, step: 0.01 }] },
  // Detail
  { id: "sharpen", name: "Sharpen", params: [{ name: "amount", min: 0, max: 2, default: 0.5, step: 0.01 }] },
  { id: "blur", name: "Blur", params: [{ name: "amount", min: 0, max: 10, default: 2, step: 0.1 }] },
  // Stylistic
  { id: "vignette", name: "Vignette", params: [{ name: "amount", min: 0, max: 1, default: 0.3, step: 0.01 }] },
  { id: "grain", name: "Film Grain", params: [{ name: "amount", min: 0, max: 0.5, default: 0.05, step: 0.01 }] },
  { id: "sepia", name: "Sepia", params: [{ name: "amount", min: 0, max: 1, default: 0.5, step: 0.01 }] },
  { id: "invert", name: "Invert", params: [{ name: "amount", min: 0, max: 1, default: 1, step: 0.01 }] },
  { id: "posterize_levels", name: "Posterize", params: [{ name: "amount", min: 2, max: 32, default: 8, step: 1 }] },
  // Creative
  { id: "chromatic_aberration", name: "Chromatic Aberration", params: [{ name: "amount", min: 0, max: 20, default: 3, step: 1 }] },
  { id: "glitch", name: "Glitch", params: [{ name: "amount", min: 0, max: 1, default: 0.3, step: 0.01 }] },
  { id: "pixelate", name: "Pixelate", params: [{ name: "amount", min: 1, max: 32, default: 4, step: 1 }] },
  { id: "edge_detect", name: "Edge Detect", params: [{ name: "amount", min: 0, max: 1, default: 0.5, step: 0.01 }] },
  { id: "emboss", name: "Emboss", params: [{ name: "amount", min: 0, max: 1, default: 0.5, step: 0.01 }] },
  { id: "scanlines", name: "Scanlines", params: [{ name: "amount", min: 0, max: 1, default: 0.3, step: 0.01 }] },
  { id: "rgb_shift", name: "RGB Shift", params: [{ name: "amount", min: -1, max: 1, default: 0.3, step: 0.01 }] },
  { id: "noise", name: "Color Noise", params: [{ name: "amount", min: 0, max: 1, default: 0.1, step: 0.01 }] },
];

class EffectLayers extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: "open" });
    this.preset = "custom";
    this.layers = [];
  }

  static get observedAttributes() {
    return ["data-value", "data-disabled"];
  }

  connectedCallback() {
    this.render();
  }

  attributeChangedCallback(name, oldValue, newValue) {
    if (name === "data-value" && oldValue !== newValue) {
      console.log("[EffectLayers] data-value changed:", newValue);
      try {
        const parsed = JSON.parse(newValue || "{}");
        console.log("[EffectLayers] parsed:", parsed);
        if (typeof parsed === "object" && parsed !== null) {
          this.preset = parsed.preset || "custom";
          this.layers = Array.isArray(parsed.layers) ? parsed.layers : [];
        } else {
          this.preset = "custom";
          this.layers = [];
        }
        console.log("[EffectLayers] this.preset set to:", this.preset);
      } catch (e) {
        console.warn("[EffectLayers] Failed to parse data-value:", e);
        this.preset = "custom";
        this.layers = [];
      }
      this.render();
    }
    if (name === "data-disabled") {
      this.render();
    }
  }

  get disabled() {
    return this.getAttribute("data-disabled") === "true";
  }

  emitChange() {
    this.dispatchEvent(
      new CustomEvent("change", {
        detail: { value: JSON.stringify({ preset: this.preset, layers: this.layers }) },
        bubbles: true,
      })
    );
  }

  setPreset(presetId) {
    this.preset = presetId;

    // When selecting a preset (not custom), populate layers with preset values
    if (presetId !== "custom" && PRESETS[presetId]) {
      const presetParams = PRESETS[presetId].params;
      this.layers = [];

      // Create a layer for each preset parameter
      for (const [paramName, paramValue] of Object.entries(presetParams)) {
        // Find matching effect
        const effect = AVAILABLE_EFFECTS.find(e => e.id === paramName);
        if (effect) {
          const params = {};
          effect.params.forEach(p => {
            params[p.name] = paramValue;
          });
          this.layers.push({
            id: Date.now().toString() + Math.random().toString(36).substr(2, 9),
            effect: paramName,
            enabled: true,
            params,
          });
        }
      }
      // Keep preset selected - it shows which preset was used as starting point
    }
    // Selecting "custom" just changes the preset label, keeps existing layers

    this.emitChange();
    this.render();
  }

  addLayer(effectId) {
    const effect = AVAILABLE_EFFECTS.find((e) => e.id === effectId);
    if (!effect) return;

    const params = {};
    effect.params.forEach((p) => {
      params[p.name] = p.default;
    });

    this.layers.push({
      id: Date.now().toString(),
      effect: effectId,
      enabled: true,
      params,
    });

    this.emitChange();
    this.render();
  }

  removeLayer(layerId) {
    this.layers = this.layers.filter((l) => l.id !== layerId);
    this.emitChange();
    this.render();
  }

  toggleLayer(layerId) {
    const layer = this.layers.find((l) => l.id === layerId);
    if (layer) {
      layer.enabled = !layer.enabled;
      this.emitChange();
      this.render();
    }
  }

  updateParam(layerId, paramName, value) {
    const layer = this.layers.find((l) => l.id === layerId);
    if (layer) {
      layer.params[paramName] = parseFloat(value);
      this.emitChange();
    }
  }

  moveLayer(layerId, direction) {
    const index = this.layers.findIndex((l) => l.id === layerId);
    if (index === -1) return;

    const newIndex = direction === "up" ? index - 1 : index + 1;
    if (newIndex < 0 || newIndex >= this.layers.length) return;

    const [layer] = this.layers.splice(index, 1);
    this.layers.splice(newIndex, 0, layer);

    this.emitChange();
    this.render();
  }

  render() {
    const disabled = this.disabled;
    const presetOptions = Object.entries(PRESETS)
      .map(([id, preset]) => `<option value="${id}" ${this.preset === id ? "selected" : ""}>${preset.name}</option>`)
      .join("");

    this.shadowRoot.innerHTML = `
      <style>
        :host {
          display: block;
          font-family: system-ui, -apple-system, sans-serif;
          font-size: 13px;
        }
        .container {
          border: 1px solid hsl(var(--border, 240 3.7% 15.9%));
          border-radius: 8px;
          background: hsl(var(--card, 240 10% 3.9%));
          overflow: hidden;
        }
        .preset-section {
          padding: 10px 12px;
          border-bottom: 1px solid hsl(var(--border, 240 3.7% 15.9%));
          background: hsl(var(--muted, 240 3.7% 15.9%));
        }
        .preset-label {
          display: block;
          font-size: 11px;
          color: hsl(var(--muted-foreground, 240 5% 64.9%));
          margin-bottom: 4px;
          text-transform: uppercase;
          letter-spacing: 0.5px;
        }
        .preset-select {
          width: 100%;
          padding: 6px 10px;
          border-radius: 4px;
          border: 1px solid hsl(var(--border, 240 3.7% 15.9%));
          background: hsl(var(--background, 240 10% 3.9%));
          color: hsl(var(--foreground, 0 0% 98%));
          font-size: 13px;
        }
        .header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 8px 12px;
          background: hsl(var(--muted, 240 3.7% 15.9%) / 0.5);
          border-bottom: 1px solid hsl(var(--border, 240 3.7% 15.9%));
        }
        .header-title {
          font-weight: 500;
          font-size: 12px;
          color: hsl(var(--muted-foreground, 240 5% 64.9%));
        }
        .layers {
          max-height: 300px;
          overflow-y: auto;
        }
        .layer {
          border-bottom: 1px solid hsl(var(--border, 240 3.7% 15.9%));
          padding: 8px 12px;
        }
        .layer:last-child {
          border-bottom: none;
        }
        .layer.disabled {
          opacity: 0.5;
        }
        .layer-header {
          display: flex;
          align-items: center;
          gap: 8px;
          margin-bottom: 8px;
        }
        .layer-name {
          flex: 1;
          font-weight: 500;
          color: hsl(var(--foreground, 0 0% 98%));
        }
        .layer-btn {
          padding: 2px 6px;
          border-radius: 3px;
          border: 1px solid hsl(var(--border, 240 3.7% 15.9%));
          background: transparent;
          color: hsl(var(--muted-foreground, 240 5% 64.9%));
          cursor: pointer;
          font-size: 11px;
        }
        .layer-btn:hover:not(:disabled) {
          background: hsl(var(--accent, 240 3.7% 15.9%));
          color: hsl(var(--foreground, 0 0% 98%));
        }
        .layer-btn.remove:hover:not(:disabled) {
          background: hsl(var(--destructive, 0 62.8% 30.6%));
          color: white;
        }
        .param {
          display: flex;
          align-items: center;
          gap: 8px;
          margin-top: 4px;
        }
        .param-label {
          width: 60px;
          color: hsl(var(--muted-foreground, 240 5% 64.9%));
          font-size: 12px;
        }
        .param-slider {
          flex: 1;
          height: 4px;
          -webkit-appearance: none;
          background: hsl(var(--muted, 240 3.7% 15.9%));
          border-radius: 2px;
        }
        .param-slider::-webkit-slider-thumb {
          -webkit-appearance: none;
          width: 12px;
          height: 12px;
          border-radius: 50%;
          background: hsl(var(--primary, 0 0% 98%));
          cursor: pointer;
        }
        .param-value {
          width: 45px;
          text-align: right;
          color: hsl(var(--foreground, 0 0% 98%));
          font-size: 12px;
          font-variant-numeric: tabular-nums;
        }
        .empty {
          padding: 16px;
          text-align: center;
          color: hsl(var(--muted-foreground, 240 5% 64.9%));
          font-size: 12px;
        }
        select {
          padding: 4px 8px;
          border-radius: 4px;
          border: 1px solid hsl(var(--border, 240 3.7% 15.9%));
          background: hsl(var(--background, 240 10% 3.9%));
          color: hsl(var(--foreground, 0 0% 98%));
          font-size: 12px;
        }
        .preset-preview {
          padding: 10px 12px;
          border-bottom: 1px solid hsl(var(--border, 240 3.7% 15.9%));
          background: hsl(var(--muted, 240 3.7% 15.9%) / 0.3);
        }
        .preset-preview-title {
          font-size: 11px;
          color: hsl(var(--muted-foreground, 240 5% 64.9%));
          margin-bottom: 6px;
          text-transform: uppercase;
          letter-spacing: 0.5px;
        }
        .preset-params {
          display: flex;
          flex-wrap: wrap;
          gap: 6px;
        }
        .preset-param {
          font-size: 11px;
          padding: 2px 6px;
          border-radius: 3px;
          background: hsl(var(--muted, 240 3.7% 15.9%));
          color: hsl(var(--foreground, 0 0% 98%));
        }
      </style>

      <div class="container">
        <div class="preset-section">
          <label class="preset-label">Preset</label>
          <select id="preset-select" class="preset-select" ${disabled ? "disabled" : ""}>
            ${presetOptions}
          </select>
        </div>
        <div class="header">
          <span class="header-title">Effect Layers (${this.layers.length})</span>
          <select id="add-effect" ${disabled ? "disabled" : ""}>
            <option value="">+ Add Effect</option>
            ${AVAILABLE_EFFECTS.map((e) => `<option value="${e.id}">${e.name}</option>`).join("")}
          </select>
        </div>
        <div class="layers">
          ${
            this.layers.length === 0
              ? '<div class="empty">No effects. Select a preset to load its effects, or add layers manually.</div>'
              : this.layers
                  .map((layer, index) => {
                    const effect = AVAILABLE_EFFECTS.find((e) => e.id === layer.effect);
                    if (!effect) return "";
                    return `
                  <div class="layer ${layer.enabled ? "" : "disabled"}" data-id="${layer.id}">
                    <div class="layer-header">
                      <button class="layer-btn toggle" data-id="${layer.id}" ${disabled ? "disabled" : ""}>
                        ${layer.enabled ? "ON" : "OFF"}
                      </button>
                      <span class="layer-name">${effect.name}</span>
                      <button class="layer-btn move-up" data-id="${layer.id}" ${disabled || index === 0 ? "disabled" : ""}>↑</button>
                      <button class="layer-btn move-down" data-id="${layer.id}" ${disabled || index === this.layers.length - 1 ? "disabled" : ""}>↓</button>
                      <button class="layer-btn remove" data-id="${layer.id}" ${disabled ? "disabled" : ""}>×</button>
                    </div>
                    ${effect.params
                      .map(
                        (param) => `
                      <div class="param">
                        <span class="param-label">${param.name}</span>
                        <input type="range" class="param-slider"
                          data-layer="${layer.id}"
                          data-param="${param.name}"
                          min="${param.min}"
                          max="${param.max}"
                          step="${param.step}"
                          value="${layer.params[param.name] ?? param.default}"
                          ${disabled || !layer.enabled ? "disabled" : ""}
                        >
                        <span class="param-value">${(layer.params[param.name] ?? param.default).toFixed(2)}</span>
                      </div>
                    `
                      )
                      .join("")}
                  </div>
                `;
                  })
                  .join("")
          }
        </div>
      </div>
    `;

    // Attach event listeners
    this.shadowRoot.querySelector("#preset-select")?.addEventListener("change", (e) => {
      this.setPreset(e.target.value);
    });

    this.shadowRoot.querySelector("#add-effect")?.addEventListener("change", (e) => {
      const value = e.target.value;
      if (value) {
        this.addLayer(value);
        e.target.value = "";
      }
    });

    this.shadowRoot.querySelectorAll(".toggle").forEach((btn) => {
      btn.addEventListener("click", () => this.toggleLayer(btn.dataset.id));
    });

    this.shadowRoot.querySelectorAll(".remove").forEach((btn) => {
      btn.addEventListener("click", () => this.removeLayer(btn.dataset.id));
    });

    this.shadowRoot.querySelectorAll(".move-up").forEach((btn) => {
      btn.addEventListener("click", () => this.moveLayer(btn.dataset.id, "up"));
    });

    this.shadowRoot.querySelectorAll(".move-down").forEach((btn) => {
      btn.addEventListener("click", () => this.moveLayer(btn.dataset.id, "down"));
    });

    this.shadowRoot.querySelectorAll(".param-slider").forEach((slider) => {
      slider.addEventListener("input", (e) => {
        const layerId = e.target.dataset.layer;
        const paramName = e.target.dataset.param;
        const value = e.target.value;

        // Update display value
        const valueDisplay = e.target.parentElement.querySelector(".param-value");
        if (valueDisplay) {
          valueDisplay.textContent = parseFloat(value).toFixed(2);
        }

        this.updateParam(layerId, paramName, value);
      });
    });
  }
}

// Register the custom element
customElements.define("effect-layers", EffectLayers);
