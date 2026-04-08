import { SHORTCUTS, isMac, type ShortcutDefinition } from "./shortcuts";

const STORAGE_KEY = "scope:shortcut-overrides";

export interface ShortcutOverride {
  key: string | string[];
  metaOrCtrl?: boolean;
  shift?: boolean;
  alt?: boolean;
}

type OverridesMap = Record<string, ShortcutOverride>;

/** Load user overrides from localStorage. */
export function loadOverrides(): OverridesMap {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return {};
    return JSON.parse(raw) as OverridesMap;
  } catch {
    return {};
  }
}

/** Save a single shortcut override. */
export function saveOverride(id: string, override: ShortcutOverride): void {
  const overrides = loadOverrides();
  overrides[id] = override;
  localStorage.setItem(STORAGE_KEY, JSON.stringify(overrides));
}

/** Remove a single shortcut override, restoring its default. */
export function resetOverride(id: string): void {
  const overrides = loadOverrides();
  delete overrides[id];
  localStorage.setItem(STORAGE_KEY, JSON.stringify(overrides));
}

/** Remove all overrides, restoring all defaults. */
export function resetAllOverrides(): void {
  localStorage.removeItem(STORAGE_KEY);
}

/** Build the human-readable keys string from an override's key binding. */
function buildKeysString(override: ShortcutOverride): string {
  const mac = isMac();
  const parts: string[] = [];

  if (override.metaOrCtrl) parts.push(mac ? "⌘" : "Ctrl +");
  if (override.shift) parts.push(mac ? "⇧" : "Shift +");
  if (override.alt) parts.push(mac ? "⌥" : "Alt +");

  const keys = Array.isArray(override.key) ? override.key : [override.key];
  // Use the first key for display, capitalize single letters
  const displayKey = keys[0].length === 1 ? keys[0].toUpperCase() : keys[0];
  parts.push(displayKey);

  return parts.join(" ");
}

/**
 * Return the full shortcut list with user overrides merged in.
 * This is the single source of truth for all shortcut consumers.
 */
export function getEffectiveShortcuts(): ShortcutDefinition[] {
  const overrides = loadOverrides();

  return SHORTCUTS.map(s => {
    const override = overrides[s.id];
    if (!override) return s;
    return {
      ...s,
      key: override.key,
      metaOrCtrl: override.metaOrCtrl ?? false,
      shift: override.shift,
      alt: override.alt,
      keys: buildKeysString(override),
    };
  });
}

/**
 * Check if a given key binding conflicts with any existing shortcut.
 * Returns the conflicting shortcut definition, or undefined if no conflict.
 */
export function findConflict(
  candidateId: string,
  override: ShortcutOverride
): ShortcutDefinition | undefined {
  const effective = getEffectiveShortcuts();
  const candidateKeys = new Set(
    (Array.isArray(override.key) ? override.key : [override.key]).map(k =>
      k.toLowerCase()
    )
  );

  return effective.find(s => {
    if (s.id === candidateId) return false;
    if (s.builtIn) return false;
    if (!!s.metaOrCtrl !== !!override.metaOrCtrl) return false;
    if (!!s.shift !== !!override.shift) return false;
    if (!!s.alt !== !!override.alt) return false;

    const sKeys = (Array.isArray(s.key) ? s.key : [s.key]).map(k =>
      k.toLowerCase()
    );
    return sKeys.some(k => candidateKeys.has(k));
  });
}
