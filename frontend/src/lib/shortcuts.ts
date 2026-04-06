export type ShortcutCategory =
  | "canvas"
  | "node"
  | "workflow"
  | "streaming"
  | "ui";

export interface ShortcutDefinition {
  id: string;
  label: string;
  description: string;
  /** Human-readable display string, e.g. "⌘D" or "⇧Enter" */
  keys: string;
  category: ShortcutCategory;
  /** The `e.key` value(s) to match. Array means "any of these". */
  key: string | string[];
  metaOrCtrl?: boolean;
  shift?: boolean;
  alt?: boolean;
  /** When true, fires even when INPUT/TEXTAREA/SELECT is focused */
  allowInInput?: boolean;
  /** When true, shortcut is suppressed while streaming */
  disabledWhileStreaming?: boolean;
  /** Display-only entry for built-in React Flow shortcuts — not handled by our listener */
  builtIn?: boolean;
}

const CATEGORY_ORDER: ShortcutCategory[] = [
  "canvas",
  "node",
  "workflow",
  "streaming",
  "ui",
];

export const CATEGORY_LABELS: Record<ShortcutCategory, string> = {
  canvas: "Canvas Navigation",
  node: "Node Operations",
  workflow: "Workflow",
  streaming: "Streaming",
  ui: "Interface",
};

export function isMac(): boolean {
  return typeof navigator !== "undefined" && /Mac|iPhone|iPad/.test(navigator.userAgent);
}

/** Platform-aware modifier symbol */
function mod(): string {
  return isMac() ? "⌘ " : "Ctrl + ";
}

function shiftMod(): string {
  return isMac() ? "⇧ " : "Shift + ";
}

/**
 * The default shortcut definitions.
 * Order within each category determines display order in the help overlay.
 */
export const SHORTCUTS: ShortcutDefinition[] = [
  // ── Canvas Navigation ─────────────────────────────────────────────
  {
    id: "zoom-in",
    label: "Zoom in",
    description: "Zoom into the canvas",
    keys: `${mod()}=`,
    category: "canvas",
    key: ["=", "+"],
    metaOrCtrl: true,
  },
  {
    id: "zoom-out",
    label: "Zoom out",
    description: "Zoom out of the canvas",
    keys: `${mod()}-`,
    category: "canvas",
    key: "-",
    metaOrCtrl: true,
  },
  {
    id: "zoom-reset",
    label: "Reset zoom",
    description: "Reset zoom to 100%",
    keys: `${mod()}0`,
    category: "canvas",
    key: "0",
    metaOrCtrl: true,
  },
  {
    id: "fit-view",
    label: "Fit view",
    description: "Zoom to fit all nodes",
    keys: `${mod()}${shiftMod()}F`,
    category: "canvas",
    key: "f",
    metaOrCtrl: true,
    shift: true,
  },
  {
    id: "fit-view-home",
    label: "Fit view",
    description: "Zoom to fit all nodes",
    keys: "Home",
    category: "canvas",
    key: "Home",
  },
  {
    id: "pan",
    label: "Pan",
    description: "Hold Space and drag to pan the canvas",
    keys: "Space + Drag",
    category: "canvas",
    key: " ",
    builtIn: true,
  },

  // ── Node Operations ──────────────��───────────────────────��────────
  {
    id: "open-add-node",
    label: "Add node",
    description: "Open the add node search",
    keys: "Tab",
    category: "node",
    key: "Tab",
    disabledWhileStreaming: true,
  },
  {
    id: "delete",
    label: "Delete selected",
    description: "Delete selected nodes",
    keys: "Delete / Backspace",
    category: "node",
    key: ["Delete", "Backspace"],
    builtIn: true,
    disabledWhileStreaming: true,
  },
  {
    id: "duplicate",
    label: "Duplicate",
    description: "Duplicate selected nodes",
    keys: `${mod()}D`,
    category: "node",
    key: "d",
    metaOrCtrl: true,
    disabledWhileStreaming: true,
  },
  {
    id: "copy",
    label: "Copy",
    description: "Copy selected nodes",
    keys: `${mod()}C`,
    category: "node",
    key: "c",
    metaOrCtrl: true,
  },
  {
    id: "paste",
    label: "Paste",
    description: "Paste copied nodes",
    keys: `${mod()}V`,
    category: "node",
    key: "v",
    metaOrCtrl: true,
  },
  {
    id: "cut",
    label: "Cut",
    description: "Cut selected nodes",
    keys: `${mod()}X`,
    category: "node",
    key: "x",
    metaOrCtrl: true,
    disabledWhileStreaming: true,
  },
  {
    id: "select-all",
    label: "Select all",
    description: "Select all nodes on the canvas",
    keys: `${mod()}A`,
    category: "node",
    key: "a",
    metaOrCtrl: true,
  },
  {
    id: "deselect",
    label: "Deselect all",
    description: "Deselect all nodes / close modals",
    keys: "Escape",
    category: "node",
    key: "Escape",
    allowInInput: true,
  },
  {
    id: "lock-node",
    label: "Lock / Unlock",
    description: "Toggle lock on selected nodes",
    keys: "L",
    category: "node",
    key: "l",
  },
  {
    id: "pin-node",
    label: "Pin / Unpin",
    description: "Toggle pin on selected nodes",
    keys: "P",
    category: "node",
    key: "p",
  },
  {
    id: "group-nodes",
    label: "Group into subgraph",
    description: "Group selected nodes into a subgraph",
    keys: "G",
    category: "node",
    key: "g",
    disabledWhileStreaming: true,
  },

  // ── Workflow ─────────��─────────────────────────────────────────────
  {
    id: "save",
    label: "Save",
    description: "Save the current workflow",
    keys: `${mod()}S`,
    category: "workflow",
    key: "s",
    metaOrCtrl: true,
    allowInInput: true,
  },
  {
    id: "export",
    label: "Export",
    description: "Export workflow to file",
    keys: `${mod()}${shiftMod()}E`,
    category: "workflow",
    key: "e",
    metaOrCtrl: true,
    shift: true,
  },

  // ── Streaming ───────���─────────────────────────────────────────────
  {
    id: "toggle-stream",
    label: "Toggle stream",
    description: "Start or stop the stream",
    keys: `${mod()}Enter`,
    category: "streaming",
    key: "Enter",
    metaOrCtrl: true,
  },
  {
    id: "toggle-stream-alt",
    label: "Toggle stream",
    description: "Start or stop the stream (alt)",
    keys: `${shiftMod()}Enter`,
    category: "streaming",
    key: "Enter",
    shift: true,
  },

  // ── Interface ─────────────────���───────────────────────────────────
  {
    id: "show-shortcuts",
    label: "Keyboard shortcuts",
    description: "Show keyboard shortcuts reference",
    keys: "?",
    category: "ui",
    key: "?",
  },
];

/**
 * Match a KeyboardEvent against a ShortcutDefinition.
 */
export function matchesShortcut(
  e: KeyboardEvent,
  s: ShortcutDefinition
): boolean {
  // metaOrCtrl check
  const hasMetaOrCtrl = e.metaKey || e.ctrlKey;
  if (s.metaOrCtrl && !hasMetaOrCtrl) return false;
  if (!s.metaOrCtrl && hasMetaOrCtrl) return false;

  // shift check — only enforce when explicitly set
  if (s.shift === true && !e.shiftKey) return false;
  if (s.shift === false && e.shiftKey) return false;
  // when s.shift is undefined, we don't care about shiftKey

  // alt check
  if (s.alt === true && !e.altKey) return false;
  if (s.alt === false && e.altKey) return false;

  // key check (case-insensitive for letters)
  const keys = Array.isArray(s.key) ? s.key : [s.key];
  const eventKey = e.key;
  return keys.some(
    k =>
      k === eventKey ||
      (k.length === 1 &&
        eventKey.length === 1 &&
        k.toLowerCase() === eventKey.toLowerCase())
  );
}

/**
 * Get the platform-aware display string for a shortcut.
 * Uses the effective (possibly overridden) shortcut.
 */
export function getDisplayKey(s: ShortcutDefinition): string {
  return s.keys;
}

/**
 * Find a shortcut by ID in a given list.
 */
export function getShortcutById(
  id: string,
  shortcuts: ShortcutDefinition[] = SHORTCUTS
): ShortcutDefinition | undefined {
  return shortcuts.find(s => s.id === id);
}

/**
 * Group shortcuts by category in display order, deduplicating entries that
 * share the same label within a category (e.g. fit-view and fit-view-home).
 */
export function getShortcutsByCategory(
  shortcuts: ShortcutDefinition[] = SHORTCUTS
): { category: ShortcutCategory; label: string; items: ShortcutDefinition[] }[] {
  const grouped = new Map<ShortcutCategory, ShortcutDefinition[]>();
  const seenLabels = new Map<ShortcutCategory, Set<string>>();

  for (const s of shortcuts) {
    if (!grouped.has(s.category)) {
      grouped.set(s.category, []);
      seenLabels.set(s.category, new Set());
    }
    const seen = seenLabels.get(s.category)!;
    // Deduplicate: skip alternate bindings for the same label (e.g. toggle-stream-alt)
    if (seen.has(s.label)) continue;
    seen.add(s.label);
    grouped.get(s.category)!.push(s);
  }

  return CATEGORY_ORDER.filter(c => grouped.has(c)).map(c => ({
    category: c,
    label: CATEGORY_LABELS[c],
    items: grouped.get(c)!,
  }));
}
