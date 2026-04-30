import { useRef, useState } from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "../ui/dialog";
import { getShortcutsByCategory } from "../../lib/shortcuts";
import { getEffectiveShortcuts } from "../../lib/shortcutOverrides";

type Tab = "basics" | "integrations" | "shortcuts";

const TABS = [
  { id: "basics", label: "Basics" },
  { id: "integrations", label: "Integrations" },
  { id: "shortcuts", label: "Shortcuts" },
] as const satisfies ReadonlyArray<{ id: Tab; label: string }>;

const KBD_CLASS =
  "inline-flex items-center gap-1 rounded border border-border bg-muted px-2 py-0.5 font-mono text-[11px] text-muted-foreground";

interface CheatSheetDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  initialTab?: Tab;
}

interface HowToItem {
  title: string;
  body: React.ReactNode;
}

const BASICS: HowToItem[] = [
  {
    title: "Add a node",
    body: (
      <>
        Click the <span className="font-mono">+</span> button in the upper-right
        of the canvas to open the node registry, press{" "}
        <kbd className={KBD_CLASS}>Tab</kbd>, or right-click empty canvas.
      </>
    ),
  },
  {
    title: "Import a workflow",
    body: (
      <>
        Open the <span className="font-mono">Graph</span> menu in the upper-left
        of the canvas, then choose{" "}
        <span className="font-mono">Import Workflow</span>.
      </>
    ),
  },
  {
    title: "Install a pipeline / plugin",
    body: (
      <>
        Pipelines are registered via the Python package. Add them to the
        project&apos;s <span className="font-mono">pyproject.toml</span> or
        install a plugin package, then restart Scope. They&apos;ll appear in the
        node registry.
      </>
    ),
  },
  {
    title: "Install a LoRA",
    body: (
      <>
        Open <span className="font-mono">Settings</span> (top-right gear) →{" "}
        <span className="font-mono">LoRAs</span> tab and paste a URL to install.
        Installed LoRAs appear in the LoRA picker on pipelines that support
        them.
      </>
    ),
  },
];

interface IntegrationRow {
  name: string;
  location: string;
  install: string;
}

const INTEGRATIONS: IntegrationRow[] = [
  {
    name: "Spout",
    location: "Source / Output node → type dropdown",
    install: "Windows only; bundled with Scope",
  },
  {
    name: "Syphon",
    location: "Source / Output node → type dropdown",
    install: "macOS only; bundled with Scope",
  },
  {
    name: "NDI",
    location: "Source / Output node → type dropdown",
    install: "All platforms. Install NDI SDK: https://ndi.video/tools",
  },
  {
    name: "OSC",
    location: "Settings → OSC tab",
    install: "Always available",
  },
  {
    name: "DMX (Art-Net)",
    location: "Settings → DMX tab",
    install: "Always available (binds ports 6454–6457)",
  },
  {
    name: "Ableton Link",
    location: "Settings → Tempo Sync",
    install: "uv sync --extra link",
  },
  {
    name: "MIDI Clock",
    location: "Settings → Tempo Sync, MIDI node in graph",
    install: "uv sync --extra midi",
  },
];

const tabId = (id: Tab) => `cheat-sheet-tab-${id}`;
const panelId = (id: Tab) => `cheat-sheet-panel-${id}`;

export function CheatSheetDialog({
  open,
  onOpenChange,
  initialTab = "basics",
}: CheatSheetDialogProps) {
  const [tab, setTab] = useState<Tab>(initialTab);
  const tabRefs = useRef<Record<Tab, HTMLButtonElement | null>>({
    basics: null,
    integrations: null,
    shortcuts: null,
  });
  const shortcuts = getEffectiveShortcuts();
  const categories = getShortcutsByCategory(shortcuts);

  const handleTabKeyDown = (e: React.KeyboardEvent<HTMLButtonElement>) => {
    const currentIndex = TABS.findIndex(t => t.id === tab);
    let nextIndex = currentIndex;
    if (e.key === "ArrowRight") nextIndex = (currentIndex + 1) % TABS.length;
    else if (e.key === "ArrowLeft")
      nextIndex = (currentIndex - 1 + TABS.length) % TABS.length;
    else if (e.key === "Home") nextIndex = 0;
    else if (e.key === "End") nextIndex = TABS.length - 1;
    else return;
    e.preventDefault();
    const nextTab = TABS[nextIndex].id;
    setTab(nextTab);
    tabRefs.current[nextTab]?.focus();
  };

  return (
    <Dialog
      open={open}
      onOpenChange={next => {
        if (next) setTab(initialTab);
        onOpenChange(next);
      }}
    >
      <DialogContent className="sm:max-w-[560px] h-[80vh] max-h-[600px] flex flex-col overflow-hidden">
        <DialogHeader>
          <DialogTitle>Scope cheat sheet</DialogTitle>
          <DialogDescription>
            Basics, integrations, and keyboard shortcuts in one place.
          </DialogDescription>
        </DialogHeader>

        <div
          role="tablist"
          aria-label="Cheat sheet sections"
          className="flex items-center gap-1 border-b border-border mt-3"
        >
          {TABS.map(t => {
            const selected = tab === t.id;
            return (
              <button
                key={t.id}
                ref={el => {
                  tabRefs.current[t.id] = el;
                }}
                id={tabId(t.id)}
                role="tab"
                aria-selected={selected}
                aria-controls={panelId(t.id)}
                tabIndex={selected ? 0 : -1}
                type="button"
                onClick={() => setTab(t.id)}
                onKeyDown={handleTabKeyDown}
                className={`px-3 py-1.5 text-xs font-medium rounded-t-md border-b-2 -mb-px transition-colors ${
                  selected
                    ? "border-foreground text-foreground"
                    : "border-transparent text-muted-foreground hover:text-foreground"
                }`}
              >
                {t.label}
              </button>
            );
          })}
        </div>

        <div className="mt-4 flex-1 min-h-0 overflow-y-auto pr-1">
          <div
            role="tabpanel"
            id={panelId("basics")}
            aria-labelledby={tabId("basics")}
            tabIndex={0}
            hidden={tab !== "basics"}
            className="space-y-3 focus:outline-none"
          >
            {BASICS.map(item => (
              <div key={item.title}>
                <h3 className="text-sm font-semibold text-foreground mb-1">
                  {item.title}
                </h3>
                <p className="text-xs text-muted-foreground leading-relaxed">
                  {item.body}
                </p>
              </div>
            ))}
          </div>

          <div
            role="tabpanel"
            id={panelId("integrations")}
            aria-labelledby={tabId("integrations")}
            tabIndex={0}
            hidden={tab !== "integrations"}
            className="space-y-3 focus:outline-none"
          >
            {INTEGRATIONS.map(row => (
              <div
                key={row.name}
                className="grid grid-cols-[1fr_2fr] gap-3 items-baseline border-b border-border pb-2 last:border-0"
              >
                <div className="text-sm font-semibold text-foreground">
                  {row.name}
                </div>
                <div className="text-xs text-muted-foreground leading-relaxed">
                  <div>{row.location}</div>
                  <div className="mt-0.5 font-mono text-[10.5px] opacity-80">
                    {row.install}
                  </div>
                </div>
              </div>
            ))}
          </div>

          <div
            role="tabpanel"
            id={panelId("shortcuts")}
            aria-labelledby={tabId("shortcuts")}
            tabIndex={0}
            hidden={tab !== "shortcuts"}
            className="space-y-5 focus:outline-none"
          >
            {categories.map(({ category, label, items }) => (
              <div key={category}>
                <h3 className="text-[11px] font-semibold tracking-widest text-muted-foreground uppercase mb-2">
                  {label}
                </h3>
                <div className="space-y-0.5">
                  {items.map(shortcut => (
                    <div
                      key={shortcut.id}
                      className="flex items-center justify-between py-1.5 px-1 rounded-md"
                    >
                      <span className="text-sm text-foreground">
                        {shortcut.label}
                      </span>
                      <kbd className={KBD_CLASS}>{shortcut.keys}</kbd>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
