import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "../ui/dialog";
import { getShortcutsByCategory } from "../../lib/shortcuts";
import { getEffectiveShortcuts } from "../../lib/shortcutOverrides";

interface KeyboardShortcutsDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function KeyboardShortcutsDialog({
  open,
  onOpenChange,
}: KeyboardShortcutsDialogProps) {
  const shortcuts = getEffectiveShortcuts();
  const categories = getShortcutsByCategory(shortcuts);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[480px] max-h-[80vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Keyboard Shortcuts</DialogTitle>
          <DialogDescription>
            Quick reference for all available shortcuts.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-5 mt-2">
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
                    <kbd className="inline-flex items-center gap-1 rounded border border-border bg-muted px-2 py-0.5 font-mono text-[11px] text-muted-foreground">
                      {shortcut.keys}
                    </kbd>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </DialogContent>
    </Dialog>
  );
}
