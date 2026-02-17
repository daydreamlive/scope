import * as Dialog from "@radix-ui/react-dialog";
import { X } from "lucide-react";
import { DagEditor } from "./DagEditor";

interface DagEditorDialogProps {
  open: boolean;
  onClose: () => void;
}

export function DagEditorDialog({ open, onClose }: DagEditorDialogProps) {
  return (
    <Dialog.Root open={open} onOpenChange={val => !val && onClose()}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 bg-black/60 z-50" />
        <Dialog.Content className="fixed top-[50%] left-[50%] translate-x-[-50%] translate-y-[-50%] w-[95vw] h-[85vh] bg-zinc-900 rounded-xl border border-zinc-700 shadow-2xl z-50 flex flex-col overflow-hidden focus:outline-none">
          <div className="flex items-center justify-between px-4 py-3 border-b border-zinc-700">
            <Dialog.Title className="text-sm font-semibold text-zinc-100">
              DAG Editor
            </Dialog.Title>
            <Dialog.Close asChild>
              <button className="text-zinc-400 hover:text-zinc-200 transition-colors">
                <X className="h-4 w-4" />
              </button>
            </Dialog.Close>
          </div>
          <div className="flex-1 overflow-hidden">
            <DagEditor />
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
