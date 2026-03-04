import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "../ui/dialog";
import { NODE_TOKENS } from "./node-ui";

interface AddNodeModalProps {
  open: boolean;
  onClose: () => void;
  onSelectNodeType: (type: "source" | "pipeline" | "sink" | "value", valueType?: "string" | "number" | "boolean") => void;
}

export function AddNodeModal({
  open,
  onClose,
  onSelectNodeType,
}: AddNodeModalProps) {

  return (
    <Dialog open={open} onOpenChange={onClose}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Add Node</DialogTitle>
          <DialogDescription>Select the type of node to add</DialogDescription>
        </DialogHeader>

        <div className="flex flex-col gap-3 mt-4">
          <button
            onClick={() => {
              onSelectNodeType("source");
              onClose();
            }}
            className={`${NODE_TOKENS.toolbarButton} w-full justify-start gap-3 px-4 py-6 flex items-center`}
          >
            <div className="w-3 h-3 rounded-full bg-green-400 shrink-0" />
            <div className="flex flex-col items-start">
              <span className="font-semibold">Source</span>
              <span className="text-xs text-[#8c8c8d]">
                Input node for the workflow
              </span>
            </div>
          </button>

          <button
            onClick={() => {
              onSelectNodeType("pipeline");
              onClose();
            }}
            className={`${NODE_TOKENS.toolbarButton} w-full justify-start gap-3 px-4 py-6 flex items-center`}
          >
            <div className="w-3 h-3 rounded-full bg-blue-400 shrink-0" />
            <div className="flex flex-col items-start">
              <span className="font-semibold">Pipeline</span>
              <span className="text-xs text-[#8c8c8d]">
                Processing pipeline node
              </span>
            </div>
          </button>

          <button
            onClick={() => {
              onSelectNodeType("sink");
              onClose();
            }}
            className={`${NODE_TOKENS.toolbarButton} w-full justify-start gap-3 px-4 py-6 flex items-center`}
          >
            <div className="w-3 h-3 rounded-full bg-orange-400 shrink-0" />
            <div className="flex flex-col items-start">
              <span className="font-semibold">Sink</span>
              <span className="text-xs text-[#8c8c8d]">
                Output node for the workflow
              </span>
            </div>
          </button>

          <div className="border-t border-[rgba(119,119,119,0.15)] my-2" />

          <div className="text-xs text-[#8c8c8d] px-2 mb-1">Values</div>

          <button
            onClick={() => {
              onSelectNodeType("value", "string");
              onClose();
            }}
            className={`${NODE_TOKENS.toolbarButton} w-full justify-start gap-3 px-4 py-6 flex items-center`}
          >
            <div className="w-3 h-3 rounded-full bg-amber-400 shrink-0" />
            <div className="flex flex-col items-start">
              <span className="font-semibold">String</span>
              <span className="text-xs text-[#8c8c8d]">
                Text value node
              </span>
            </div>
          </button>

          <button
            onClick={() => {
              onSelectNodeType("value", "number");
              onClose();
            }}
            className={`${NODE_TOKENS.toolbarButton} w-full justify-start gap-3 px-4 py-6 flex items-center`}
          >
            <div className="w-3 h-3 rounded-full bg-sky-400 shrink-0" />
            <div className="flex flex-col items-start">
              <span className="font-semibold">Number</span>
              <span className="text-xs text-[#8c8c8d]">
                Numeric value node
              </span>
            </div>
          </button>

          <button
            onClick={() => {
              onSelectNodeType("value", "boolean");
              onClose();
            }}
            className={`${NODE_TOKENS.toolbarButton} w-full justify-start gap-3 px-4 py-6 flex items-center`}
          >
            <div className="w-3 h-3 rounded-full bg-emerald-400 shrink-0" />
            <div className="flex flex-col items-start">
              <span className="font-semibold">Boolean</span>
              <span className="text-xs text-[#8c8c8d]">
                True/false value node
              </span>
            </div>
          </button>
        </div>
      </DialogContent>
    </Dialog>
  );
}
