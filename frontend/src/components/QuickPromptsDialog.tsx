import * as React from "react";
import { useState, useCallback, useEffect, useId } from "react";
import {
  DndContext,
  closestCenter,
  KeyboardSensor,
  PointerSensor,
  useSensor,
  useSensors,
  type DragEndEvent,
  type UniqueIdentifier,
} from "@dnd-kit/core";
import { restrictToVerticalAxis } from "@dnd-kit/modifiers";
import {
  arrayMove,
  SortableContext,
  useSortable,
  verticalListSortingStrategy,
} from "@dnd-kit/sortable";
import { CSS } from "@dnd-kit/utilities";
import { GripVertical, Plus, BookOpen, X } from "lucide-react";

import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
  DialogDescription,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";

const MAX_PROMPTS = 7;

const SAMPLE_PROMPTS = [
  "A 3D animated scene. A **panda** walks along a path towards the camera in a park on a spring day.",
  "A cinematic shot of a **futuristic city** at night with neon lights reflecting on wet streets.",
  "An underwater scene with **colorful coral reefs** and tropical fish swimming gracefully.",
  "A **mystical forest** with glowing mushrooms and fireflies floating in the misty air.",
  "A **cozy cabin** interior with a warm fireplace and snow falling outside the window.",
  "An **astronaut** floating in space with Earth visible in the background.",
  "A **desert landscape** at sunset with sand dunes stretching to the horizon.",
];

interface QuickPrompt {
  id: string;
  text: string;
}

interface DragHandleProps {
  id: string;
}

function DragHandle({ id }: DragHandleProps) {
  const { attributes, listeners } = useSortable({ id });

  return (
    <button
      {...attributes}
      {...listeners}
      className="flex items-center justify-center w-6 h-6 text-muted-foreground hover:text-foreground cursor-grab active:cursor-grabbing"
    >
      <GripVertical className="w-4 h-4" />
      <span className="sr-only">Drag to reorder</span>
    </button>
  );
}

interface SortableRowProps {
  prompt: QuickPrompt;
  index: number;
  onTextChange: (id: string, text: string) => void;
  onApply: (text: string) => void;
  onDelete: (id: string) => void;
  isMac: boolean;
  canDelete: boolean;
}

function SortableRow({
  prompt,
  index,
  onTextChange,
  onApply,
  onDelete,
  isMac,
  canDelete,
}: SortableRowProps) {
  const { setNodeRef, transform, transition, isDragging } = useSortable({
    id: prompt.id,
  });

  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
  };

  const shortcutNumber = index + 1;

  return (
    <div
      ref={setNodeRef}
      style={style}
      className={cn(
        "flex items-center gap-2 p-2 rounded-md border bg-background",
        isDragging && "opacity-50 z-10 shadow-lg"
      )}
    >
      <DragHandle id={prompt.id} />

      <kbd className="inline-flex items-center justify-center h-6 min-w-[4rem] px-2 text-xs font-mono bg-muted text-muted-foreground rounded border border-border shrink-0">
        {isMac ? "⌘" : "Ctrl"}+{shortcutNumber}
      </kbd>

      <Input
        value={prompt.text}
        onChange={e => onTextChange(prompt.id, e.target.value)}
        className="flex-1 h-8 text-sm"
        placeholder="Enter prompt..."
      />

      <Button
        variant="outline"
        size="sm"
        className="h-8 px-3 shrink-0"
        onClick={() => onApply(prompt.text)}
      >
        Apply
      </Button>

      {canDelete ? (
        <Button
          variant="ghost"
          size="icon"
          className="h-8 w-8 shrink-0 text-muted-foreground hover:text-destructive"
          onClick={() => onDelete(prompt.id)}
          title="Delete prompt"
        >
          <X className="h-4 w-4" />
        </Button>
      ) : (
        <div className="w-8 shrink-0" />
      )}
    </div>
  );
}

interface QuickPromptsDialogProps {
  onApplyPrompt: (text: string) => void;
  className?: string;
}

export function QuickPromptsDialog({
  onApplyPrompt,
  className,
}: QuickPromptsDialogProps) {
  const [open, setOpen] = useState(false);
  const [prompts, setPrompts] = useState<QuickPrompt[]>(() => {
    // Try to load from localStorage
    const saved = localStorage.getItem("scope-quick-prompts");
    if (saved) {
      try {
        return JSON.parse(saved);
      } catch {
        // Fall through to defaults
      }
    }
    // Initialize with sample prompts
    return SAMPLE_PROMPTS.map((text, i) => ({
      id: `prompt-${i}`,
      text,
    }));
  });

  const sortableId = useId();
  const [isMac, setIsMac] = useState(false);

  useEffect(() => {
    setIsMac(navigator.platform.toUpperCase().indexOf("MAC") >= 0);
  }, []);

  // Save to localStorage whenever prompts change
  useEffect(() => {
    localStorage.setItem("scope-quick-prompts", JSON.stringify(prompts));
  }, [prompts]);

  const sensors = useSensors(
    useSensor(PointerSensor, {
      activationConstraint: {
        distance: 5,
      },
    }),
    useSensor(KeyboardSensor)
  );

  const dataIds = React.useMemo<UniqueIdentifier[]>(
    () => prompts.map(p => p.id),
    [prompts]
  );

  const handleDragEnd = useCallback(
    (event: DragEndEvent) => {
      const { active, over } = event;
      if (active && over && active.id !== over.id) {
        setPrompts(currentPrompts => {
          const oldIndex = dataIds.indexOf(active.id);
          const newIndex = dataIds.indexOf(over.id);
          return arrayMove(currentPrompts, oldIndex, newIndex);
        });
      }
    },
    [dataIds]
  );

  const handleTextChange = useCallback((id: string, text: string) => {
    setPrompts(currentPrompts =>
      currentPrompts.map(p => (p.id === id ? { ...p, text } : p))
    );
  }, []);

  const handleAddPrompt = useCallback(() => {
    if (prompts.length >= MAX_PROMPTS) return;

    const newId = `prompt-${Date.now()}`;
    setPrompts(currentPrompts => [...currentPrompts, { id: newId, text: "" }]);
  }, [prompts.length]);

  const handleDeletePrompt = useCallback((id: string) => {
    setPrompts(currentPrompts => currentPrompts.filter(p => p.id !== id));
  }, []);

  const handleApply = useCallback(
    (text: string) => {
      if (text.trim()) {
        onApplyPrompt(text);
        setOpen(false);
      }
    },
    [onApplyPrompt]
  );

  // Global keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const modKey = isMac ? e.metaKey : e.ctrlKey;
      const num = parseInt(e.key);

      if (modKey && num >= 1 && num <= 7) {
        e.preventDefault();
        const promptIndex = num - 1;
        if (promptIndex < prompts.length && prompts[promptIndex].text.trim()) {
          onApplyPrompt(prompts[promptIndex].text);
        }
      }
    };

    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [isMac, prompts, onApplyPrompt]);

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button
          variant="ghost"
          size="icon"
          className={cn("h-6 w-6", className)}
          title="Quick Prompts"
        >
          <BookOpen className="h-4 w-4" />
          <span className="sr-only">Quick Prompts</span>
        </Button>
      </DialogTrigger>
      <DialogContent className="max-w-2xl max-h-[80vh] overflow-hidden flex flex-col">
        <DialogHeader>
          <DialogTitle>Quick Prompts</DialogTitle>
          <DialogDescription>
            Save and quickly apply prompts using keyboard shortcuts. Drag to
            reorder.
          </DialogDescription>
        </DialogHeader>

        <div className="flex-1 overflow-y-auto space-y-2 py-2">
          <DndContext
            sensors={sensors}
            collisionDetection={closestCenter}
            modifiers={[restrictToVerticalAxis]}
            onDragEnd={handleDragEnd}
            id={sortableId}
          >
            <SortableContext
              items={dataIds}
              strategy={verticalListSortingStrategy}
            >
              {prompts.map((prompt, index) => (
                <SortableRow
                  key={prompt.id}
                  prompt={prompt}
                  index={index}
                  onTextChange={handleTextChange}
                  onApply={handleApply}
                  onDelete={handleDeletePrompt}
                  isMac={isMac}
                  canDelete={index > 0}
                />
              ))}
            </SortableContext>
          </DndContext>
        </div>

        {prompts.length < MAX_PROMPTS && (
          <div className="pt-2">
            <Button
              variant="outline"
              size="sm"
              onClick={handleAddPrompt}
              className="w-full"
            >
              <Plus className="h-4 w-4 mr-2" />
              Add Prompt ({prompts.length}/{MAX_PROMPTS})
            </Button>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}
