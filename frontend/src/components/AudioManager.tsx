import { useState } from "react";
import { Plus, X, Music } from "lucide-react";
import { MediaPicker } from "./MediaPicker";

interface AudioManagerProps {
  audioPath: string | null;
  onAudioChange: (path: string | null) => void;
  disabled?: boolean;
  label?: string;
}

export function AudioManager({
  audioPath,
  onAudioChange,
  disabled,
  label = "Audio Input",
}: AudioManagerProps) {
  const [isPickerOpen, setIsPickerOpen] = useState(false);

  const fileName = audioPath ? audioPath.split(/[/\\]/).pop() ?? audioPath : null;

  return (
    <div>
      {audioPath == null ? (
        <button
          onClick={() => setIsPickerOpen(true)}
          disabled={disabled}
          className="w-full h-16 border-2 border-dashed rounded-lg flex items-center justify-center gap-2 hover:bg-accent hover:border-accent-foreground disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          <Plus className="h-4 w-4 text-muted-foreground" />
          <span className="text-xs text-muted-foreground">{label}</span>
        </button>
      ) : (
        <div className="flex items-center gap-2 border rounded-lg px-3 py-2 bg-card group">
          <Music className="h-4 w-4 shrink-0 text-emerald-400" />
          <span className="text-xs truncate flex-1" title={audioPath}>
            {fileName}
          </span>
          <button
            onClick={() => onAudioChange(null)}
            disabled={disabled}
            className="shrink-0 p-0.5 rounded hover:bg-destructive/20 opacity-0 group-hover:opacity-100 transition-opacity disabled:opacity-50"
            title="Remove audio"
          >
            <X className="h-3 w-3" />
          </button>
        </div>
      )}

      <MediaPicker
        isOpen={isPickerOpen}
        onClose={() => setIsPickerOpen(false)}
        onSelectImage={path => {
          onAudioChange(path);
          setIsPickerOpen(false);
        }}
        disabled={disabled}
        accept="audio"
      />
    </div>
  );
}
