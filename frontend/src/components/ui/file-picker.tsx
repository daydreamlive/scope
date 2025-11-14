import { useState, useEffect, useMemo, useRef } from "react";
import { ChevronRight, ChevronDown } from "lucide-react";
import { cn } from "../../lib/utils";

export interface FileInfo {
  name: string;
  path: string;
  size_mb: number;
  folder?: string | null;
}

interface FilePickerProps {
  value: string;
  onChange: (path: string) => void;
  files: FileInfo[];
  disabled?: boolean;
  placeholder?: string;
  emptyMessage?: string;
}

export function FilePicker({
  value,
  onChange,
  files,
  disabled,
  placeholder = "Select file",
  emptyMessage = "No files found",
}: FilePickerProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [expandedFolders, setExpandedFolders] = useState<Set<string>>(
    new Set(["Root"])
  );
  const dropdownRef = useRef<HTMLDivElement>(null);

  const groupedFiles = useMemo(() => {
    const groups: Record<string, FileInfo[]> = {};

    files.forEach(file => {
      const folder = file.folder || "Root";
      if (!groups[folder]) {
        groups[folder] = [];
      }
      groups[folder].push(file);
    });

    return Object.entries(groups).sort(([a], [b]) => {
      if (a === "Root") return -1;
      if (b === "Root") return 1;
      return a.localeCompare(b);
    });
  }, [files]);

  const selectedFile = files.find(f => f.path === value);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(event.target as Node)
      ) {
        setIsOpen(false);
      }
    };

    if (isOpen) {
      document.addEventListener("mousedown", handleClickOutside);
      return () =>
        document.removeEventListener("mousedown", handleClickOutside);
    }
  }, [isOpen]);

  const toggleFolder = (folder: string) => {
    setExpandedFolders(prev => {
      const next = new Set(prev);
      if (next.has(folder)) {
        next.delete(folder);
      } else {
        next.add(folder);
      }
      return next;
    });
  };

  const selectFile = (path: string) => {
    onChange(path);
    setIsOpen(false);
  };

  return (
    <div className="relative" ref={dropdownRef}>
      <button
        type="button"
        onClick={() => !disabled && setIsOpen(!isOpen)}
        disabled={disabled}
        className={cn(
          "flex h-7 w-full items-center justify-between rounded-md border border-input bg-background px-2 text-xs",
          "hover:bg-accent hover:text-accent-foreground",
          "disabled:cursor-not-allowed disabled:opacity-50"
        )}
      >
        <span className="truncate">
          {selectedFile ? selectedFile.name : placeholder}
        </span>
        <ChevronDown className="h-3 w-3 ml-1 shrink-0" />
      </button>

      {isOpen && (
        <div className="absolute z-50 mt-1 w-full max-h-80 overflow-y-auto rounded-md border bg-popover text-popover-foreground shadow-md">
          {groupedFiles.length === 0 ? (
            <div className="p-2 text-xs text-muted-foreground">
              {emptyMessage}
            </div>
          ) : (
            <div className="p-1">
              {groupedFiles.map(([folder, folderFiles]) => {
                const isExpanded = expandedFolders.has(folder);
                return (
                  <div key={folder}>
                    <button
                      type="button"
                      onClick={() => toggleFolder(folder)}
                      className="flex w-full items-center gap-1 rounded-sm px-2 py-1.5 text-xs font-medium hover:bg-accent"
                    >
                      {isExpanded ? (
                        <ChevronDown className="h-3 w-3 shrink-0" />
                      ) : (
                        <ChevronRight className="h-3 w-3 shrink-0" />
                      )}
                      <span className="text-muted-foreground">{folder}</span>
                      <span className="text-muted-foreground/60">
                        ({folderFiles.length})
                      </span>
                    </button>
                    {isExpanded && (
                      <div className="ml-4 space-y-0.5">
                        {folderFiles.map(file => (
                          <button
                            key={file.path}
                            type="button"
                            onClick={() => selectFile(file.path)}
                            className={cn(
                              "flex w-full items-center justify-between rounded-sm px-2 py-1.5 text-xs",
                              "hover:bg-accent hover:text-accent-foreground",
                              value === file.path && "bg-accent"
                            )}
                          >
                            <span className="truncate">{file.name}</span>
                            <span className="text-muted-foreground ml-2 shrink-0">
                              ({file.size_mb}MB)
                            </span>
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
