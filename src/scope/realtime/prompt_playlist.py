"""
PromptPlaylist - Load and navigate through a list of prompts from caption files.

Features:
- Load prompts from text files (one per line)
- Trigger phrase swapping (e.g., "1988 Cel Animation" -> "Rankin/Bass Animagic Stop-Motion")
- Navigation: next, prev, goto, current
- Optional shuffle and loop modes
"""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _apply_trigger_swap(prompts: list[str], old_trigger: str, new_trigger: str) -> list[str]:
    """Replace old trigger phrase with new one in all prompts.

    Uses case-insensitive matching for flexibility.
    """
    result = []
    pattern = re.compile(re.escape(old_trigger), re.IGNORECASE)
    for prompt in prompts:
        swapped = pattern.sub(new_trigger, prompt)
        result.append(swapped)
    return result


@dataclass
class PromptPlaylist:
    """A navigable playlist of prompts loaded from a caption file."""

    source_file: str = ""
    prompts: list[str] = field(default_factory=list)
    current_index: int = 0

    # Trigger swapping: (old_trigger, new_trigger)
    trigger_swap: tuple[str, str] | None = None

    # Original prompts (before any trigger swap) - enables re-swapping
    original_prompts: list[str] = field(default_factory=list)

    # Source trigger: what trigger phrase is in the original prompts (from file)
    # This is what we search for when doing swaps
    source_trigger: str | None = None

    # Current trigger applied to prompts (what we've swapped TO)
    current_trigger: str | None = None

    # Metadata
    original_count: int = 0

    # Prompt bookmarks - indices of bookmarked prompts for quick navigation
    bookmarked_indices: set[int] = field(default_factory=set)

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        trigger_swap: tuple[str, str] | None = None,
        skip_empty: bool = True,
    ) -> "PromptPlaylist":
        """
        Load prompts from a text file (one prompt per line).

        Args:
            path: Path to the caption file
            trigger_swap: Optional (old, new) trigger phrase to swap
            skip_empty: Whether to skip empty lines

        Returns:
            PromptPlaylist instance
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Caption file not found: {path}")

        lines = path.read_text().strip().split("\n")
        original_count = len(lines)

        # First pass: collect original prompts (no swap applied)
        original_prompts = []
        for line in lines:
            line = line.strip()
            if skip_empty and not line:
                continue
            original_prompts.append(line)

        # Second pass: apply trigger swap if configured
        prompts = list(original_prompts)  # copy
        source_trigger = None
        current_trigger = None
        if trigger_swap:
            old_trigger, new_trigger = trigger_swap
            source_trigger = old_trigger  # What's in the file
            current_trigger = new_trigger  # What we're swapping to
            prompts = _apply_trigger_swap(prompts, old_trigger, new_trigger)

        logger.info(
            f"Loaded {len(prompts)} prompts from {path.name}"
            + (f" (swapped '{trigger_swap[0]}' -> '{trigger_swap[1]}')" if trigger_swap else "")
        )

        playlist = cls(
            source_file=str(path),
            prompts=prompts,
            current_index=0,
            trigger_swap=trigger_swap,
            original_prompts=original_prompts,
            source_trigger=source_trigger,
            current_trigger=current_trigger,
            original_count=original_count,
        )

        # Load any saved bookmarks
        playlist._load_bookmarks()

        return playlist

    @property
    def current(self) -> str:
        """Get the current prompt."""
        if not self.prompts:
            return ""
        return self.prompts[self.current_index]

    @property
    def total(self) -> int:
        """Total number of prompts."""
        return len(self.prompts)

    @property
    def has_next(self) -> bool:
        """Check if there's a next prompt."""
        return self.current_index < len(self.prompts) - 1

    @property
    def has_prev(self) -> bool:
        """Check if there's a previous prompt."""
        return self.current_index > 0

    def next(self) -> str:
        """Move to next prompt and return it."""
        if self.has_next:
            self.current_index += 1
        return self.current

    def prev(self) -> str:
        """Move to previous prompt and return it."""
        if self.has_prev:
            self.current_index -= 1
        return self.current

    def goto(self, index: int) -> str:
        """Go to a specific prompt index."""
        if self.prompts:
            self.current_index = max(0, min(index, len(self.prompts) - 1))
        return self.current

    def first(self) -> str:
        """Go to first prompt."""
        return self.goto(0)

    def last(self) -> str:
        """Go to last prompt."""
        return self.goto(len(self.prompts) - 1)

    # Bookmark methods
    def bookmark_current(self) -> bool:
        """Bookmark the current prompt index. Returns True if newly added."""
        if self.current_index in self.bookmarked_indices:
            return False
        self.bookmarked_indices.add(self.current_index)
        logger.info(f"Bookmarked prompt {self.current_index}")
        self._save_bookmarks()
        return True

    def unbookmark_current(self) -> bool:
        """Remove bookmark from current prompt index. Returns True if removed."""
        if self.current_index not in self.bookmarked_indices:
            return False
        self.bookmarked_indices.discard(self.current_index)
        logger.info(f"Unbookmarked prompt {self.current_index}")
        self._save_bookmarks()
        return True

    def toggle_bookmark(self) -> bool:
        """Toggle bookmark on current prompt. Returns True if now bookmarked."""
        if self.current_index in self.bookmarked_indices:
            self.unbookmark_current()
            return False
        else:
            self.bookmark_current()
            return True

    def is_bookmarked(self, index: int | None = None) -> bool:
        """Check if an index (or current) is bookmarked."""
        idx = index if index is not None else self.current_index
        return idx in self.bookmarked_indices

    def next_bookmarked(self) -> str | None:
        """Move to the next bookmarked prompt. Returns None if no bookmarks ahead."""
        if not self.bookmarked_indices:
            return None
        sorted_bookmarks = sorted(self.bookmarked_indices)
        for idx in sorted_bookmarks:
            if idx > self.current_index:
                self.current_index = idx
                logger.info(f"Jumped to bookmarked prompt {idx}")
                return self.current
        # Wrap around to first bookmark
        if sorted_bookmarks:
            self.current_index = sorted_bookmarks[0]
            logger.info(f"Wrapped to first bookmarked prompt {sorted_bookmarks[0]}")
            return self.current
        return None

    def prev_bookmarked(self) -> str | None:
        """Move to the previous bookmarked prompt. Returns None if no bookmarks behind."""
        if not self.bookmarked_indices:
            return None
        sorted_bookmarks = sorted(self.bookmarked_indices, reverse=True)
        for idx in sorted_bookmarks:
            if idx < self.current_index:
                self.current_index = idx
                logger.info(f"Jumped to bookmarked prompt {idx}")
                return self.current
        # Wrap around to last bookmark
        if sorted_bookmarks:
            self.current_index = sorted_bookmarks[0]
            logger.info(f"Wrapped to last bookmarked prompt {sorted_bookmarks[0]}")
            return self.current
        return None

    def clear_bookmarks(self) -> int:
        """Clear all bookmarks. Returns count of cleared bookmarks."""
        count = len(self.bookmarked_indices)
        self.bookmarked_indices.clear()
        logger.info(f"Cleared {count} bookmarks")
        self._save_bookmarks()
        return count

    def _get_bookmarks_path(self) -> Path | None:
        """Get the path for the bookmarks sidecar file."""
        if not self.source_file:
            return None
        source = Path(self.source_file)
        return source.parent / f"{source.stem}.bookmarks.json"

    def _save_bookmarks(self) -> bool:
        """Save bookmarks to sidecar JSON file."""
        bookmarks_path = self._get_bookmarks_path()
        if not bookmarks_path:
            return False
        try:
            data = {
                "source_file": self.source_file,
                "bookmarked_indices": sorted(self.bookmarked_indices),
            }
            bookmarks_path.write_text(json.dumps(data, indent=2))
            logger.info(f"Saved {len(self.bookmarked_indices)} bookmarks to {bookmarks_path}")
            return True
        except Exception as e:
            logger.warning(f"Failed to save bookmarks: {e}")
            return False

    def _load_bookmarks(self) -> bool:
        """Load bookmarks from sidecar JSON file if it exists."""
        bookmarks_path = self._get_bookmarks_path()
        if not bookmarks_path or not bookmarks_path.exists():
            return False
        try:
            data = json.loads(bookmarks_path.read_text())
            indices = data.get("bookmarked_indices", [])
            # Filter to valid indices
            valid_indices = {i for i in indices if 0 <= i < len(self.prompts)}
            self.bookmarked_indices = valid_indices
            logger.info(f"Loaded {len(valid_indices)} bookmarks from {bookmarks_path}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load bookmarks: {e}")
            return False

    def set_source_trigger(self, trigger: str) -> None:
        """Set the source trigger phrase (what's in the original prompts).

        Call this if the playlist was loaded without --swap and you want to
        enable auto-trigger-swap on style changes.
        """
        self.source_trigger = trigger
        logger.info(f"Source trigger set to: '{trigger}'")

    def swap_trigger(self, new_trigger: str) -> bool:
        """Swap the trigger phrase to a new one.

        Always swaps from source_trigger (what's in original file) to new_trigger.
        Returns True if swap was applied, False if no change needed.
        """
        if not self.original_prompts:
            logger.warning("Cannot swap trigger: no original prompts stored")
            return False

        if new_trigger == self.current_trigger:
            logger.debug(f"Trigger already set to '{new_trigger}', skipping swap")
            return False

        if not self.source_trigger:
            # No source trigger known - can't do a swap
            # This happens if playlist was loaded without trigger_swap parameter
            logger.warning(
                "Cannot swap trigger: source_trigger not set. "
                "Load playlist with --swap to specify the source trigger phrase."
            )
            return False

        # Always swap from source (what's in file) to new trigger
        self.prompts = _apply_trigger_swap(
            self.original_prompts, self.source_trigger, new_trigger
        )
        logger.info(f"Swapped trigger: '{self.source_trigger}' -> '{new_trigger}'")

        self.current_trigger = new_trigger
        return True

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for API responses."""
        return {
            "source_file": self.source_file,
            "current_index": self.current_index,
            "total": self.total,
            "current_prompt": self.current,
            "has_next": self.has_next,
            "has_prev": self.has_prev,
            "trigger_swap": list(self.trigger_swap) if self.trigger_swap else None,
            "source_trigger": self.source_trigger,
            "current_trigger": self.current_trigger,
            "bookmarked_indices": sorted(self.bookmarked_indices),
            "is_bookmarked": self.is_bookmarked(),
        }

    def preview(self, context: int = 2, max_prompt_len: int = 0) -> dict[str, Any]:
        """Get a preview window around current position.

        Args:
            context: Number of prompts to show before/after current
            max_prompt_len: Max length for prompts (0 = no truncation)
        """
        if not self.prompts:
            return {"prompts": [], "current_index": 0}

        start = max(0, self.current_index - context)
        end = min(len(self.prompts), self.current_index + context + 1)

        items = []
        for i in range(start, end):
            prompt = self.prompts[i]
            # Only truncate if max_prompt_len is set
            if max_prompt_len > 0 and len(prompt) > max_prompt_len:
                prompt = prompt[:max_prompt_len - 3] + "..."
            items.append({
                "index": i,
                "prompt": prompt,
                "current": i == self.current_index,
                "bookmarked": i in self.bookmarked_indices,
            })

        return {
            "prompts": items,
            "current_index": self.current_index,
            "total": self.total,
            "bookmarked_indices": sorted(self.bookmarked_indices),
        }

    def preview_bookmarks(self, max_prompt_len: int = 0) -> dict[str, Any]:
        """Get all bookmarked prompts (for filtered view).

        Args:
            max_prompt_len: Max length for prompts (0 = no truncation)
        """
        if not self.prompts:
            return {"prompts": [], "current_index": 0, "total": 0, "bookmarked_indices": []}

        items = []
        # Include all bookmarked indices plus current (even if not bookmarked)
        indices_to_show = sorted(self.bookmarked_indices | {self.current_index})

        for i in indices_to_show:
            if i < 0 or i >= len(self.prompts):
                continue
            prompt = self.prompts[i]
            if max_prompt_len > 0 and len(prompt) > max_prompt_len:
                prompt = prompt[:max_prompt_len - 3] + "..."
            items.append({
                "index": i,
                "prompt": prompt,
                "current": i == self.current_index,
                "bookmarked": i in self.bookmarked_indices,
            })

        return {
            "prompts": items,
            "current_index": self.current_index,
            "total": self.total,
            "bookmarked_indices": sorted(self.bookmarked_indices),
        }
