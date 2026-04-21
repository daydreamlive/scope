"""``python -m evals`` entry point."""

from __future__ import annotations

import sys

from .runner import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
