#!/bin/bash
# Script to run VibeVoice test with proper environment setup

# Run the test with uv (vibevoice is installed as a package dependency)
uv run python test_vibevoice.py "$@"
