#!/bin/bash
# Script to run VibeVoice test with proper environment setup

# Add VibeVoice to Python path
export PYTHONPATH="/home/user/VibeVoice:$PYTHONPATH"

# Run the test with uv
uv run python test_vibevoice.py "$@"
