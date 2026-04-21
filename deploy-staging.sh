#!/bin/bash
# Deploy the Livepeer fal wrapper to a fal.ai app.
#
# Reads from env (typically sourced from .env.local):
#   SCOPE_FAL_APP_NAME  required, e.g. "scope-livepeer-emran"
#   SCOPE_FAL_ENV       optional, defaults to "main"
#   SCOPE_FAL_AUTH      optional, defaults to "public"
#
# Exits non-zero on any failure so callers can fail fast.

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

if [ -f "$HERE/.env.local" ]; then
    # shellcheck disable=SC1091
    source "$HERE/.env.local"
fi

: "${SCOPE_FAL_APP_NAME:?Set SCOPE_FAL_APP_NAME in .env.local (see .env.example). Example: scope-livepeer-<your-name>}"
SCOPE_FAL_ENV="${SCOPE_FAL_ENV:-main}"
SCOPE_FAL_AUTH="${SCOPE_FAL_AUTH:-public}"

VENV_DIR="$HERE/.venv-fal"

# Ensure a Python 3.12 venv for fal (matches the scope image).
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating Python 3.12 venv at $VENV_DIR..."
    uv venv --python 3.12 "$VENV_DIR"
fi

if ! "$VENV_DIR/bin/python" -c "import fal" &>/dev/null; then
    echo "Installing fal..."
    uv pip install --python "$VENV_DIR/bin/python" fal
fi

if ! "$VENV_DIR/bin/fal" auth whoami &>/dev/null; then
    echo "Not logged in to fal. Running 'fal auth login' (interactive)..."
    "$VENV_DIR/bin/fal" auth login
fi

echo "Deploying src/scope/cloud/livepeer_fal_app.py"
echo "  → app:  $SCOPE_FAL_APP_NAME"
echo "  → env:  $SCOPE_FAL_ENV"
echo "  → auth: $SCOPE_FAL_AUTH"

"$VENV_DIR/bin/fal" deploy \
    "$HERE/src/scope/cloud/livepeer_fal_app.py" \
    --app "$SCOPE_FAL_APP_NAME" \
    --auth "$SCOPE_FAL_AUTH" \
    --env "$SCOPE_FAL_ENV"
