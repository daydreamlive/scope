#!/bin/bash
# Run daydream-scope in livepeer cloud mode.
#
# Requires `.env.local` (gitignored) exporting at minimum:
#   SCOPE_CLOUD_APP_ID   e.g. daydream/scope-livepeer-<user>/ws
#   SCOPE_CLOUD_API_KEY  daydream cloud API key (sk_...)
# Optional in `.env.local`:
#   SCOPE_USER_ID        daydream user id (used by test-cloud-connect.sh)
#   LIVEPEER_DEBUG=1     surface per-orchestrator rejection reasons
#
# See .env.example for a template.

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

if [ -f "$HERE/.env.local" ]; then
    # shellcheck disable=SC1091
    source "$HERE/.env.local"
fi

: "${SCOPE_CLOUD_APP_ID:?Set SCOPE_CLOUD_APP_ID in .env.local (see .env.example)}"

# Env vars sourced from .env.local are already exported; the previous
# attempt to inline-prefix them with ${VAR:+VAR=$VAR} broke under
# bash's word-splitting rules ("SCOPE_CLOUD_API_KEY=sk_... command not
# found"). Just re-export and exec.
export SCOPE_CLOUD_MODE=livepeer
exec uv run daydream-scope "$@"
