#!/bin/bash
# End-to-end cloud-connect test for the livepeer fal deploy.
#
# Flow:
#   1. (optional) push current branch to origin
#   2. (optional) wait for CI `build-cloud` to succeed for HEAD
#   3. (optional) run deploy-staging.sh to deploy the fal wrapper
#   4. start daydream-scope locally via ./run-app.sh
#   5. POST /api/v1/cloud/connect
#   6. poll /api/v1/cloud/status until connected, errored, or timed out
#   7. (--full-session) load pipeline, start session, wait for frames,
#      stop session, cloud disconnect
#
# Exit codes (bisect-friendly):
#   0  success (connected, and if --full-session then frames flowed)
#   1  cloud reported error
#   2  timed out waiting for connect / pipeline / frames
#   3  infra failure (push / CI / deploy / scope startup)
#   4  session-level failure (pipeline load, session start, no frames)

set -euo pipefail

PORT="${PORT:-8000}"
TIMEOUT_CONNECT="${TIMEOUT_CONNECT:-180}"
TIMEOUT_HEALTH="${TIMEOUT_HEALTH:-60}"
TIMEOUT_CI="${TIMEOUT_CI:-1800}"
TIMEOUT_PIPELINE="${TIMEOUT_PIPELINE:-300}"
TIMEOUT_FRAMES="${TIMEOUT_FRAMES:-60}"
PIPELINE_ID="${PIPELINE_ID:-passthrough}"
TEST_VIDEO="${TEST_VIDEO:-/tmp/test_input.mp4}"
SKIP_PUSH=0
SKIP_BUILD_WAIT=0
SKIP_DEPLOY=0
KEEP_SCOPE=0
FULL_SESSION=0

usage() {
    cat <<EOF
Usage: $0 [options]

Options:
  --skip-push         do not git push
  --skip-build-wait   do not wait for GitHub Actions build-cloud
  --skip-deploy       do not run deploy-staging.sh
  --keep-scope        leave scope running after test (do not kill)
  --full-session      after connect, load pipeline + start session +
                      verify frames + stop + cloud-disconnect (exercises
                      full Kafka event stream: pipeline_loaded /
                      session_created / stream_started / stream_heartbeat)
  --port N            scope port (default 8000, env PORT)
  -h, --help          show this help

Env overrides: PORT, TIMEOUT_CONNECT, TIMEOUT_HEALTH, TIMEOUT_CI,
               TIMEOUT_PIPELINE, TIMEOUT_FRAMES, PIPELINE_ID, TEST_VIDEO
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-push) SKIP_PUSH=1; shift ;;
        --skip-build-wait) SKIP_BUILD_WAIT=1; shift ;;
        --skip-deploy) SKIP_DEPLOY=1; shift ;;
        --keep-scope) KEEP_SCOPE=1; shift ;;
        --full-session) FULL_SESSION=1; shift ;;
        --port) PORT="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown arg: $1"; usage; exit 3 ;;
    esac
done

SCOPE_URL="http://localhost:${PORT}"
LOG_DIR="/tmp/test-cloud-connect"
mkdir -p "$LOG_DIR"
DRIVER_LOG="$LOG_DIR/driver.log"
SCOPE_LOG="$LOG_DIR/scope.log"
: > "$DRIVER_LOG"
: > "$SCOPE_LOG"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$DRIVER_LOG"; }
fail() { log "FAIL: $*"; exit "${2:-3}"; }

SCOPE_PID=""
cleanup() {
    local ec=$?
    if [[ $KEEP_SCOPE -eq 0 && -n "$SCOPE_PID" ]]; then
        log "Stopping scope (pid=$SCOPE_PID)"
        kill "$SCOPE_PID" 2>/dev/null || true
        wait "$SCOPE_PID" 2>/dev/null || true
    elif [[ $KEEP_SCOPE -eq 1 && -n "$SCOPE_PID" ]]; then
        log "Leaving scope running (pid=$SCOPE_PID, logs $SCOPE_LOG)"
    fi
    log "Exit code: $ec"
    exit $ec
}
trap cleanup EXIT INT TERM

# JSON field extractor via python3 (jq not available everywhere)
json_get() {
    # $1 = field path (e.g. ".connected" or ".error")
    # stdin = json
    python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
except Exception as e:
    print(f'<parse_err:{e}>', file=sys.stderr)
    sys.exit(0)
path = '$1'.lstrip('.').split('.')
v = d
for p in path:
    if isinstance(v, dict):
        v = v.get(p)
    else:
        v = None
        break
if v is None:
    print('')
elif isinstance(v, bool):
    print('true' if v else 'false')
else:
    print(v)
"
}

# --- 1. Push -------------------------------------------------------
if [[ $SKIP_PUSH -eq 0 ]]; then
    if ! git diff-index --quiet HEAD --; then
        fail "Uncommitted changes present. Commit first or pass --skip-push." 3
    fi
    BRANCH=$(git rev-parse --abbrev-ref HEAD)
    log "Pushing $BRANCH to origin..."
    git push origin "$BRANCH" 2>&1 | tee -a "$DRIVER_LOG"
fi

SHA=$(git rev-parse HEAD)
SHORT_SHA=$(git rev-parse --short HEAD)
log "Testing commit: $SHORT_SHA"

# --- 2. Wait for CI build-cloud ------------------------------------
if [[ $SKIP_BUILD_WAIT -eq 0 ]]; then
    log "Locating CI build-cloud run for $SHORT_SHA..."
    START=$(date +%s)
    RUN_ID=""
    while [[ -z "$RUN_ID" ]]; do
        if [[ $(($(date +%s) - START)) -gt 180 ]]; then
            fail "No CI run found for $SHORT_SHA after 3 min" 3
        fi
        RUN_ID=$(gh run list --workflow=docker-build.yml --commit "$SHA" \
            --json databaseId --jq '.[0].databaseId' 2>/dev/null || true)
        [[ -z "$RUN_ID" ]] && sleep 5
    done
    log "Watching CI run $RUN_ID (timeout ${TIMEOUT_CI}s)..."
    if ! timeout "$TIMEOUT_CI" gh run watch "$RUN_ID" --exit-status --interval 15 \
            2>&1 | tee -a "$DRIVER_LOG"; then
        fail "CI run $RUN_ID did not succeed" 3
    fi
    log "CI succeeded"
fi

# --- 3. Deploy -----------------------------------------------------
if [[ $SKIP_DEPLOY -eq 0 ]]; then
    if [[ ! -x ./deploy-staging.sh ]]; then
        fail "./deploy-staging.sh not found or not executable. Create one that runs \`fal deploy src/scope/cloud/livepeer_fal_app.py --app <your-app> --auth public --env main\`, or pass --skip-deploy." 3
    fi
    log "Running ./deploy-staging.sh..."
    if ! ./deploy-staging.sh 2>&1 | tee -a "$DRIVER_LOG"; then
        fail "deploy-staging.sh failed" 3
    fi
    log "Deploy completed"
fi

# --- 4. Start scope ------------------------------------------------
log "Freeing port $PORT..."
lsof -ti:"$PORT" 2>/dev/null | xargs -r kill -9 2>/dev/null || true
sleep 1

log "Starting scope (logs: $SCOPE_LOG)..."
./run-app.sh --port "$PORT" > "$SCOPE_LOG" 2>&1 &
SCOPE_PID=$!
log "Scope pid=$SCOPE_PID"

log "Waiting for /health..."
START=$(date +%s)
while ! curl -sf "$SCOPE_URL/health" > /dev/null 2>&1; do
    if [[ $(($(date +%s) - START)) -gt $TIMEOUT_HEALTH ]]; then
        log "Scope health timeout. Last 50 log lines:"
        tail -50 "$SCOPE_LOG" | tee -a "$DRIVER_LOG"
        fail "Scope did not become healthy" 3
    fi
    if ! kill -0 "$SCOPE_PID" 2>/dev/null; then
        log "Scope process died. Last 50 log lines:"
        tail -50 "$SCOPE_LOG" | tee -a "$DRIVER_LOG"
        fail "Scope process exited" 3
    fi
    sleep 1
done
log "Scope healthy"

# --- 5. Connect ----------------------------------------------------
# Source .env.local so SCOPE_USER_ID is available for the connect body.
if [ -f "$(dirname "$0")/.env.local" ]; then
    # shellcheck disable=SC1091
    source "$(dirname "$0")/.env.local"
fi
CONNECT_BODY='{}'
if [[ -n "${SCOPE_USER_ID:-}" ]]; then
    CONNECT_BODY=$(python3 -c "import json,os; print(json.dumps({'user_id': os.environ['SCOPE_USER_ID']}))")
fi
log "POST /api/v1/cloud/connect (user_id=${SCOPE_USER_ID:-<unset>})"
CONNECT_RESP=$(curl -sf -X POST "$SCOPE_URL/api/v1/cloud/connect" \
    -H 'Content-Type: application/json' -d "$CONNECT_BODY")
log "Connect response: $CONNECT_RESP"

# --- 6. Poll status ------------------------------------------------
log "Polling /api/v1/cloud/status (timeout ${TIMEOUT_CONNECT}s)..."
START=$(date +%s)
LAST_STAGE=""
while true; do
    ELAPSED=$(($(date +%s) - START))
    if [[ $ELAPSED -gt $TIMEOUT_CONNECT ]]; then
        log "TIMEOUT after ${ELAPSED}s"
        curl -s "$SCOPE_URL/api/v1/cloud/status" | tee -a "$DRIVER_LOG"
        echo
        log "Last 30 scope log lines:"
        tail -30 "$SCOPE_LOG" | tee -a "$DRIVER_LOG"
        exit 2
    fi
    STATUS=$(curl -s "$SCOPE_URL/api/v1/cloud/status")
    CONNECTED=$(echo "$STATUS" | json_get ".connected")
    ERROR=$(echo "$STATUS" | json_get ".error")
    STAGE=$(echo "$STATUS" | json_get ".connect_stage")

    if [[ "$CONNECTED" == "true" ]]; then
        log "CONNECTED (${ELAPSED}s)"
        echo "$STATUS" | tee -a "$DRIVER_LOG"
        echo
        break
    fi
    if [[ -n "$ERROR" && "$ERROR" != "None" ]]; then
        log "CLOUD ERROR (${ELAPSED}s): $ERROR"
        echo "$STATUS" | tee -a "$DRIVER_LOG"
        echo
        log "Last 30 scope log lines:"
        tail -30 "$SCOPE_LOG" | tee -a "$DRIVER_LOG"
        exit 1
    fi
    if [[ "$STAGE" != "$LAST_STAGE" ]]; then
        log "  stage: $STAGE (${ELAPSED}s)"
        LAST_STAGE="$STAGE"
    fi
    sleep 3
done

if [[ $FULL_SESSION -eq 0 ]]; then
    exit 0
fi

# --- 7. Full session: pipeline + session + frames + cleanup --------

# 7a. Ensure test video exists
if [[ ! -f "$TEST_VIDEO" ]]; then
    log "Creating $TEST_VIDEO (512x512 red frames @30fps, 10s)..."
    uv run --with opencv-python --with numpy python -c "
import cv2, numpy as np
w = cv2.VideoWriter('$TEST_VIDEO', cv2.VideoWriter_fourcc(*'mp4v'), 30, (512, 512))
frame = np.zeros((512, 512, 3), dtype=np.uint8)
frame[:] = (0, 0, 255)
for _ in range(300):
    w.write(frame)
w.release()
" 2>&1 | tee -a "$DRIVER_LOG"
    [[ -f "$TEST_VIDEO" ]] || fail "Failed to create $TEST_VIDEO" 4
fi
log "Test video: $TEST_VIDEO"

# 7b. Load pipeline
log "POST /api/v1/pipeline/load (pipeline_id=$PIPELINE_ID)"
LOAD_BODY=$(python3 -c "import json; print(json.dumps({'pipeline_ids': ['$PIPELINE_ID']}))")
LOAD_RESP=$(curl -sf -X POST "$SCOPE_URL/api/v1/pipeline/load" \
    -H 'Content-Type: application/json' -d "$LOAD_BODY") \
    || fail "pipeline/load request failed" 4
log "Load response: $LOAD_RESP"

# 7c. Poll pipeline status — require both status=loaded AND pipeline_id
# matches what we loaded (cloud-mode status can show a stale "loaded"
# from a previous session for a brief window after POST).
log "Polling /api/v1/pipeline/status (timeout ${TIMEOUT_PIPELINE}s)..."
# Give the async load a moment to propagate before first check.
sleep 5
START=$(date +%s)
LAST_KEY=""
while true; do
    ELAPSED=$(($(date +%s) - START))
    if [[ $ELAPSED -gt $TIMEOUT_PIPELINE ]]; then
        log "Pipeline load TIMEOUT after ${ELAPSED}s. Last status:"
        curl -s "$SCOPE_URL/api/v1/pipeline/status" | tee -a "$DRIVER_LOG"
        echo
        exit 2
    fi
    PSTATUS=$(curl -s "$SCOPE_URL/api/v1/pipeline/status")
    PS=$(echo "$PSTATUS" | json_get ".status")
    PID=$(echo "$PSTATUS" | json_get ".pipeline_id")
    STAGE=$(echo "$PSTATUS" | json_get ".loading_stage")
    if [[ "$PS" == "loaded" && "$PID" == "$PIPELINE_ID" ]]; then
        log "Pipeline loaded (${ELAPSED}s, id=$PID)"
        break
    fi
    if [[ "$PS" == "error" ]]; then
        log "Pipeline load ERROR after ${ELAPSED}s"
        echo "$PSTATUS" | tee -a "$DRIVER_LOG"
        echo
        exit 4
    fi
    KEY="${PS}|${PID}|${STAGE}"
    if [[ "$KEY" != "$LAST_KEY" ]]; then
        log "  pipeline status=$PS pipeline_id=$PID stage=$STAGE (${ELAPSED}s)"
        LAST_KEY="$KEY"
    fi
    sleep 3
done

# 7d. Start session with video-file input
log "POST /api/v1/session/start (pipeline=$PIPELINE_ID, source=$TEST_VIDEO)"
SESSION_BODY=$(python3 -c "
import json, os
body = {
    'pipeline_id': '$PIPELINE_ID',
    'input_mode': 'video',
    'input_source': {
        'enabled': True,
        'source_type': 'video_file',
        'source_name': os.environ.get('TEST_VIDEO', '$TEST_VIDEO'),
    },
}
print(json.dumps(body))
")
SESSION_RESP=$(curl -s -o /tmp/session_start.json -w '%{http_code}' \
    -X POST "$SCOPE_URL/api/v1/session/start" \
    -H 'Content-Type: application/json' -d "$SESSION_BODY") || true
if [[ "$SESSION_RESP" != "200" ]]; then
    log "session/start failed (http $SESSION_RESP)"
    cat /tmp/session_start.json | tee -a "$DRIVER_LOG"
    echo
    exit 4
fi
log "Session started"

# 7e. Wait for frames
log "Waiting for frames to flow (timeout ${TIMEOUT_FRAMES}s)..."
START=$(date +%s)
FRAMES_IN=0
FRAMES_OUT=0
while true; do
    ELAPSED=$(($(date +%s) - START))
    if [[ $ELAPSED -gt $TIMEOUT_FRAMES ]]; then
        log "Frame-wait TIMEOUT (frames_in=$FRAMES_IN frames_out=$FRAMES_OUT)"
        curl -s "$SCOPE_URL/api/v1/session/metrics" | tee -a "$DRIVER_LOG"
        echo
        exit 2
    fi
    METRICS=$(curl -s "$SCOPE_URL/api/v1/session/metrics")
    FRAMES_IN=$(echo "$METRICS" | json_get ".frames_in")
    FRAMES_OUT=$(echo "$METRICS" | json_get ".frames_out")
    FRAMES_IN=${FRAMES_IN:-0}
    FRAMES_OUT=${FRAMES_OUT:-0}
    if [[ "$FRAMES_OUT" != "0" && "$FRAMES_OUT" != "" ]]; then
        log "Frames flowing: in=$FRAMES_IN out=$FRAMES_OUT (${ELAPSED}s)"
        break
    fi
    sleep 2
done

# 7f. Let it run a bit so stream_heartbeat events fire
log "Streaming for 10s to let heartbeat events fire..."
sleep 10
METRICS=$(curl -s "$SCOPE_URL/api/v1/session/metrics")
log "Final metrics: $METRICS"

# 7g. Stop session
log "POST /api/v1/session/stop"
curl -sf -X POST "$SCOPE_URL/api/v1/session/stop" > /dev/null \
    || log "session/stop returned non-2xx (continuing)"

# 7h. Cloud disconnect (explicit, to cleanly fire websocket_disconnected)
log "POST /api/v1/cloud/disconnect"
curl -sf -X POST "$SCOPE_URL/api/v1/cloud/disconnect" > /dev/null \
    || log "cloud/disconnect returned non-2xx (continuing)"

log "Full-session test OK"
exit 0
