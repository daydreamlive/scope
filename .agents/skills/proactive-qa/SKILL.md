---
name: proactive-qa
description: Nightly proactive QA smoke test for Scope. Runs Tier 1 critical path checks against the local Scope installation using the gray pipeline (no model downloads). Reports structured pass/fail results. Use when running scheduled QA checks or verifying Scope health after changes.
---

# Proactive QA — Scope Tier 1 Smoke Test

This skill runs a structured set of Tier 1 critical path checks against Scope to verify the real-time pipeline is functional. It uses the `gray` pipeline (grayscale filter — no model downloads required) for fast, reliable smoke testing.

## Prerequisites

- Python/uv environment available in the Scope repo
- OpenCV available (install with `uv pip install opencv-python` if needed)
- Port 8022 available (or kill existing process)

## Step 1 — Generate Test Video

```bash
uv run python -c "
import cv2, numpy as np
w = cv2.VideoWriter('/tmp/qa-test.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (512,512))
frame = np.zeros((512,512,3), dtype=np.uint8)
frame[:] = (0, 0, 255)  # Red frame (BGR)
for _ in range(300):  # 10s at 30fps
    w.write(frame)
w.release()
print('Test video created: /tmp/qa-test.mp4')
" || { echo "FAIL: Could not generate test video"; exit 1; }
```

## Step 2 — Start Scope

Kill any existing process on port 8022 and start a fresh instance from source:

```bash
lsof -ti:8022 | xargs kill -9 2>/dev/null
sleep 1

CUDA_VISIBLE_DEVICES="" uv run daydream-scope --port 8022 > /tmp/scope-qa.log 2>&1 &
SCOPE_PID=$!

# Wait up to 30s for healthy
HEALTHY=0
for i in $(seq 1 30); do
    if curl -s http://localhost:8022/health > /dev/null 2>&1; then
        HEALTHY=1
        break
    fi
    sleep 1
done

if [ $HEALTHY -eq 0 ]; then
    echo "FAIL: Scope did not become healthy within 30s"
    cat /tmp/scope-qa.log | tail -30
    kill $SCOPE_PID 2>/dev/null
    exit 1
fi
echo "Scope started (PID $SCOPE_PID)"
```

## Step 3 — Run Tier 1 Critical Path Checks

Execute all checks in sequence. Record pass/fail for each.

### Check 1: Health

```bash
HEALTH_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8022/health)
[ "$HEALTH_STATUS" = "200" ] && echo "PASS: Health check" || echo "FAIL: Health check (HTTP $HEALTH_STATUS)"
```

### Check 2: Pipeline Load

```bash
LOAD_RESP=$(curl -s -X POST http://localhost:8022/api/v1/pipeline/load \
  -H "Content-Type: application/json" \
  -d '{"pipeline_ids": ["gray"]}')
echo "Pipeline load response: $LOAD_RESP"
echo $LOAD_RESP | python3 -c "import json,sys; d=json.load(sys.stdin); exit(0 if d.get('status') != 'error' else 1)" \
  && echo "PASS: Pipeline load" || echo "FAIL: Pipeline load"
```

### Check 3: Poll Until Loaded

```bash
LOADED=0
for i in $(seq 1 30); do
    STATUS=$(curl -s http://localhost:8022/api/v1/pipeline/status | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('status','unknown'))" 2>/dev/null)
    if [ "$STATUS" = "loaded" ]; then
        LOADED=1
        echo "PASS: Pipeline status loaded (attempt $i)"
        break
    fi
    sleep 2
done
[ $LOADED -eq 0 ] && echo "FAIL: Pipeline never reached loaded status"
```

### Check 4: Session Start

```bash
SESSION_RESP=$(curl -s -X POST http://localhost:8022/api/v1/session/start \
  -H "Content-Type: application/json" \
  -d '{
    "input_mode": "video",
    "graph": {
      "nodes": [
        {"id": "input", "type": "source", "source_mode": "video_file", "source_name": "/tmp/qa-test.mp4"},
        {"id": "gray_pipeline", "type": "pipeline", "pipeline_id": "gray"},
        {"id": "output", "type": "sink"}
      ],
      "edges": [
        {"from": "input", "from_port": "video", "to_node": "gray_pipeline", "to_port": "video", "kind": "stream"},
        {"from": "gray_pipeline", "from_port": "video", "to_node": "output", "to_port": "video", "kind": "stream"}
      ]
    }
  }')
echo "Session start response: $SESSION_RESP"
echo $SESSION_RESP | python3 -c "import json,sys; d=json.load(sys.stdin); exit(0 if d.get('status') != 'error' else 1)" \
  && echo "PASS: Session start" || echo "FAIL: Session start"
```

### Check 5: Wait for Frames (10s)

```bash
sleep 10
echo "Waited 10s for frames to flow"
```

### Check 6: Metrics — frames_in > 0 AND frames_out > 0

```bash
METRICS=$(curl -s http://localhost:8022/api/v1/session/metrics)
echo "Metrics: $METRICS"
FRAMES_IN=$(echo $METRICS | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('frames_in', 0))" 2>/dev/null || echo 0)
FRAMES_OUT=$(echo $METRICS | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('frames_out', 0))" 2>/dev/null || echo 0)
echo "frames_in=$FRAMES_IN  frames_out=$FRAMES_OUT"

if [ "$FRAMES_IN" -gt 0 ] && [ "$FRAMES_OUT" -gt 0 ]; then
    echo "PASS: Frame flow (frames_in=$FRAMES_IN, frames_out=$FRAMES_OUT)"
else
    echo "FAIL: Frame flow (frames_in=$FRAMES_IN, frames_out=$FRAMES_OUT)"
fi
```

### Check 7: Session Stop

```bash
STOP_RESP=$(curl -s -X POST http://localhost:8022/api/v1/session/stop)
echo "Session stop response: $STOP_RESP"
echo $STOP_RESP | python3 -c "import json,sys; d=json.load(sys.stdin); exit(0 if d.get('status') != 'error' else 1)" \
  && echo "PASS: Session stop" || echo "FAIL: Session stop"
```

## Step 4 — Cleanup

```bash
kill $SCOPE_PID 2>/dev/null
lsof -ti:8022 | xargs kill -9 2>/dev/null
```

## Step 5 — Report Results

Produce a structured pass/fail table:

```
## Scope Proactive QA Results — $(date -u +"%Y-%m-%d %H:%M UTC")

| Check | Result | Notes |
|-------|--------|-------|
| 1. Health check | PASS/FAIL | HTTP status code |
| 2. Pipeline load | PASS/FAIL | gray pipeline |
| 3. Pipeline status | PASS/FAIL | reached "loaded" |
| 4. Session start | PASS/FAIL | graph with gray + video file |
| 5. Frame wait | PASS | 10s wait |
| 6. Frame flow | PASS/FAIL | frames_in=N, frames_out=N |
| 7. Session stop | PASS/FAIL | |

**Overall: PASS / FAIL**
Key metrics: frames_in=N, frames_out=N
```

## Step 6 — On Failure

If any check fails, fetch logs and include them in the output:

```bash
curl -s "http://localhost:8022/api/v1/logs/tail?lines=50" 2>/dev/null || cat /tmp/scope-qa.log | tail -50
```

If `DISCORD_BOT_TOKEN` is set in the environment, post failure summary to the `#scope-engineering` Discord channel. The bot token and channel ID should be available in the runtime environment.

## Full Automated Script

For convenience, here is the full QA sequence as a single script. Copy and run it:

```bash
#!/bin/bash
set -e
cd /paperclip/instances/default/projects/62f01501-054b-47be-a7ac-2561c650f680/1c90122e-8d87-49fc-bb7b-8326f360af69/scope

PASS=0; FAIL=0
check() { local name="$1" result="$2" notes="$3"
  if [ "$result" = "PASS" ]; then PASS=$((PASS+1)); else FAIL=$((FAIL+1)); fi
  echo "| $name | $result | $notes |"
}

# Generate test video
uv run python -c "
import cv2, numpy as np
w = cv2.VideoWriter('/tmp/qa-test.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (512,512))
frame = np.zeros((512,512,3), dtype=np.uint8); frame[:] = (0, 0, 255)
[w.write(frame) for _ in range(300)]; w.release()
" 2>/dev/null || uv pip install opencv-python -q && uv run python -c "
import cv2, numpy as np
w = cv2.VideoWriter('/tmp/qa-test.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (512,512))
frame = np.zeros((512,512,3), dtype=np.uint8); frame[:] = (0, 0, 255)
[w.write(frame) for _ in range(300)]; w.release()
"

# Start Scope
lsof -ti:8022 | xargs kill -9 2>/dev/null; sleep 1
CUDA_VISIBLE_DEVICES="" uv run daydream-scope --port 8022 > /tmp/scope-qa.log 2>&1 &
SCOPE_PID=$!
HEALTHY=0
for i in $(seq 1 30); do curl -s http://localhost:8022/health > /dev/null 2>&1 && HEALTHY=1 && break; sleep 1; done

echo "## Scope Proactive QA — $(date -u +'%Y-%m-%d %H:%M UTC')"
echo "| Check | Result | Notes |"
echo "|-------|--------|-------|"

[ $HEALTHY -eq 1 ] && check "1. Health" "PASS" "HTTP 200" || { check "1. Health" "FAIL" "No response"; kill $SCOPE_PID 2>/dev/null; exit 1; }

LOAD_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST http://localhost:8022/api/v1/pipeline/load \
  -H "Content-Type: application/json" -d '{"pipeline_ids": ["gray"]}')
[ "$LOAD_CODE" = "200" ] && check "2. Pipeline load" "PASS" "gray" || check "2. Pipeline load" "FAIL" "HTTP $LOAD_CODE"

LOADED=0
for i in $(seq 1 30); do
  S=$(curl -s http://localhost:8022/api/v1/pipeline/status | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('status',''))" 2>/dev/null)
  [ "$S" = "loaded" ] && LOADED=1 && break; sleep 2
done
[ $LOADED -eq 1 ] && check "3. Pipeline status" "PASS" "loaded" || check "3. Pipeline status" "FAIL" "timeout"

SC=$(curl -s -o /dev/null -w "%{http_code}" -X POST http://localhost:8022/api/v1/session/start \
  -H "Content-Type: application/json" \
  -d '{"input_mode":"video","graph":{"nodes":[{"id":"input","type":"source","source_mode":"video_file","source_name":"/tmp/qa-test.mp4"},{"id":"gray_pipeline","type":"pipeline","pipeline_id":"gray"},{"id":"output","type":"sink"}],"edges":[{"from":"input","from_port":"video","to_node":"gray_pipeline","to_port":"video","kind":"stream"},{"from":"gray_pipeline","from_port":"video","to_node":"output","to_port":"video","kind":"stream"}]}}')
[ "$SC" = "200" ] && check "4. Session start" "PASS" "graph+video" || check "4. Session start" "FAIL" "HTTP $SC"

sleep 10; check "5. Frame wait" "PASS" "10s"

METRICS=$(curl -s http://localhost:8022/api/v1/session/metrics)
FIN=$(echo $METRICS | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('frames_in',0))" 2>/dev/null || echo 0)
FOUT=$(echo $METRICS | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('frames_out',0))" 2>/dev/null || echo 0)
([ "$FIN" -gt 0 ] && [ "$FOUT" -gt 0 ]) && check "6. Frame flow" "PASS" "in=$FIN out=$FOUT" || check "6. Frame flow" "FAIL" "in=$FIN out=$FOUT"

curl -s -X POST http://localhost:8022/api/v1/session/stop > /dev/null
check "7. Session stop" "PASS" ""

echo ""
echo "**Results: $PASS passed, $FAIL failed**"
kill $SCOPE_PID 2>/dev/null; lsof -ti:8022 | xargs kill -9 2>/dev/null

[ $FAIL -gt 0 ] && { echo ""; echo "### Logs (last 50 lines)"; curl -s "http://localhost:8022/api/v1/logs/tail?lines=50" 2>/dev/null || cat /tmp/scope-qa.log | tail -50; exit 1; }
exit 0
```
