#!/usr/bin/env bash
set -euo pipefail

# Orchestrates a local end-to-end run by starting the API server,
# waiting for health, running the Python E2E, and cleaning up.

HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8000}
BASE=${BASE:-http://$HOST:$PORT}
WAIT_ATTEMPTS=${WAIT_ATTEMPTS:-60}  # ~30s at 0.5s intervals

echo "[e2e-local] Base=$BASE"

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[e2e-local] Stopping server pid=$SERVER_PID"
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "[e2e-local] Starting server..."
uv run uvicorn stylgen_v0.main:app --host "$HOST" --port "$PORT" --log-level info &
SERVER_PID=$!
echo "[e2e-local] Server pid=$SERVER_PID"

echo "[e2e-local] Waiting for health..."
ok=false
for i in $(seq 1 "$WAIT_ATTEMPTS"); do
  if curl -fsS "$BASE/health" >/dev/null; then
    ok=true
    break
  fi
  sleep 0.5
done
if [[ "$ok" != true ]]; then
  echo "[e2e-local] Server did not become healthy in time" >&2
  exit 1
fi
echo "[e2e-local] Health OK"

echo "[e2e-local] Running E2E script..."
uv run python scripts/e2e.py "$BASE"
echo "[e2e-local] E2E completed"

