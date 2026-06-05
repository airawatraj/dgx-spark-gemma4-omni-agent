#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-spark-brain}"
PORT="${PORT:-8000}"

echo "=== Gemma 4 vLLM status ==="
echo

if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  STARTED_AT=$(docker inspect "$CONTAINER_NAME" --format '{{.State.StartedAt}}')
  echo "  Container: running (started $STARTED_AT)"
else
  echo "  Container: NOT RUNNING"
fi

if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then
  echo "  API:       healthy (http://localhost:$PORT)"
else
  echo "  API:       not reachable"
fi

echo
echo "=== Models ==="
curl -sf "http://localhost:$PORT/v1/models" 2>/dev/null || echo "  Models endpoint not available"

echo
echo
echo "=== Recent logs ==="
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  docker logs "$CONTAINER_NAME" --tail 10 2>&1 | sed 's/^/  /'
else
  echo "  No container found"
fi
