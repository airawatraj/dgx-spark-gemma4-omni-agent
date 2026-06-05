#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-spark-brain}"
VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:gemma4-unified}"
MODEL_ID="${MODEL_ID:-google/gemma-4-12B-it}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Cogni-Brain}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
DTYPE="${DTYPE:-bfloat16}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.75}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-196608}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
LIMIT_MM_PER_PROMPT="${LIMIT_MM_PER_PROMPT:-{\"image\":4,\"audio\":1,\"video\":1}}"
SPECULATIVE_CONFIG="${SPECULATIVE_CONFIG:-{\"method\":\"mtp\",\"model\":\"google/gemma-4-12B-it-assistant\",\"num_speculative_tokens\":5}}"

echo "=== vLLM Gemma 4 preflight ==="
echo "  Model ID:        $MODEL_ID"
echo "  Served name:     $SERVED_MODEL_NAME"
echo "  Container:       $CONTAINER_NAME"
echo "  Image:           $VLLM_IMAGE"
echo "  Port:            $PORT"
echo "  Max model len:   $MAX_MODEL_LEN"
echo "  Max num seqs:    $MAX_NUM_SEQS"
echo

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "WARNING: HF_TOKEN is not set. The container may fail if the model requires auth."
  echo
fi

if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  echo "Cleaning up existing container..."
  docker stop "$CONTAINER_NAME" >/dev/null 2>&1 || true
  docker rm "$CONTAINER_NAME" >/dev/null 2>&1 || true
fi

mkdir -p "$HOME/.cache/huggingface" "$HOME/.cache/triton" "$HOME/.cache/vllm"

echo "Starting vLLM..."
docker run -d --name "$CONTAINER_NAME" \
  --gpus all \
  --restart=unless-stopped \
  --ipc=host \
  --network host \
  --shm-size=32gb \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$HOME/.cache/triton:/root/.cache/triton" \
  -v "$HOME/.cache/vllm:/root/.cache/vllm" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e TRITON_CACHE_DIR=/root/.cache/triton \
  "$VLLM_IMAGE" \
    "$MODEL_ID" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --host "$HOST" \
    --port "$PORT" \
    --dtype "$DTYPE" \
    --kv-cache-dtype "$KV_CACHE_DTYPE" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --enable-auto-tool-choice \
    --tool-call-parser gemma4 \
    --reasoning-parser gemma4 \
    --limit-mm-per-prompt "$LIMIT_MM_PER_PROMPT" \
    --generation-config vllm \
    --safetensors-load-strategy prefetch \
    --speculative-config "$SPECULATIVE_CONFIG"

echo
echo "Container started."
echo "Next: docker logs -f $CONTAINER_NAME"
echo "Ready check: curl -sf http://localhost:$PORT/health && echo OK"
