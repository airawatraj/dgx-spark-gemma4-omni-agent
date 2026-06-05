#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-google/gemma-4-12B-it}"
SPECULATIVE_MODEL_ID="${SPECULATIVE_MODEL_ID:-google/gemma-4-12B-it-assistant}"

echo "=== Downloading Gemma 4 models into the Hugging Face cache ==="
echo "  Main model:        $MODEL_ID"
echo "  Speculative model: $SPECULATIVE_MODEL_ID"
echo

if ! command -v uvx >/dev/null 2>&1; then
  echo "ERROR: uvx is not available."
  echo "Run: bash setup/install.sh"
  exit 1
fi

uvx hf download "$MODEL_ID"
uvx hf download "$SPECULATIVE_MODEL_ID"

echo
echo "Download complete."
echo "Next: bash docker/start.sh"
