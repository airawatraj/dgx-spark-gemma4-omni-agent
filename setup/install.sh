#!/usr/bin/env bash
set -euo pipefail

echo "=== DGX Spark Gemma 4 setup check ==="

echo "[1/4] Checking Docker..."
docker version --format 'Docker {{.Server.Version}}' >/dev/null
docker version --format '  Server {{.Server.Version}}'

echo "[2/4] Checking GPU visibility..."
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
  echo "WARNING: nvidia-smi is not on PATH."
fi

echo "[3/4] Checking uv / uvx..."
if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv is not installed."
  echo "Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi
if ! command -v uvx >/dev/null 2>&1; then
  echo "ERROR: uvx is not installed."
  echo "Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi
uv --version
uvx --version

echo "[4/4] Checking Hugging Face auth..."
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "WARNING: HF_TOKEN is not set. Private/gated models may fail to download."
fi
uvx hf auth whoami || {
  echo "WARNING: Hugging Face CLI auth check failed."
  echo "Run: uvx hf auth login"
}

echo
echo "Setup check complete."
echo "Next: bash setup/download_model.sh"
