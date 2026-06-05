# DGX Spark Gemma 4 Omni Agent

Native multimodal Gemma 4 agent brain for NVIDIA DGX Spark.

This repo is a minimal local setup for running `google/gemma-4-12B-it` through the OpenAI-compatible vLLM server on DGX Spark. It positions Gemma 4 as an omni-agent perception and reasoning brain:

- input: text, images, audio, and video-as-frames
- output: text
- spoken output: use a separate TTS service if needed

The catch: Gemma 4 12B is not a voice or video output model. It can reason over multimodal inputs and call tools, but the response channel is still text.

> Personal workstation setup. Not for enterprise use. Use at your own risk.

## Why This Setup

Gemma 4 12B gives the local agent stack native multimodal perception without splitting every input type across separate specialist models.

What this setup adds over the earlier Nemotron/Qwen local-agent stacks:

- native image understanding
- native audio understanding
- video as frame understanding
- tool calling
- reasoning parser
- approximately 25-30 TPS short text with MTP, depending on workload
- very large context beyond 131K if stable
- one OpenAI-compatible endpoint

Practical boundary:

- Gemma 4 12B can process image input.
- Video is processed as frames.
- E2B/E4B/12B support audio input.
- Audio input is limited to about 30 seconds.
- Video input is limited to about 60 seconds at 1 FPS.
- vLLM projects raw 16 kHz waveform frames into LM space for Gemma 4 12B audio.

## Quick Start

```bash
# 1. Verify Docker, GPU visibility, uv, and Hugging Face auth
bash setup/install.sh

# 2. Optional: prefetch model weights into the local HF cache
bash setup/download_model.sh

# 3. Launch vLLM
bash docker/start.sh

# 4. Follow logs
docker logs -f spark-brain
```

Health check:

```bash
curl -sf http://localhost:8000/health && echo OK
```

## Working Launch Command

`docker/start.sh` wraps this known-good command:

```bash
docker run -d --name spark-brain \
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
  -e HF_TOKEN="$HF_TOKEN" \
  -e TRITON_CACHE_DIR=/root/.cache/triton \
  vllm/vllm-openai:gemma4-unified \
    google/gemma-4-12B-it \
    --served-model-name Cogni-Brain \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype bfloat16 \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.75 \
    --max-model-len 196608 \
    --max-num-seqs 2 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --max-num-batched-tokens 8192 \
    --enable-auto-tool-choice \
    --tool-call-parser gemma4 \
    --reasoning-parser gemma4 \
    --limit-mm-per-prompt '{"image":4,"audio":1,"video":1}' \
    --generation-config vllm \
    --safetensors-load-strategy prefetch \
    --speculative-config '{"method":"mtp","model":"google/gemma-4-12B-it-assistant","num_speculative_tokens":5}'
```

## Repository Structure

```text
.
├── benchmark/
│   ├── benchmark_speed.py
│   ├── benchmark_speed_arena.py
│   └── benchmark_smarts.py
├── docker/
│   ├── start.sh
│   ├── status.sh
│   └── stop.sh
├── setup/
│   ├── download_model.sh
│   └── install.sh
└── README.md
```

## Benchmarks

```bash
# Full local endpoint check: TPS, TTFT, concurrency, context, health
uv run benchmark/benchmark_speed.py

# Long llama-benchy sweep with context depths through 190000
uv run benchmark/benchmark_speed_arena.py --save-result benchmark/results_full.csv

# Tool-use smarts checks
# Optional preinstall: uv tool install git+https://github.com/SeraphimSerapis/tool-eval-bench.git
uv run benchmark/benchmark_smarts.py --mode short
```

The default speed benchmark now matches the broader adjacent DGX Spark benchmark shape instead of the minimal smoke test.
