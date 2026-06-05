# DGX Spark Gemma 4 Omni Agent

Native multimodal Gemma 4 agent brain for NVIDIA DGX Spark.

This repo is a minimal local setup for running [Gemma 4 12B](https://hfviewer.com/google/gemma-4-12B-it) through the OpenAI-compatible vLLM server on DGX Spark. It positions Gemma 4 as an omni-agent perception and reasoning brain:

- input: text, images, audio, and video-as-frames
- output: text
- spoken output: use a separate TTS service if needed

The catch: Gemma 4 12B is not a voice or video output model. It can reason over multimodal inputs and call tools, but the response channel is still text.

> Personal workstation setup. Not for enterprise use. Use at your own risk.

## Which DGX Spark Agent Repo

Use the repo that matches the workload:

| Goal | Repo | Fit |
|---|---|---|
| Speed-oriented local agent stack | [airawatraj/dgx-spark-qwen-super-agent](https://github.com/airawatraj/dgx-spark-qwen-super-agent) | ✅ Best fit |
| Larger reasoning model setup | [airawatraj/dgx-spark-nemotron-super-agent](https://github.com/airawatraj/dgx-spark-nemotron-super-agent) | ✅ Best fit |
| Native multimodal inputs | [airawatraj/dgx-spark-gemma4-omni-agent](https://github.com/airawatraj/dgx-spark-gemma4-omni-agent) | ✅ Best fit |
| Larger configured context window | [airawatraj/dgx-spark-gemma4-omni-agent](https://github.com/airawatraj/dgx-spark-gemma4-omni-agent) | ⚠️ Configured for `196,608` max model length; benchmark depths test up to `190,000` |
| Voice or video output | [airawatraj/dgx-spark-gemma4-omni-agent](https://github.com/airawatraj/dgx-spark-gemma4-omni-agent) | ❌ Use separate TTS/video tooling |

Current Gemma 4 12B Omni Agent measurements on this setup: approximately `25-30 tok/s` short-text generation with MTP, and `83/100` on `tool-eval-bench --short`. Treat these as local configuration results, not universal model claims.

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

## Runtime Defaults

`docker/start.sh` is the canonical launch path. It starts `vllm/vllm-openai:gemma4-unified` with [Gemma 4 12B](https://hfviewer.com/google/gemma-4-12B-it), serves it as `Cogni-Brain`, and enables Gemma 4 tool/reasoning parsers, multimodal limits, prefix caching, chunked prefill, and MTP speculative decoding with the [Gemma 4 12B assistant model](https://hfviewer.com/google/gemma-4-12B-it-assistant).

Common overrides:

```bash
PORT=8001 MAX_MODEL_LEN=131072 bash docker/start.sh
MODEL_ID=google/gemma-4-12B-it SERVED_MODEL_NAME=Cogni-Brain bash docker/start.sh
```

## Repository Structure

```text
.
├── benchmark/
│   ├── benchmark_speed.py
│   ├── benchmark_speed_arena.py
│   └── benchmark_smarts.py
├── assets/
│   ├── benchmark_test_1-3.png
│   ├── benchmark_test_4-5.png
│   ├── benchmark_smarts_1.png
│   ├── benchmark_smarts_2.png
│   └── benchmark_smarts_3.png
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

### Local Speed and Context Benchmark

<p align="center">
  <img src="./assets/benchmark_test_1-3.png" width="600" alt="Local speed benchmark tests 1 through 3">
</p>

<p align="center">
  <img src="./assets/benchmark_test_4-5.png" width="600" alt="Local speed benchmark tests 4 and 5">
</p>

### Tool-Eval-Bench Capability Benchmark

<p align="center">
  <img src="./assets/benchmark_smarts_1.png" width="600" alt="Tool-eval benchmark summary 1">
</p>

<p align="center">
  <img src="./assets/benchmark_smarts_2.png" width="600" alt="Tool-eval benchmark summary 2">
</p>

<p align="center">
  <img src="./assets/benchmark_smarts_3.png" width="600" alt="Tool-eval benchmark summary 3">
</p>
