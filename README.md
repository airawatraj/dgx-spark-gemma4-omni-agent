# DGX Spark Gemma 4 Omni Agent

Native multimodal Gemma 4 agent brain for NVIDIA DGX Spark.

This repo is a minimal local setup for running [Gemma 4 12B](https://hfviewer.com/google/gemma-4-12B-it) through the OpenAI-compatible vLLM server on DGX Spark.

Gemma 4 is positioned here as an omni-agent perception and reasoning brain:

- input: text, images, audio, and video-as-frames
- output: text
- spoken output: use a separate TTS service if needed

The catch: Gemma 4 12B is not a voice or video output model. It can reason over multimodal inputs and call tools, but the response channel is still text.

> Personal workstation setup. Not for enterprise use. Use at your own risk.

## Which DGX Spark Agent Repo?

Use the repo that matches the workload:

| Workload | Repo | Why |
|---|---|---|
| Fast local text/tool agent | [dgx-spark-qwen-super-agent](https://github.com/airawatraj/dgx-spark-qwen-super-agent) | Speed-oriented Atlas/NVFP4 stack |
| Larger reasoning model | [dgx-spark-nemotron-super-agent](https://github.com/airawatraj/dgx-spark-nemotron-super-agent) | Larger model, stable long-context text agent |
| Native multimodal agent | [dgx-spark-gemma4-omni-agent](https://github.com/airawatraj/dgx-spark-gemma4-omni-agent) | Text, image, audio, video-as-frames, tools |
| Voice/video output | Separate service required | Gemma 4 outputs text; use TTS/video tooling externally |

Current Gemma 4 12B Omni Agent measurements on this setup: approximately `25-30 tok/s` on local short-text generation with MTP, `22.11 tok/s` on the spark-arena `tg128` submission, and `83/100` on `tool-eval-bench --short`. Treat these as local configuration results, not universal model claims.

## Why This Setup

Gemma 4 12B adds native multimodal perception to the local DGX Spark agent stack without splitting every input type across separate specialist models.

This setup provides:

- native image understanding
- native audio understanding
- video-as-frames understanding
- tool calling
- reasoning parser support
- approximately `25-30 TPS` short text generation with MTP, depending on workload
- large configured context beyond `131K`
- one OpenAI-compatible endpoint

Practical boundary:

- Gemma 4 12B can process image input.
- Video is processed as frames.
- This Gemma 4 12B setup supports audio input through vLLM.
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

Health and model checks:

```bash
curl -sf http://localhost:8000/health && echo OK
curl -sf http://localhost:8000/v1/models
```

Helper scripts:

```bash
bash docker/status.sh
bash docker/stop.sh
```

## Runtime Defaults

`docker/start.sh` is the canonical launch path. It starts `vllm/vllm-openai:gemma4-unified` with [Gemma 4 12B](https://hfviewer.com/google/gemma-4-12B-it), serves it as `Cogni-Brain`, and enables Gemma 4 tool/reasoning parsers, multimodal limits, prefix caching, chunked prefill, and MTP speculative decoding with the [Gemma 4 12B assistant model](https://hfviewer.com/google/gemma-4-12B-it-assistant).

Common overrides:

```bash
PORT=8001 MAX_MODEL_LEN=131072 bash docker/start.sh
MODEL_ID=google/gemma-4-12B-it SERVED_MODEL_NAME=Cogni-Brain bash docker/start.sh
SPECULATIVE_MODEL_ID=google/gemma-4-12B-it-assistant NUM_SPECULATIVE_TOKENS=5 bash docker/start.sh
```

For unusual vLLM builds, `SPECULATIVE_CONFIG` can still be set directly to replace the generated MTP JSON.

## Field Notes

See [`FIELD_NOTES.md`](FIELD_NOTES.md) for debugging notes and configuration tradeoffs discovered while turning the simple Gemma 4 launch recipe into a stable working setup on DGX Spark.

It covers:

- FP8 weight quantization issues on SM121
- MTP assistant configuration
- `TRITON_ATTN` startup confusion
- why `196608` became the daily context target
- why `0.75` memory utilization is the stability line
- multimodal warmup warnings
- tool/reasoning parser tradeoffs

## Repository Structure

```text
.
├── benchmark/
│   ├── benchmark_speed.py
│   ├── benchmark_speed_arena.py
│   └── benchmark_smarts.py
├── assets/
├── docker/
│   ├── start.sh
│   ├── status.sh
│   └── stop.sh
├── setup/
│   ├── download_model.sh
│   └── install.sh
├── FIELD_NOTES.md
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

## Benchmark Results

> Results may vary depending on runtime configuration, concurrency, context length, upstream benchmark versions, multimodal settings, and memory allocation.

### Official spark-arena Submission

> Benchmarked using [llama-benchy](https://github.com/eugr/llama-benchy) with the standardized spark-arena methodology. This run used the Gemma 4 omni-agent profile rather than a stripped text-only maximum-throughput profile.

| Metric | Result |
|---|---|
| Single session TPS (`tg128`) | **22.11 tok/s** |
| Runtime | **vLLM** |
| Model | **google/gemma-4-12B-it** |
| Weight dtype | **BF16** |
| KV cache dtype | **FP8** |
| Speculative decoding | **Gemma 4 MTP assistant, 5 speculative tokens** |
| Configured max context | **196,608 tokens** |
| GPU memory utilization | **0.75** |
| Hardware | **Single DGX Spark (GB10)** |

<p align="center">
  <img src="./assets/spark_arena_gemma4.png" width="800" alt="spark-arena Gemma 4 12B benchmark result">
  <br><i>spark-arena community benchmark for Gemma 4 12B on single DGX Spark: https://spark-arena.com/benchmark/sub1780704040457</i>
</p>

## Multimodal Agent Smoke Tests

These are informal multimodal checks through the agent interface, not formal benchmark scores. They test whether the Gemma 4 stack can reason over images in practical workflows.

<p align="center">
  <img src="./assets/cogni_chat_multimodal_tests.gif" width="420" alt="Gemma 4 multimodal Telegram smoke tests">
  <br><i>Telegram-based multimodal smoke tests: aerial-scene reasoning and visual puzzle solving.</i>
</p>

### Test Prompts

```text
Object Counting:
count the number of visible vehicles on the top overpass compared to the lower street level

Depth and Layering:
Analyze the structural layers of this highway system. Describe the exact stacking order of the overpasses from the highest point down to the ground level. Are there any sections where a lower road is completely obscured by an upper road?

Spatial Reasoning:
Analyze the traffic flow on the ground level. Based on the orientation of the parked cars and the direction the moving vehicles are facing, determine if the streets on the right side operate as a one-way or two-way system. Furthermore, identify how a vehicle would transition from the ground level onto the primary overpass. If the connection points or ramps are not visible within the frame, explain the visual evidence that led to that conclusion.

Contextual Inference:
Examine the lighting and shadows in this scene. Based on the length, direction, and harshness of the shadows cast by the buildings and the overpass, estimate the general time of day. Additionally, analyze the architectural density, the types of vehicles present, and the road layouts. What do these elements suggest about the function of this district?

Visual Puzzle:
Solve the visual puzzle. Answer briefly with the option letter and one short reason.
```

Observed behavior:

- aerial scene: plausible reasoning about vehicle counts, road layers, traffic flow, shadows, and dense commercial-district characteristics
- visual puzzle: correctly answered `Option B`

## Local Speed and Context Benchmark

<p align="center">
  <img src="./assets/benchmark_test_1-3.png" width="600" alt="Local speed benchmark tests 1 through 3">
</p>

<p align="center">
  <img src="./assets/benchmark_test_4-5.png" width="600" alt="Local speed benchmark tests 4 and 5">
</p>

## Tool-Eval-Bench Capability Benchmark

<p align="center">
  <img src="./assets/benchmark_smarts_1.png" width="600" alt="Tool-eval benchmark summary 1">
</p>

<p align="center">
  <img src="./assets/benchmark_smarts_2.png" width="600" alt="Tool-eval benchmark summary 2">
</p>

<p align="center">
  <img src="./assets/benchmark_smarts_3.png" width="600" alt="Tool-eval benchmark summary 3">
</p>
