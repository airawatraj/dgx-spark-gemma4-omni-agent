# DGX Spark · Gemma 4 Omni Agent · Field Notes

Configuration decisions and debugging notes specific to this stack. Setup and usage are in `README.md`.

---

## War Stories

### FP8 weight quantization can kill GPU workers on SM121

`--quantization fp8` caused vLLM to select `CutlassFP8ScaledMMLinearKernel` for weight computation. On GB10 SM121, that path killed GPU worker subprocesses without a clean vLLM error. EngineCore stayed alive and waited forever for dead workers.

Failure signature:

- Logs stop permanently at `Using AttentionBackendEnum.TRITON_ATTN backend.`
- Zombie Python processes appear in `top` (`Z` state)
- GPU utilization stays at 0% indefinitely
- System RAM stays flat around early-init levels
- Model weights never start loading

This looks like slow initialization. Check before waiting forever:

```bash
top -b -n 1 | grep -E "python|vllm"
# Z state = worker crash. No Z state = likely still loading.

nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
# 0 MB / 0% with flat RAM = workers are probably dead, not loading.

du -sh ~/.cache/triton/
# 4.0K = Triton has compiled nothing = not a compilation wait.
```

Fix: do not use FP8 weight quantization.

Use BF16 weights with FP8 KV cache instead:

```bash
--dtype bfloat16
--kv-cache-dtype fp8
```

FP8 KV cache works on this stack. FP8 weight quantization via the Cutlass path did not.

---

### Speculative config field name drift

The `SpeculativeConfig` dataclass in the `gemma4-unified` image uses `model`, not `speculative_model`.

Wrong:

```json
{"method":"mtp","speculative_model":"google/gemma-4-12B-it-assistant","num_speculative_tokens":5}
```

Right:

```json
{"method":"mtp","model":"google/gemma-4-12B-it-assistant","num_speculative_tokens":5}
```

Passing the wrong key exits on startup with a Pydantic `ValidationError`.

Nightly field names can drift. Introspect before debugging ghosts:

```bash
docker run --rm --entrypoint python3 vllm/vllm-openai:gemma4-unified \
  -c "from vllm.config import SpeculativeConfig; \
      print(list(SpeculativeConfig.__dataclass_fields__.keys()))"
```

---

### Logs freeze at TRITON_ATTN does not always mean stuck

After fixing FP8, initialization still appears to hang at the same point:

```text
(EngineCore) INFO Using AttentionBackendEnum.TRITON_ATTN backend.
(EngineCore) INFO Using AttentionBackendEnum.TRITON_ATTN backend.
```

The double line is expected. Gemma 4 has heterogeneous attention: local layers use `head_dim=256`, global layers use `head_dim=512`, and vLLM forces the Triton attention backend.

After that, weight loading can be quiet for several minutes.

How to tell the difference:

```bash
free -h | grep Mem
# RAM should climb during BF16 weight load.

nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
# GPU memory should climb toward the model footprint.
```

If memory is climbing and there are no zombie workers, wait.

Mount Triton cache so kernel work survives restarts:

```bash
-v "$HOME/.cache/triton:/root/.cache/triton"
-e TRITON_CACHE_DIR=/root/.cache/triton
```

---

### MTP was the unlock

Plain Gemma 4 12B was too slow to justify as a daily model on Spark. The real unlock was MTP speculative decoding with the assistant model:

```text
google/gemma-4-12B-it-assistant
num_speculative_tokens=5
method=mtp
```

Pre-download the assistant model before the first container start so init does not stall on a live fetch.

Useful log signs that MTP loaded correctly:

```text
Resolved architecture: Gemma4MTPModel
Detected MTP model
Sharing target model embedding weights with the draft model
Gemma4 MTP: draft layer ...
```

Without MTP, Gemma 4 felt like a curiosity. With MTP, it became usable.

---

### 196K beat 262K for the full omni stack

262K context can boot in text-heavy configurations, but it becomes unreliable when the full daily stack is enabled:

- image input
- audio input
- video-as-frames
- tool parser
- reasoning parser
- MTP 5
- `gpu_memory_utilization=0.75`

The daily compromise:

```text
131K = safe baseline, fastest boot
196K = daily sweet spot for the full omni stack
262K = text-heavy or experimental profile
```

---

### 0.75 memory utilization is the stability line

With BF16 weights, FP8 KV cache, the MTP assistant, and multimodal limits enabled, the final working profile reported:

```text
Model loading took:        ~23.62 GiB
Available KV cache:        ~63.33 GiB
GPU KV cache size:         ~4.28M tokens
196K concurrency estimate: 21.77x
```

This was at:

```bash
--gpu-memory-utilization 0.75
```

Pushing higher may work for benchmarks, but with swap disabled on DGX Spark unified memory, failures can destabilize the whole machine instead of cleanly failing the container.

---

### Tool parsing stays on for agentic use

Removing the tool and reasoning parsers can improve raw TPS slightly.

For this repo, they stay on because the target use case is an agentic daily driver:

```bash
--enable-auto-tool-choice
--tool-call-parser gemma4
--reasoning-parser gemma4
```

For isolated TPS benchmarking, strip them from the launch command.

---

### Multimodal warmup failure is non-fatal

The server may log:

```text
Multi-modal warmup failed
Readonly multi-modal warmup failed
```

This did not prevent the server from becoming healthy. If startup reaches `Application startup complete`, test actual image/audio/video requests before calling the deployment broken.

---

### The draft model multimodal warning is expected

With MTP active:

```text
Draft model does not support multimodal inputs, falling back to text-only mode
```

Expected.

The assistant model handles text token prediction. The main Gemma 4 model handles multimodal input. Speculative acceleration still applies to text generation.

---

### Enforce eager is a debug tool

`--enforce-eager` helped isolate boot issues during debugging.

Remove it for daily use. It disables `torch.compile` and CUDA graph capture, both of which contribute to sustained inference speed.
