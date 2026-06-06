# Telegram Voice Notes with NeMoHermes / OpenShell

This repo runs Cogni-Brain Omni inside a NeMoHermes / OpenShell sandbox. Telegram voice notes do not go directly to the LLM by default.

The runtime path is:

```text
Telegram voice note
→ Hermes Gateway
→ local speech-to-text
→ Cogni-Brain Omni / Gemma 4
→ Telegram text reply
```

Gemma 4 can reason over audio, but the Telegram gateway expects a speech-to-text layer first. In this setup, local STT is handled with `faster-whisper`.

## Final Working Setup

```text
STT provider: local
STT engine: faster-whisper
Whisper model: Systran/faster-whisper-base
Model location inside sandbox: /sandbox/models/faster-whisper-base-flat
Hermes config: /sandbox/.hermes/config.yaml
Gateway boot script: ~/boot.sh
```

## 1. Install `faster-whisper` inside the sandbox

Enter the sandbox:

```bash
nemohermes cogni connect
```

Install `faster-whisper` using the policy-allowed system pip, into the sandbox user site:

```bash
/usr/bin/pip3 install --break-system-packages --user faster-whisper
```

Verify:

```bash
/usr/bin/python3 -c "import faster_whisper; print('faster-whisper OK')"
```

Expected output:

```text
faster-whisper OK
```

## 2. Ensure Hermes Gateway can see sandbox user packages

Hermes runs from its own environment, so the gateway must receive `PYTHONPATH` before startup.

Use this `~/boot.sh`:

```bash
#!/bin/bash
# 1. Fix paths
export PATH="/sandbox/.local/bin:$PATH"
export PYTHONPATH="/sandbox/.local/lib/python3.11/site-packages:${PYTHONPATH:-}"

# 2. Clear ghost locks
rm -f ~/.hermes/gateway.pid
rm -rf ~/.local/state/hermes/gateway-locks/

# 3. Start the gateway quietly in the background
nohup hermes gateway run --replace > ~/gateway.log 2>&1 &
echo "⚕️ Hermes Gateway, DuckDuckGo PATH, and local STT path initialized."
```

Make it executable if needed:

```bash
chmod +x ~/boot.sh
```

## 3. Enable local STT in Hermes config

Edit `/sandbox/.hermes/config.yaml` and add or update:

```yaml
stt:
  enabled: true
  provider: "local"
  local:
    model: "/sandbox/models/faster-whisper-base-flat"
```

Important:

```yaml
model: "base"
```

will make `faster-whisper` try to download the model from Hugging Face at runtime. In the OpenShell sandbox, that may fail due to network policy/proxy restrictions.

Using a local model path avoids runtime downloads.

## 4. Download the Whisper model on the host

On the host, download the model:

```bash
uvx --from huggingface-hub hf download Systran/faster-whisper-base
```

If the Hugging Face cache has permission issues:

```bash
mkdir -p ~/.cache/huggingface/hub/.locks
sudo chown -R "$USER:$USER" ~/.cache/huggingface
```

Retry:

```bash
uvx --from huggingface-hub hf download Systran/faster-whisper-base
```

## 5. Create a flat model directory on the host

Hugging Face snapshots use symlinks. Create a flat copy before uploading:

```bash
rm -rf /tmp/faster-whisper-base-flat
mkdir -p /tmp/faster-whisper-base-flat

cp -L ~/.cache/huggingface/hub/models--Systran--faster-whisper-base/snapshots/*/* \
  /tmp/faster-whisper-base-flat/

ls -lh /tmp/faster-whisper-base-flat
```

Expected files:

```text
config.json
model.bin
README.md
tokenizer.json
vocabulary.txt
```

`model.bin` should be roughly `139M`.

Create a tarball:

```bash
cd /tmp
tar -czf faster-whisper-base-flat.tgz faster-whisper-base-flat
```

## 6. Upload the model into the sandbox

Use OpenShell’s file upload command:

```bash
openshell sandbox upload cogni /tmp/faster-whisper-base-flat.tgz /sandbox/
```

Reconnect to the sandbox:

```bash
nemohermes cogni connect
```

Extract inside the sandbox:

```bash
mkdir -p /sandbox/models
tar -xzf /sandbox/faster-whisper-base-flat.tgz -C /sandbox/models
```

Verify:

```bash
ls -lh /sandbox/models/faster-whisper-base-flat
```

Expected:

```text
config.json
model.bin
README.md
tokenizer.json
vocabulary.txt
```

## 7. Restart Hermes Gateway

Inside the sandbox:

```bash
./boot.sh
```

Send a short Telegram voice note.

Check logs if needed:

```bash
tail -n 120 ~/gateway.log
```

A working path should no longer show:

```text
no speech-to-text provider is configured
```

or:

```text
Local transcription failed: 403 Forbidden
```

## Failure Signatures

### STT not configured

Telegram response:

```text
I received your voice message but can't transcribe it — no speech-to-text provider is configured.
```

Fix:

```yaml
stt:
  enabled: true
  provider: "local"
```

### `faster-whisper` missing

Error:

```text
ModuleNotFoundError: No module named 'faster_whisper'
```

Fix:

```bash
/usr/bin/pip3 install --break-system-packages --user faster-whisper
```

### Gateway cannot see `faster-whisper`

Cause: `PYTHONPATH` was exported after `hermes gateway run`.

Fix: export `PYTHONPATH` before the `nohup hermes gateway run` line in `boot.sh`.

### Hugging Face 403 during model download

Error:

```text
httpx.ProxyError: 403 Forbidden
```

Cause: the sandbox blocks or restricts model downloads from Hugging Face for the process doing STT.

Fix: download the model on the host, upload it into `/sandbox/models`, and point `stt.local.model` to the local path.

## Working Result

After this setup, Telegram voice notes work locally:

```text
Telegram voice note
→ local faster-whisper transcription
→ Cogni-Brain Omni
→ Telegram reply
```

This preserves the local-first agent path while avoiding cloud STT APIs.
