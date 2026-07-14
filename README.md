# Gemma-4-E2B-Duplex — Demo

A runnable web app for the **[`dockhardman/gemma-4-E2B-duplex`](https://huggingface.co/dockhardman/gemma-4-E2B-duplex)**
speech LoRA: open a page, press **Start**, and have a real-time, full-duplex voice conversation with
the model. Turn-taking, barge-in, and multi-turn are handled **natively by the model** — no external
VAD or scripting layer.

This repo is **inference-only**. It loads the published adapter from Hugging Face onto
`google/gemma-4-E2B-it`, so no weights are shipped here.

## What you get

- **Turn-taking** — the model stays quiet while you talk and opens up when you finish
- **Barge-in** — it stops mid-reply the instant you start speaking
- **Multi-turn** — the conversation holds across many turns
- **Tool calling** — structured tool calls (mock-executed here) rendered as cards, then spoken
- **Text input** — type a message mid-conversation

## How it runs

```text
browser  (mic → downsample to 16k PCM16 → websocket)
   → app.py :8642        FastAPI: serves web/ and bridges the /ws connection
      → GemmaDuplexModel  Gemma-4-E2B + LoRA, frame-synchronous, emits TEXT only
      → Cartesia sonic-3.5 TTS   (cloud) turns that text into 24k PCM16 speech
   → audio streamed back to the browser → gapless playback
```

A **frame** is 80ms = 2 audio tokens + 1 text token. On every frame the model decides *wait* vs.
*speak*; that shared clock is what makes turn-taking and barge-in the model's own behavior. The
adapter is tiny because it only emits text — speech is external TTS.

## Requirements

- An NVIDIA GPU with CUDA (the model runs in bf16)
- Python 3.13
- A **Cartesia** account for the voice ([cartesia.ai](https://cartesia.ai)) — the model produces
  text; Cartesia speaks it
- Hugging Face access to `google/gemma-4-E2B-it` (gated: accept the license and `hf auth login`)

## Quickstart

```bash
# 1. install (uv recommended)
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt

# 2. configure
cp .env.example .env        # then fill in CARTESIA_API_KEY and CARTESIA_VOICE_ID

# 3. run
fastapi run app.py --port 8642
```

Open <http://localhost:8642>, press **Start**, and talk. The first launch downloads the base model
and the adapter from Hugging Face (about a minute to become ready).

## Layout

```text
app.py              FastAPI app: loads the model once, serves web/, bridges /ws
engine/
  protocol.py       browser ↔ server event types (torch-free)
  duplex.py         frame-paced driver + the 3-layer barge-in (torch-free)
  gemma_duplex.py   Gemma-4-E2B + LoRA, frame-synchronous engine + Cartesia TTS — the core
web/
  app.html app.css app.js   mic UI, 16k streaming, gapless playback, tool cards
requirements.txt
.env.example
```

## Tuning

The engine's live-experience guards are on by default and tunable via environment variables
(sensible defaults are baked in):

| Variable | Default | Effect |
|---|---|---|
| `GEMMA_BARGE_RMS` | `0.045` | mic loudness that counts as a barge-in |
| `GEMMA_BARGE_MIN_MS` | `300` | how long that loudness must hold before the reply is cut |
| `GEMMA_FLOOR_K` | `3.0` | adaptive silence threshold = noise floor × K |
| `GEMMA_OPEN_NUDGE` | `3.0` | nudge toward taking a turn after the user goes quiet (0 disables) |
| `GEMMA_AGC_TARGET` | `0.12` | input loudness normalization target (0 disables) |

## Not included

Training data, training/QC scripts, and the synthesis pipeline live elsewhere. This repo shares the
**model and how to run it** — see the [model card](https://huggingface.co/dockhardman/gemma-4-E2B-duplex)
for how it was built.

## License

Apache-2.0.
