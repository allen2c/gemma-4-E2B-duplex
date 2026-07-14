"""The frame-paced duplex driver — turns a full-duplex model into a live session.

A ``DuplexSession`` runs a wall-clock loop: every frame it slices one chunk of mic audio, calls the
model's ``step()``, and demuxes the result into ``protocol`` events for the browser. The model is a
plain duck-typed object (see the ``DuplexModel`` interface) that owns its own weights and KV cache;
this file stays torch-free so it reads as pure control logic.

Barge-in is enforced in three layers, all kept because each covers a case the others miss:
  1. energy gate      — sustained loud mic input while the model speaks cuts the reply;
  2. playback window  — audio is delivered faster than it plays, so the gate stays armed until the
                        browser's buffer is estimated to have drained;
  3. new-turn preempt — if the model starts a fresh turn, any stale buffered audio is flushed first,
                        so overlap is impossible even if the gate missed the interruption.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import AsyncIterator, Protocol

import numpy as np

from .protocol import (
    AudioDelta, AudioInput, Interrupt, Interrupted, SessionConfig,
    TextDelta, TextInput, TurnComplete,
)

logger = logging.getLogger(__name__)

BARGE_RMS = float(os.environ.get("GEMMA_BARGE_RMS", "0.045"))    # mic loudness that counts as barge-in
BARGE_MIN_MS = float(os.environ.get("GEMMA_BARGE_MIN_MS", "300"))  # held this long before we cut the reply
MAX_BACKLOG_S = 2.0                                              # drop older mic audio if the model lags


@dataclass
class InputFrame:
    """One wall-clock tick of mic input: PCM16 mono, silence-padded to a full frame."""
    pcm: bytes = b""


@dataclass
class OutputFrame:
    """One tick of model output. Speech is already-vocoded PCM (Cartesia); text is the spoken words."""
    pcm: bytes = b""                    # PCM16 mono @ output_sample_rate
    text: str = ""                      # interleaved text emitted this tick (may be "")
    is_speaking: bool = False           # the model's native listen-vs-speak signal


class DuplexModel(Protocol):
    """What ``DuplexSession`` needs from a model. Implemented by ``GemmaDuplexModel``."""

    frame_rate_hz: float

    def begin_session(self, *, system_prompt: str, voice: str | None,
                      tools: list[str] | None): ...
    def step(self, state, frame: InputFrame) -> OutputFrame: ...
    def reset_listen(self, state) -> None: ...
    def post_text(self, state, text: str) -> None: ...
    def end_session(self, state) -> None: ...


class DuplexSession:
    """One live conversation: paces the model and bridges it to the browser event stream."""

    def __init__(self, model: DuplexModel, config: SessionConfig) -> None:
        self.model = model
        self.config = config
        self.state = model.begin_session(system_prompt=config.system_prompt, voice=config.voice,
                                         tools=config.extra.get("tools"))
        self._sr = config.input_sample_rate
        self._out_sr = config.output_sample_rate
        self._frame_bytes = max(2, int(self._sr * 2 / model.frame_rate_hz))
        self._max_backlog_bytes = int(MAX_BACKLOG_S * self._sr * 2)
        self._inbuf = bytearray()
        self._speaking = False
        self._loud_ms = 0.0
        self._play_until = 0.0          # monotonic time the browser is estimated to finish playing
        self._suppress = False          # after a barge, drop the model's stale tail until it yields
        self._suppress_frames = 0
        self._suppress_cap = max(1, int(2.0 * model.frame_rate_hz))
        self._out: asyncio.Queue = asyncio.Queue()
        self._closed = False
        self._task = asyncio.create_task(self._run())

    async def send(self, event) -> None:
        if isinstance(event, AudioInput):
            self._inbuf += event.pcm
            self._detect_barge(event.pcm)
        elif isinstance(event, TextInput):
            self.model.post_text(self.state, event.text)
        elif isinstance(event, Interrupt):
            self._barge_in()

    async def events(self) -> AsyncIterator:
        while True:
            yield await self._out.get()

    async def close(self) -> None:
        self._closed = True
        if self._task is not None:
            self._task.cancel()
        self.model.end_session(self.state)

    async def _run(self) -> None:
        # Pace to one frame per tick: subtract the step latency from the sleep so the loop holds
        # realtime instead of drifting (period would otherwise be step + interval).
        interval = 1.0 / self.model.frame_rate_hz
        loop = asyncio.get_event_loop()
        logger.info("duplex session started (interval=%.0fms)", interval * 1000)
        try:
            while not self._closed:
                t0 = loop.time()
                frame = self._take_input_frame()
                out = await loop.run_in_executor(None, self.model.step, self.state, frame)
                self._emit_output(out)
                await asyncio.sleep(max(0.0, interval - (loop.time() - t0)))
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("duplex session loop failed")

    def _take_input_frame(self) -> InputFrame:
        # Drop the oldest backlog if the model has fallen behind, so it steps on near-live audio.
        if self._max_backlog_bytes and len(self._inbuf) > self._max_backlog_bytes + self._frame_bytes:
            drop = len(self._inbuf) - self._max_backlog_bytes
            drop -= drop % self._frame_bytes
            if drop > 0:
                del self._inbuf[:drop]
                logger.warning("duplex: model behind realtime — dropped %.0fms of mic backlog",
                               1000.0 * drop / (self._sr * 2))
        take = min(len(self._inbuf), self._frame_bytes)
        pcm = bytes(self._inbuf[:take])
        del self._inbuf[:take]
        if take < self._frame_bytes:
            pcm += bytes(self._frame_bytes - take)      # pad with silence to a full frame
        return InputFrame(pcm=pcm)

    def _detect_barge(self, pcm: bytes) -> None:
        armed = self._speaking or time.monotonic() < self._play_until
        if not armed or not pcm:
            self._loud_ms = 0.0
            return
        a = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(a * a))) if a.size else 0.0
        if rms > BARGE_RMS:
            self._loud_ms += 1000.0 * a.size / self._sr
            if self._loud_ms >= BARGE_MIN_MS:
                self._barge_in()
        else:
            self._loud_ms = 0.0

    def _emit_output(self, out: OutputFrame) -> None:
        if self._suppress:                              # post-barge: swallow the stale tail
            self._suppress_frames += 1
            if not out.is_speaking or self._suppress_frames >= self._suppress_cap:
                self._suppress = False
            return
        now = time.monotonic()
        if out.is_speaking and not self._speaking:      # rising edge: a new utterance begins
            if now < self._play_until:                  # flush any still-playing tail first (preempt)
                self._out.put_nowait(Interrupted())
                self._play_until = 0.0
            self._speaking = True
        if out.text:
            self._out.put_nowait(TextDelta(text=out.text))
        if out.pcm:
            self._out.put_nowait(AudioDelta(pcm=out.pcm, sample_rate=self._out_sr))
            self._play_until = max(now, self._play_until) + len(out.pcm) / 2.0 / self._out_sr
        if self._speaking and not out.is_speaking:      # falling edge: utterance done
            self._out.put_nowait(TurnComplete())
            self._speaking = False

    def _barge_in(self) -> None:
        in_play_window = time.monotonic() < self._play_until
        self.model.reset_listen(self.state)
        self._loud_ms = 0.0
        self._play_until = 0.0
        if self._speaking or in_play_window:
            self._speaking = False
            self._suppress = True
            self._suppress_frames = 0
            self._out.put_nowait(Interrupted())         # browser flushes playback
