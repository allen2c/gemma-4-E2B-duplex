"""Wire protocol for the duplex demo — the small set of events the browser and server exchange.

Everything is a plain dataclass with a ``type`` tag. Audio travels as PCM16 mono; the ``pcm``
bytes are base64-encoded on the JSON wire (see ``to_wire`` / ``from_wire``).

Audio convention: input 16 kHz, output 24 kHz.
"""
from __future__ import annotations

import base64
from dataclasses import asdict, dataclass, field
from typing import Any

INPUT_SAMPLE_RATE = 16_000
OUTPUT_SAMPLE_RATE = 24_000


@dataclass
class SessionConfig:
    """Per-connection settings passed to the engine when a session opens."""
    system_prompt: str = ""
    voice: str | None = None                    # Cartesia voice id (falls back to CARTESIA_VOICE_ID)
    input_sample_rate: int = INPUT_SAMPLE_RATE
    output_sample_rate: int = OUTPUT_SAMPLE_RATE
    extra: dict[str, Any] = field(default_factory=dict)   # e.g. {"tools": [...]}


# ---- browser -> server ----
@dataclass
class AudioInput:
    """One chunk of mic audio (PCM16 mono @ input_sample_rate)."""
    pcm: bytes = b""
    type: str = "audio_input"


@dataclass
class TextInput:
    """A typed message from the user."""
    text: str = ""
    type: str = "text_input"


@dataclass
class Interrupt:
    """User pressed stop / barged in — drop the in-flight reply."""
    type: str = "interrupt"


# ---- server -> browser ----
@dataclass
class AudioDelta:
    """A chunk of synthesized speech (PCM16 mono @ output_sample_rate)."""
    pcm: bytes = b""
    sample_rate: int = OUTPUT_SAMPLE_RATE
    type: str = "audio_delta"


@dataclass
class TextDelta:
    """A chunk of the assistant's text as it speaks."""
    text: str = ""
    type: str = "text_delta"


@dataclass
class Interrupted:
    """The engine acknowledges the reply was cut (browser flushes playback)."""
    type: str = "interrupted"


@dataclass
class TurnComplete:
    """The assistant finished a turn."""
    type: str = "turn_complete"


@dataclass
class EngineError:
    message: str = ""
    type: str = "error"


ClientEvent = AudioInput | TextInput | Interrupt
ServerEvent = AudioDelta | TextDelta | Interrupted | TurnComplete | EngineError

_TYPES: dict[str, type] = {
    c.__dataclass_fields__["type"].default: c
    for c in (AudioInput, TextInput, Interrupt,
              AudioDelta, TextDelta, Interrupted, TurnComplete, EngineError)
}


def to_wire(event: ClientEvent | ServerEvent) -> dict[str, Any]:
    d = asdict(event)
    if isinstance(d.get("pcm"), (bytes, bytearray)):
        d["pcm"] = base64.b64encode(d["pcm"]).decode("ascii")
    return d


def from_wire(d: dict[str, Any]) -> ClientEvent | ServerEvent:
    cls = _TYPES[d["type"]]
    kwargs = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
    if isinstance(kwargs.get("pcm"), str):
        kwargs["pcm"] = base64.b64decode(kwargs["pcm"])
    return cls(**kwargs)
