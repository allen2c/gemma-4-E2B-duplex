"""Gemma-4-E2B-Duplex inference engine.

Public surface used by ``app.py``:
  - ``GemmaDuplexModel`` — load once at startup (weights resident);
  - ``DuplexSession``    — one per websocket connection, paces the model over a conversation;
  - ``SessionConfig`` + ``from_wire`` / ``to_wire`` — the browser event protocol.

Importing this package pulls in numpy but not torch: the heavy ML imports live inside
``GemmaDuplexModel.__init__``, so nothing loads CUDA until the model is actually constructed.
"""
from .duplex import DuplexSession, InputFrame, OutputFrame
from .gemma_duplex import GemmaDuplexModel
from .protocol import EngineError, SessionConfig, from_wire, to_wire

__all__ = [
    "GemmaDuplexModel",
    "DuplexSession",
    "InputFrame",
    "OutputFrame",
    "SessionConfig",
    "EngineError",
    "from_wire",
    "to_wire",
]
