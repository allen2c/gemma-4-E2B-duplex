"""Gemma-4-E2B-Duplex demo — a single-process FastAPI app.

Serves the static mic UI and one ``/ws`` endpoint that bridges the browser to a live
``DuplexSession``. The model is loaded once at startup (weights resident) and shared across
connections; each websocket gets its own session (its own KV cache and Cartesia connection).

Run:
    fastapi run app.py --port 8642
Then open http://localhost:8642
"""
from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from engine import DuplexSession, EngineError, GemmaDuplexModel, SessionConfig, from_wire, to_wire

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("app")

WEB_DIR = Path(__file__).parent / "web"
PORT = int(os.environ.get("PORT", "8642"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not (os.environ.get("CARTESIA_API_KEY") and os.environ.get("CARTESIA_VOICE_ID")):
        logger.warning("CARTESIA_API_KEY / CARTESIA_VOICE_ID not set — the model will emit text but no speech")
    logger.info("loading model (this takes ~a minute the first time) ...")
    app.state.model = await asyncio.to_thread(GemmaDuplexModel)
    logger.info("model ready — open http://localhost:%d", PORT)
    yield


app = FastAPI(title="gemma-4-E2B-duplex", lifespan=lifespan)
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    page = WEB_DIR / "app.html"
    return page.read_text(encoding="utf-8") if page.exists() else "<h1>gemma-4-E2B-duplex</h1>"


@app.websocket("/ws")
async def ws(websocket: WebSocket) -> None:
    await websocket.accept()
    cfg = SessionConfig()
    tools_param = websocket.query_params.get("tools")   # comma-separated tool names; absent = all tools
    if tools_param is not None:
        cfg.extra["tools"] = [t for t in tools_param.split(",") if t]

    model = websocket.app.state.model
    try:
        # begin_session is blocking (GPU prefill + Cartesia connect/warm-up); run it off the event loop
        # so this connection — and any others — stay responsive while it prepares.
        state = await asyncio.to_thread(
            model.begin_session, system_prompt=cfg.system_prompt, voice=cfg.voice,
            tools=cfg.extra.get("tools"))
        session = DuplexSession(model, cfg, state)
    except Exception as e:
        logger.exception("failed to open session")
        await websocket.send_json(to_wire(EngineError(message=str(e))))
        await websocket.close()
        return

    await websocket.send_json({"type": "ready"})   # the browser waits for this before sending mic audio

    async def inbound() -> None:
        while True:
            await session.send(from_wire(await websocket.receive_json()))

    async def outbound() -> None:
        async for ev in session.events():
            await websocket.send_json(to_wire(ev))

    tasks = [asyncio.create_task(inbound(), name="inbound"),
             asyncio.create_task(outbound(), name="outbound")]
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:   # surface real errors; a clean disconnect just ends the task
            exc = task.exception()
            if exc is not None and not isinstance(exc, WebSocketDisconnect):
                logger.exception("ws task %s failed", task.get_name(), exc_info=exc)
    except WebSocketDisconnect:
        pass
    finally:
        for task in tasks:
            task.cancel()
        await session.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
