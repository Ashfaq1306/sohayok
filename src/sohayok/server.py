"""Local web API and live event stream for the assistant window."""

from __future__ import annotations

import asyncio
import collections
import threading
from importlib import resources
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from sohayok.assistant import Assistant


class CommandIn(BaseModel):
    text: str


class ConfirmIn(BaseModel):
    answer: bool


class EventHub:
    """Fans assistant events (from any thread) out to every connected window."""

    def __init__(self, keep: int = 200) -> None:
        self.recent: collections.deque[dict[str, Any]] = collections.deque(maxlen=keep)
        self.clients: set[asyncio.Queue[dict[str, Any]]] = set()
        self.loop: asyncio.AbstractEventLoop | None = None
        self._lock = threading.Lock()

    def publish(self, event: dict[str, Any]) -> None:
        if event.get("type") == "level":
            pass  # microphone levels are live-only
        else:
            self.recent.append(event)
        loop = self.loop
        if loop is None:
            return
        with self._lock:
            clients = list(self.clients)
        for client in clients:
            loop.call_soon_threadsafe(_offer, client, event)


def _offer(client: asyncio.Queue, event: dict[str, Any]) -> None:
    if client.full():
        client.get_nowait()
    client.put_nowait(event)


def create_app(
    assistant: Assistant,
    info: dict[str, Any] | None = None,
    controls: dict[str, Any] | None = None,
) -> FastAPI:
    app = FastAPI(title="Sohayok", docs_url=None, redoc_url=None)
    hub = EventHub()
    assistant.listeners.append(hub.publish)
    info = info or {}

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return resources.files("sohayok").joinpath("ui/index.html").read_text(encoding="utf-8")

    @app.get("/api/status")
    def status() -> dict[str, Any]:
        pending = assistant.broker.pending
        return {
            "state": assistant.state,
            "language": assistant.language,
            "pending_confirmation": pending.question if pending else None,
            "tools": assistant.registry.names(),
            "voice_input": bool(assistant.transcriber and assistant.recorder),
            **info,
        }

    @app.post("/api/command")
    def command(body: CommandIn) -> dict[str, str]:
        assistant.submit(body.text, source="text")
        return {"status": "queued"}

    @app.post("/api/listen")
    def listen() -> dict[str, str]:
        threading.Thread(target=assistant.listen_once, name="listen", daemon=True).start()
        return {"status": "listening"}

    controls = controls or {}

    @app.post("/api/show")
    def show() -> dict[str, str]:
        """A second launch of Sohayok calls this instead of starting another copy."""
        if "show" in controls:
            controls["show"]()
        return {"status": "shown" if "show" in controls else "no window"}

    @app.post("/api/stop")
    def stop() -> dict[str, str]:
        assistant.interrupt()
        return {"status": "stopped"}

    @app.post("/api/confirm")
    def confirm(body: ConfirmIn) -> dict[str, bool]:
        return {"resolved": assistant.broker.resolve(body.answer)}

    @app.websocket("/ws")
    async def events(socket: WebSocket) -> None:
        await socket.accept()
        hub.loop = asyncio.get_running_loop()
        client: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=500)
        with hub._lock:
            hub.clients.add(client)
        try:
            await socket.send_json(
                {"type": "hello", "state": assistant.state, "history": list(hub.recent), **info}
            )
            while True:
                await socket.send_json(await client.get())
        except WebSocketDisconnect:
            pass
        finally:
            with hub._lock:
                hub.clients.discard(client)

    app.state.hub = hub
    return app
