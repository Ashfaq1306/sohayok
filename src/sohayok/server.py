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


class SettingsIn(BaseModel):
    provider: str | None = None
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None  # stored in Windows Credential Manager; never sent back
    speak_replies: bool | None = None
    confirm_risky_actions: bool | None = None
    wake_word: bool | None = None


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

    @app.get("/api/settings")
    def get_settings() -> dict[str, Any]:
        from sohayok.brain.providers import PROVIDERS, get_key

        config = assistant.config
        return {
            "provider": config.brain.provider,
            "model": config.brain.lite_model,
            "base_url": config.brain.base_url,
            "providers": {
                name: {
                    "label": p.label,
                    "default_model": p.default_model,
                    "needs_key": p.kind != "ollama",
                    "key_saved": p.kind != "ollama" and bool(get_key(name)),
                }
                for name, p in PROVIDERS.items()
            },
            "speak_replies": config.voice.enabled,
            "confirm_risky_actions": config.assistant.confirm_risky_actions,
            "wake_word": config.activation.wake_word,
            "decider": "Laya-Bangla (ashfaq1306/sohayok-laya-actions)",
        }

    @app.get("/api/models")
    def models(provider: str, base_url: str = "") -> dict[str, Any]:
        from sohayok.brain.providers import list_models

        try:
            return {"models": list_models(provider, base_url, assistant.config.brain.ollama_url)}
        except Exception as exc:
            return {"models": [], "error": str(exc)[:300]}

    @app.post("/api/settings")
    def set_settings(body: SettingsIn) -> dict[str, Any]:
        from sohayok.brain.providers import PROVIDERS, set_key
        from sohayok.config import save_settings

        config = assistant.config
        brain: dict[str, Any] = {}
        if body.provider is not None:
            if body.provider not in PROVIDERS:
                return {"ok": False, "error": f"unknown provider {body.provider}"}
            brain["provider"] = config.brain.provider = body.provider
        if body.model is not None:
            brain["lite_model"] = config.brain.lite_model = body.model.strip()
        if body.base_url is not None:
            brain["base_url"] = config.brain.base_url = body.base_url.strip()
        if body.api_key:
            set_key(body.provider or config.brain.provider, body.api_key.strip())
        updates: dict[str, dict[str, Any]] = {"brain": brain} if brain else {}
        if body.speak_replies is not None:
            updates["voice"] = {"enabled": body.speak_replies}
            config.voice.enabled = body.speak_replies
            if assistant.speaker is not None:
                assistant.speaker.enabled = body.speak_replies
        if body.confirm_risky_actions is not None:
            updates["assistant"] = {"confirm_risky_actions": body.confirm_risky_actions}
            config.assistant.confirm_risky_actions = body.confirm_risky_actions
            assistant.executor.confirm = body.confirm_risky_actions
        if body.wake_word is not None:
            updates["activation"] = {"wake_word": body.wake_word}
            config.activation.wake_word = body.wake_word
        save_settings(updates)
        result: dict[str, Any] = {"ok": True}
        if (brain or body.api_key) and "reload_brain" in controls:
            try:
                result.update(controls["reload_brain"]())
            except Exception as exc:
                result.update(ok=False, error=str(exc)[:300])
        if body.wake_word is not None:
            result["restart_needed"] = True
        return result

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
