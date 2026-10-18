"""Wire everything together and run the desktop app: window, hotkey, voice, and server."""

from __future__ import annotations

import logging
import os
import socket
import threading
import time
from typing import Any

from sohayok import winapi
from sohayok.assistant import Assistant
from sohayok.config import Config, app_dir
from sohayok.lang import detect_language

log = logging.getLogger(__name__)
WINDOW_TITLE = "Sohayok"


def is_own_window(title: str) -> bool:
    """Sohayok's window, or its page open in a browser ("Sohayok - Google Chrome")."""
    return title == WINDOW_TITLE or title.startswith(f"{WINDOW_TITLE} - ")


class ForegroundTracker:
    """Remembers the last window that was not Sohayok's, so commands act on the user's app."""

    def __init__(self) -> None:
        self.hwnd = 0
        self.title = ""
        threading.Thread(target=self._run, name="foreground", daemon=True).start()

    def _run(self) -> None:
        import win32gui

        while True:
            try:
                hwnd = winapi.foreground_window()
                title = win32gui.GetWindowText(hwnd) if hwnd else ""
                if hwnd and title and not is_own_window(title):
                    self.hwnd, self.title = hwnd, title
            except Exception:
                pass
            time.sleep(0.4)

    def refocus(self) -> None:
        import win32gui

        current = winapi.foreground_window()
        if current and is_own_window(win32gui.GetWindowText(current)) and self.hwnd:
            if win32gui.IsWindow(self.hwnd):
                winapi.force_foreground(self.hwnd)
                time.sleep(0.25)


def port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.3)
        return sock.connect_ex((host, port)) == 0


def build_brain(config: Config) -> tuple[Any, dict[str, str]]:
    """Return the configured language model backend and a description for the UI."""
    brain_config = config.brain
    if brain_config.backend == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            log.warning("ANTHROPIC_API_KEY is not set; falling back to Ollama")
        else:
            from sohayok.brain.claude import ClaudeBrain

            brain = ClaudeBrain(
                brain_config.anthropic_model,
                config.assistant.name,
                config.assistant.home_city,
                max(brain_config.max_steps, 25),
                brain_config.computer_use,
            )
            return brain, {"backend": "Claude", "model": brain_config.anthropic_model}
    from sohayok.brain.ollama import OllamaBrain

    brain = OllamaBrain(
        brain_config.ollama_model,
        brain_config.ollama_url,
        config.assistant.name,
        config.assistant.home_city,
        brain_config.max_steps,
        brain_config.temperature,
        brain_config.ollama_keep_alive,
    )
    threading.Thread(target=_warm, args=(brain,), name="warm-llm", daemon=True).start()
    return brain, {"backend": "Ollama (local)", "model": brain_config.ollama_model}


def _warm(brain: Any) -> None:
    try:
        brain.warm_up()
    except Exception as exc:
        log.warning("could not preload the model: %s", exc)


def load_speech(assistant: Assistant, config: Config, info: dict[str, str]) -> None:
    """Load the recognizer in the background; voice input turns on when it is ready."""
    from sohayok.audio import Recorder
    from sohayok.stt import Transcriber

    speech = config.speech
    try:
        started = time.monotonic()
        recorder = Recorder(
            speech.input_device or None, speech.silence_seconds, speech.max_record_seconds
        )
        last = [0.0]

        def level(value: float) -> None:
            now = time.monotonic()
            if now - last[0] > 0.1:
                last[0] = now
                assistant.emit("level", value=round(value, 4))

        recorder.on_level = level
        transcriber = Transcriber(
            speech.whisper_model,
            speech.bangla_model,
            speech.device,
            speech.compute_type,
            speech.beam_size,
            speech.languages,
        )
        # Compile GPU kernels now, not on the first command.
        import numpy as np

        transcriber.transcribe(np.zeros(16_000, dtype=np.float32), "en")
        assistant.recorder, assistant.transcriber = recorder, transcriber
        info["stt"] = "voice ready"
        assistant.emit("ready", stt=info["stt"], seconds=round(time.monotonic() - started, 1))
    except Exception as exc:
        log.exception("speech recognition failed to load")
        info["stt"] = "voice unavailable"
        assistant.emit("error", message=f"speech recognition unavailable: {exc}")


def start_hotkey(assistant: Assistant, combo: str) -> Any:
    from pynput import keyboard

    def pressed() -> None:
        threading.Thread(target=assistant.listen_once, name="listen", daemon=True).start()

    listener = keyboard.GlobalHotKeys({combo: pressed})
    listener.daemon = True
    listener.start()
    return listener


def start_wake_word(assistant: Assistant, config: Config) -> None:
    from sohayok.wake import WakeWordListener

    listener = WakeWordListener(
        config.activation.wake_model,
        config.activation.wake_threshold,
        config.speech.input_device or None,
    )
    listener.on_wake = lambda: assistant.listen_once() if assistant.state == "idle" else None
    listener.paused = lambda: assistant.state != "idle"
    threading.Thread(target=listener.run, name="wake-word", daemon=True).start()


def run(config: Config, show_window: bool = True, greet: bool = True) -> None:
    import uvicorn

    from sohayok.server import create_app
    from sohayok.tools import load_all
    from sohayok.tools.timers import SCHEDULER
    from sohayok.tts import Speaker

    host, port = config.server.host, config.server.port
    url = f"http://{host}:{port}/"
    if port_in_use(host, port):
        log.info("Sohayok is already running; opening its window")
        if show_window:
            _window(url)
        return

    winapi.make_dpi_aware()
    registry = load_all()
    tracker = ForegroundTracker()
    brain, info = build_brain(config)
    info["stt"] = "loading voice…"
    speaker = Speaker(
        config.voice.bangla_voice,
        config.voice.english_voice,
        config.voice.rate,
        config.voice.enabled,
    )
    assistant = Assistant(config, registry, brain, speaker, window_title=lambda: tracker.title)
    assistant.executor.focus_target = tracker.refocus
    from sohayok.audio import chime

    assistant.chime = chime

    SCHEDULER.load(app_dir() / "reminders.json")
    SCHEDULER.on_fire = lambda r: assistant.reply(r.text, detect_language(r.text, "bn"))
    SCHEDULER.start()

    app = create_app(assistant, info)
    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="warning"))
    threading.Thread(target=server.run, name="server", daemon=True).start()
    threading.Thread(
        target=load_speech, args=(assistant, config, info), name="load-speech", daemon=True
    ).start()
    start_hotkey(assistant, config.activation.hotkey)
    if config.activation.wake_word:
        try:
            start_wake_word(assistant, config)
            info["wake"] = 'say "hey jarvis"'
        except Exception as exc:  # e.g. the optional openwakeword package is missing
            log.warning("wake word unavailable (%s); use the hotkey", exc)
    while not port_in_use(host, port):
        time.sleep(0.1)
    assistant.start(greet=greet)
    log.info("Sohayok is running at %s (hotkey %s)", url, config.activation.hotkey)
    if show_window:
        _window(url)
        server.should_exit = True
    else:
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            server.should_exit = True


def _window(url: str) -> None:
    import webview

    webview.create_window(WINDOW_TITLE, url, width=440, height=720, min_size=(360, 480))
    webview.start()
