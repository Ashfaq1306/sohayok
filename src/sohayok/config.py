"""Settings: defaults here, overridden by %APPDATA%\\Sohayok\\config.toml."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any


def app_dir() -> Path:
    base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    return base / "Sohayok"


@dataclass
class AssistantConfig:
    name: str = "Sohayok"
    # Language of the first greeting and of replies when the language is unclear.
    default_language: str = "bn"
    home_city: str = "Dhaka"
    # Ask before CONFIRM-level actions (delete, shut down, run commands, publish).
    confirm_risky_actions: bool = True
    confirm_timeout_seconds: float = 30.0
    history_turns: int = 6


@dataclass
class BrainConfig:
    # "ollama" (local, private, free) or "anthropic" (Claude API; needs ANTHROPIC_API_KEY).
    backend: str = "ollama"
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen3.5:4b"
    ollama_keep_alive: str = "30m"
    anthropic_model: str = "claude-sonnet-5-5"
    # Let Claude see the screen and use the mouse (Anthropic backend only).
    computer_use: bool = True
    max_steps: int = 12
    temperature: float = 0.2


@dataclass
class SpeechConfig:
    # Multilingual model: detects the language and transcribes English.
    whisper_model: str = "deepdml/faster-whisper-large-v3-turbo-ct2"
    # Bangla-specialized model used when Bangla is detected: "auto" = the BuzzASR model
    # prepared by `sohayok setup` (if present), a path, or "" for the multilingual one only.
    bangla_model: str = "auto"
    device: str = "cuda"
    compute_type: str = "int8_float16"
    beam_size: int = 1
    # Languages the detector may choose between.
    languages: list[str] = field(default_factory=lambda: ["bn", "en"])
    input_device: str = ""  # sounddevice name or index; "" = Windows default microphone
    silence_seconds: float = 1.0
    # Keep the last N voice commands (audio + transcript) in %APPDATA%\Sohayok\recordings,
    # on this PC only, to check recognition later. 0 turns it off.
    keep_recordings: int = 20
    max_record_seconds: float = 20.0


@dataclass
class VoiceConfig:
    enabled: bool = True
    bangla_voice: str = "bn-BD-NabanitaNeural"
    english_voice: str = "en-US-AvaMultilingualNeural"
    rate: str = "+5%"


@dataclass
class ActivationConfig:
    hotkey: str = "<ctrl>+<alt>+<space>"
    # Listen continuously (on this PC only) for the wake word as well as the hotkey.
    # Needs the "wake" extra; without it only the hotkey works.
    wake_word: bool = True
    # "tammy" (Sohayok's own model), "hey_jarvis", or a path to an openWakeWord .onnx model.
    wake_model: str = "tammy"
    wake_threshold: float = 0.5
    # Double-check each wake with Whisper before waking (fewer false wake-ups).
    wake_confirm: bool = True


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8765


@dataclass
class FilesConfig:
    extra_search_roots: list[str] = field(default_factory=list)


@dataclass
class ClaudeCodeConfig:
    timeout_seconds: int = 900
    allowed_tools: list[str] = field(
        default_factory=lambda: [
            "Read",
            "Glob",
            "Grep",
            "WebSearch",
            "WebFetch",
            "Bash(gh:*)",
            "Bash(git status:*)",
            "Bash(git log:*)",
            "Bash(git diff:*)",
        ]
    )


@dataclass
class Config:
    assistant: AssistantConfig = field(default_factory=AssistantConfig)
    brain: BrainConfig = field(default_factory=BrainConfig)
    speech: SpeechConfig = field(default_factory=SpeechConfig)
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    activation: ActivationConfig = field(default_factory=ActivationConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    files: FilesConfig = field(default_factory=FilesConfig)
    claude_code: ClaudeCodeConfig = field(default_factory=ClaudeCodeConfig)


def _merge(target: Any, values: dict[str, Any], where: str = "") -> None:
    known = {f.name: f for f in fields(target)}
    for key, value in values.items():
        if key not in known:
            raise ValueError(f"unknown setting {where}{key}")
        current_value = getattr(target, key)
        if is_dataclass(current_value):
            if not isinstance(value, dict):
                raise ValueError(f"[{where}{key}] must be a table")
            _merge(current_value, value, f"{where}{key}.")
        else:
            setattr(target, key, value)


def load(path: Path | None = None) -> Config:
    config = Config()
    path = path or app_dir() / "config.toml"
    if path.exists():
        with path.open("rb") as handle:
            _merge(config, tomllib.load(handle))
    return config


_current: Config | None = None


def current() -> Config:
    global _current
    if _current is None:
        _current = load()
    return _current


def use(config: Config) -> None:
    global _current
    _current = config
