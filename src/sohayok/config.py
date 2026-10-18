"""Settings: defaults here, overridden by %APPDATA%\\Sohayok\\config.toml."""

from __future__ import annotations

import json
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
    # "lite": Laya-Bangla decides the action and a small model fills in details (fast, light);
    # "ollama": one local model with all tools; "anthropic": Claude (needs ANTHROPIC_API_KEY).
    backend: str = "lite"
    # Laya checkpoint for decisions: "auto" = Sohayok's fine-tune if prepared, else the
    # published algolyzer/laya-bangla.
    decider_model: str = "auto"
    # Laya takes ~160 ms on the CPU, which keeps 1.3 GB of GPU memory free for other work.
    decider_device: str = "cpu"
    # Answer model: "ollama" (this PC, free) or an API: "openai", "gemini", "glm", "openrouter",
    # "anthropic", "custom" (any OpenAI-compatible URL). Keys are saved in Windows Credential
    # Manager from Settings, or read from OPENAI_API_KEY, GEMINI_API_KEY, ... .
    provider: str = "ollama"
    lite_model: str = "qwen3.5:2b"
    base_url: str = ""  # only for provider "custom"
    # Below this decision confidence, or for multi-step requests, the agent model takes over.
    decider_min_confidence: float = 0.7  # below it Laya is right only ~41% of the time
    ollama_url: str = "http://127.0.0.1:11434"
    # Agent model for multi-step or unusual requests (and the whole job with backend "ollama").
    ollama_model: str = "qwen3.5:2b"
    # Unload the language model from the GPU this long after the last command.
    ollama_keep_alive: str = "2m"
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
    wake_threshold: float = 0.3  # measured operating point for "tammy" with the Whisper check
    # Scores at or above this wake Sohayok instantly; scores between wake_threshold and this are
    # double-checked by Whisper first (fewer false wake-ups, a little slower).
    wake_instant_threshold: float = 0.8
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
    settings = path.with_name("settings.json")  # written by the Settings panel
    if settings.exists():
        try:
            _merge(config, json.loads(settings.read_text("utf-8")))
        except (ValueError, OSError):
            pass
    return config


SETTABLE = {
    "brain": {"provider", "lite_model", "base_url"},
    "voice": {"enabled"},
    "assistant": {"confirm_risky_actions"},
    "activation": {"wake_word"},
}


def save_settings(updates: dict[str, dict[str, Any]], folder: Path | None = None) -> dict:
    """Merge Settings-panel changes into settings.json (only the keys above are allowed)."""
    path = (folder or app_dir()) / "settings.json"
    current_settings = json.loads(path.read_text("utf-8")) if path.exists() else {}
    for section, values in updates.items():
        allowed = SETTABLE.get(section, set())
        bad = set(values) - allowed
        if bad:
            raise ValueError(f"cannot change {section}.{', '.join(sorted(bad))} here")
        current_settings.setdefault(section, {}).update(values)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(current_settings, indent=2, ensure_ascii=False), "utf-8")
    return current_settings


_current: Config | None = None


def current() -> Config:
    global _current
    if _current is None:
        _current = load()
    return _current


def use(config: Config) -> None:
    global _current
    _current = config
