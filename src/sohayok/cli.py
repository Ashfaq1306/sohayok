"""Command line: `sohayok` starts the assistant; subcommands check, test, and benchmark it."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path

from sohayok import config as config_module

DEFAULT_CONFIG = """\
# Sohayok settings. Delete a line to go back to its default.
[assistant]
default_language = "bn"     # "bn" or "en": greeting and fallback reply language
home_city = "Dhaka"         # for weather
confirm_risky_actions = true

[brain]
backend = "ollama"          # "ollama" (local) or "anthropic" (Claude; set ANTHROPIC_API_KEY)
ollama_model = "{ollama_model}"
anthropic_model = "claude-sonnet-5-5"

[speech]
whisper_model = "{whisper_model}"
bangla_model = "{bangla_model}"

[activation]
hotkey = "<ctrl>+<alt>+<space>"
wake_word = true            # also listen for "hey jarvis" (false: hotkey only)
"""


def _setup_logging(verbose: bool) -> None:
    app_dir = config_module.app_dir()
    app_dir.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [
        logging.FileHandler(app_dir / "sohayok.log", encoding="utf-8")
    ]
    if sys.stderr is not None and sys.stderr.isatty():
        handlers.append(logging.StreamHandler())
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        handlers=handlers,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    for noisy in ("httpx", "httpcore", "faster_whisper", "comtypes", "PIL", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def cmd_run(args: argparse.Namespace) -> None:
    from sohayok.desktop import run

    run(config_module.current(), show_window=not args.no_window, greet=not args.quiet)


def cmd_do(args: argparse.Namespace) -> None:
    """Run one typed command through the real pipeline (it really acts) and print events."""
    from sohayok.assistant import Assistant, Command
    from sohayok.desktop import build_brain
    from sohayok.tools import load_all

    config = config_module.current()
    brain, _ = build_brain(config)
    speaker = None
    if args.speak:
        from sohayok.tts import Speaker

        speaker = Speaker(config.voice.bangla_voice, config.voice.english_voice, config.voice.rate)
    assistant = Assistant(config, load_all(), brain, speaker)
    assistant.executor.confirm = not args.yes
    if args.yes:
        assistant.executor.ask_user = lambda question: True

    def show(event: dict) -> None:
        if event["type"] in {"heard", "tool", "reply", "route", "error", "confirm"}:
            shown = {k: v for k, v in event.items() if k not in {"type", "time"}}
            print(f"[{event['type']}] {json.dumps(shown, ensure_ascii=False)}", flush=True)

    assistant.listeners.append(show)
    if not args.yes:
        assistant.broker.on_ask = lambda p: print(
            f"[confirm] {p.question} — answering no (pass --yes to approve)", flush=True
        )
        assistant.broker.timeout = 0.1
    assistant.handle(Command(args.text, args.language))


def cmd_devices(_: argparse.Namespace) -> None:
    from sohayok.audio import input_devices

    print("\n".join(input_devices()))


def cmd_listen(args: argparse.Namespace) -> None:
    """Record one utterance from the microphone and print the transcript."""
    from sohayok.audio import Recorder
    from sohayok.stt import Transcriber

    speech = config_module.current().speech
    transcriber = Transcriber(
        speech.whisper_model,
        speech.bangla_model,
        speech.device,
        speech.compute_type,
        speech.beam_size,
        speech.languages,
    )
    recorder = Recorder(
        speech.input_device or None,
        speech.silence_seconds,
        speech.max_record_seconds,
        start_timeout=10,
    )
    for _ in range(args.times):
        print("Speak now (Bangla or English)…", flush=True)
        audio = recorder.record()
        print(f"recorded {len(audio) / 16000:.1f} s", flush=True)
        if audio.size:
            heard = transcriber.transcribe(audio)
            print(json.dumps(heard.__dict__, ensure_ascii=False), flush=True)


def cmd_say(args: argparse.Namespace) -> None:
    from sohayok.tts import Speaker

    voice = config_module.current().voice
    Speaker(voice.bangla_voice, voice.english_voice, voice.rate).say(args.text)


def cmd_doctor(_: argparse.Namespace) -> None:
    """Check every dependency the assistant needs and say what to fix."""
    import httpx

    config = config_module.current()
    checks: list[tuple[str, bool, str]] = []
    try:
        import torch

        ok = torch.cuda.is_available()
        checks.append(
            (
                "GPU (CUDA)",
                ok,
                torch.cuda.get_device_name(0)
                if ok
                else "not available; "
                'speech recognition will run on CPU (set speech.device = "cpu")',
            )
        )
    except Exception as exc:
        checks.append(("GPU (CUDA)", False, str(exc)))
    try:
        tags = httpx.get(f"{config.brain.ollama_url}/api/tags", timeout=3).json()
        names = [m["name"] for m in tags.get("models", [])]
        has = config.brain.ollama_model in names
        checks.append(("Ollama", True, f"running; models: {', '.join(names) or 'none'}"))
        checks.append(
            (
                f"Model {config.brain.ollama_model}",
                has,
                "installed" if has else f"run: ollama pull {config.brain.ollama_model}",
            )
        )
    except Exception:
        checks.append(
            (
                "Ollama",
                config.brain.backend != "ollama",
                "not running; install from https://ollama.com or `winget install Ollama.Ollama`",
            )
        )
    import os

    key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    checks.append(
        (
            "Claude API key",
            key or config.brain.backend != "anthropic",
            "set" if key else 'not set (only needed for brain.backend = "anthropic")',
        )
    )
    try:
        from sohayok.audio import input_devices

        devices = input_devices()
        default = next(
            (d for d in devices if d.endswith("(default)")), devices[0] if devices else ""
        )
        checks.append(("Microphone", bool(devices), default or "no input device"))
    except Exception as exc:
        checks.append(("Microphone", False, str(exc)))
    try:
        from sohayok.tools.browser import chrome_path

        chrome = chrome_path()
        checks.append(
            (
                "Google Chrome",
                bool(chrome),
                chrome or "not found; websites open in the default browser",
            )
        )
    except Exception as exc:
        checks.append(("Google Chrome", False, str(exc)))
    from sohayok.models import resolve_bangla_model

    bangla = resolve_bangla_model(config.speech.bangla_model)
    checks.append(("Bangla speech model", bool(bangla), bangla or "run: sohayok setup"))
    from sohayok.tools.shell import gh_path

    checks.append(("GitHub CLI", bool(gh_path()), gh_path() or "winget install GitHub.cli"))
    checks.append(
        (
            "Claude Code",
            bool(shutil.which("claude")),
            shutil.which("claude") or "optional: https://claude.com/claude-code",
        )
    )
    try:
        from sohayok.tts import synthesize_edge

        samples, _ = synthesize_edge("পরীক্ষা", config.voice.bangla_voice)
        checks.append(("Bangla voice", samples.size > 0, config.voice.bangla_voice))
    except Exception as exc:
        checks.append(("Bangla voice", False, f"{exc} (needs internet)"))
    width = max(len(name) for name, _, _ in checks)
    for name, ok, detail in checks:
        print(f"{'OK ' if ok else 'FIX'}  {name.ljust(width)}  {detail}")
    print(f"\nSettings: {config_module.app_dir() / 'config.toml'}")


def cmd_config(_: argparse.Namespace) -> None:
    path = config_module.app_dir() / "config.toml"
    if not path.exists():
        defaults = config_module.Config()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            DEFAULT_CONFIG.format(
                ollama_model=defaults.brain.ollama_model,
                whisper_model=defaults.speech.whisper_model,
                bangla_model=defaults.speech.bangla_model,
            ),
            encoding="utf-8",
        )
        print(f"created {path}")
    subprocess.Popen(["notepad.exe", str(path)])


def _startup_shortcut() -> Path:
    return Path.home() / "AppData/Roaming/Microsoft/Windows/Start Menu/Programs/Startup/Sohayok.lnk"


def cmd_setup(args: argparse.Namespace) -> None:
    """Prepare the Bangla speech model and pull the local language model."""
    from sohayok.models import bangla_model_dir, prepare_bangla_model

    config = config_module.current()
    target = bangla_model_dir()
    if (target / "model.bin").exists() and not args.force:
        print(f"Bangla speech model ready: {target}")
    else:
        print("Preparing the Bangla speech model (about 3 GB download)…", flush=True)
        print(f"ready: {prepare_bangla_model(target)}")
    ollama = shutil.which("ollama") or str(Path.home() / "AppData/Local/Programs/Ollama/ollama.exe")
    if Path(ollama).exists():
        subprocess.run([ollama, "pull", config.brain.ollama_model], check=False)
    else:
        print("Ollama is not installed: winget install Ollama.Ollama")


def cmd_startup(args: argparse.Namespace) -> None:
    """Start Sohayok when you sign in to Windows (or stop doing so)."""
    shortcut = _startup_shortcut()
    if args.remove:
        shortcut.unlink(missing_ok=True)
        print(f"removed {shortcut}")
        return
    import win32com.client

    exe = Path(sys.executable).with_name("pythonw.exe")
    link = win32com.client.Dispatch("WScript.Shell").CreateShortcut(str(shortcut))
    link.TargetPath = str(exe)
    link.Arguments = "-m sohayok run --quiet"
    link.WorkingDirectory = str(Path.home())
    link.Description = "Sohayok voice assistant"
    link.Save()
    print(f"created {shortcut}")


def _ensure_streams() -> None:
    """Under pythonw (startup shortcut, no console) stdout and stderr are None, and the first
    progress bar or warning from a library would crash the app. Send them to a log file."""
    if sys.stdout is None or sys.stderr is None:
        log_dir = config_module.app_dir()
        log_dir.mkdir(parents=True, exist_ok=True)
        stream = open(log_dir / "console.log", "a", encoding="utf-8", buffering=1)  # noqa: SIM115
        sys.stdout = sys.stdout or stream
        sys.stderr = sys.stderr or stream
    else:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main(argv: list[str] | None = None) -> None:
    _ensure_streams()
    parser = argparse.ArgumentParser(prog="sohayok", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="start the assistant (default)")
    run.add_argument("--no-window", action="store_true", help="run without the window")
    run.add_argument("--quiet", action="store_true", help="no spoken greeting")
    run.set_defaults(func=cmd_run)

    do = sub.add_parser("do", help="run one typed command for real and print what happens")
    do.add_argument("text")
    do.add_argument("--language", choices=["bn", "en"])
    do.add_argument("--yes", action="store_true", help="approve confirmations")
    do.add_argument("--speak", action="store_true", help="speak the reply")
    do.set_defaults(func=cmd_do)

    sub.add_parser("devices", help="list microphones").set_defaults(func=cmd_devices)
    listen = sub.add_parser("listen", help="transcribe from the microphone")
    listen.add_argument("--times", type=int, default=1)
    listen.set_defaults(func=cmd_listen)
    say = sub.add_parser("say", help="speak text aloud")
    say.add_argument("text")
    say.set_defaults(func=cmd_say)
    sub.add_parser("doctor", help="check the setup").set_defaults(func=cmd_doctor)
    setup = sub.add_parser("setup", help="download and prepare the models")
    setup.add_argument("--force", action="store_true", help="rebuild the Bangla model")
    setup.set_defaults(func=cmd_setup)
    sub.add_parser("config", help="open the settings file").set_defaults(func=cmd_config)
    startup = sub.add_parser("startup", help="start with Windows")
    startup.add_argument("--remove", action="store_true")
    startup.set_defaults(func=cmd_startup)

    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    started = time.monotonic()
    (args.func if args.command else cmd_run)(
        args if args.command else parser.parse_args(["run", *(argv or sys.argv[1:])])
    )
    logging.getLogger(__name__).debug("finished in %.1f s", time.monotonic() - started)
