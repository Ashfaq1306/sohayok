# সহায়ক · Sohayok

**A voice assistant for Windows that understands Bangla and English and controls the whole desktop.**

Say "Hey Jarvis" (or press <kbd>Ctrl</kbd>+<kbd>Alt</kbd>+<kbd>Space</kbd>), wait for the
chime, then talk in Bangla, English, Banglish, or a mix:

| You say | Sohayok |
| --- | --- |
| ক্রোম খোলো | opens Google Chrome |
| ভলিউম পঞ্চাশ করো | sets the volume to 50% |
| ইউটিউবে আইয়ুব বাচ্চুর গান চালাও | plays the top YouTube result |
| নোটপ্যাড খুলে লেখো: আমি ভালো আছি | opens a blank Notepad and types the Bangla text |
| ঢাকার আবহাওয়া কেমন? আর র‍্যাম কত ব্যবহার হচ্ছে? | answers both, aloud, in Bangla |
| what's on my GitHub? | reads your notifications, PRs, and assigned issues |
| কম্পিউটার রিস্টার্ট করো | asks "কম্পিউটার রিস্টার্ট করব? হ্যাঁ বা না বলুন।" first |
| chrome bondho koro | closes Chrome |

It runs on your own PC: speech recognition, the wake word, and the default language model
all run locally on the GPU. Replies are spoken with Microsoft's Bangladeshi Bangla and
English neural voices.

## How it works

```
 microphone ─▶ wake word / hotkey ─▶ voice activity detection ─▶ speech recognition
   (openWakeWord "hey jarvis")        (Silero VAD)               Whisper large-v3-turbo detects bn/en,
                                                                  BuzzASR Bengali transcribes Bangla
                                                     │
                                                     ▼
            fast path: everyday commands in Bangla, Banglish, English (instant, no model)
                                                     │ anything else
                                                     ▼
            language model with 48 tools (Ollama locally, or Claude with computer use)
                                                     │
                                                     ▼
       safety layer: risky tools need your "হ্যাঁ"/"yes" ─▶ Windows (apps, windows, keyboard,
                                                              mouse, files, volume, Chrome, GitHub…)
                                                     │
                                                     ▼
                              spoken reply (bn-BD / en voices) + the Sohayok window
```

**Tools** (48): open, find, and close apps; switch, minimize, maximize, snap, and close
windows; type text (Bangla included), press shortcuts, click, scroll; read the active
window's buttons and fields and click them by name; screenshots; open websites, search
Google/YouTube/Wikipedia/Maps/GitHub, play YouTube, look things up on Wikipedia and the web;
volume and media keys; Settings pages; lock, sleep, shut down, restart; find, open, read,
create, move, and recycle files; notes; clipboard; timers and reminders; weather; system
status (CPU, RAM, GPU, disks); GitHub overview and repositories (via `gh`); PowerShell; and
handing big jobs to Claude Code.

## Safety

Sohayok can do almost anything you can, so it is careful about what cannot be undone:

- **Asks first** before shutting down, restarting, signing out, deleting (to the Recycle
  Bin only, never permanently), moving or overwriting files, running PowerShell, or handing a
  task to Claude Code. Answer by voice ("হ্যাঁ" / "না", "yes" / "no") or with the buttons.
  Spoken "yes" must be short and clearly spoken; noise and long sentences are not accepted.
- **Never claims** a declined or failed action happened.
- Before sending, posting, buying, or submitting anything inside an app, the model must ask
  you with `confirm_with_user`.
- **Never types passwords**, PINs, card numbers, or one-time codes; it asks you to.
- Text read from web pages, files, and the screen is treated as information, not
  instructions (prompt-injection defense).
- Typing never lands in Sohayok's own window or overwrites selected text, and dictation into
  Notepad starts a blank tab instead of editing a file Notepad reopened.

Turn confirmations off at your own risk with `confirm_risky_actions = false`.

## Install

Requirements: Windows 10/11, an NVIDIA GPU with 8 GB (CPU works, slowly), a microphone,
[uv](https://docs.astral.sh/uv/), and [Ollama](https://ollama.com) (`winget install Ollama.Ollama`).

```powershell
git clone <this repository> sohayok
cd sohayok
uv sync --extra wake      # Python 3.12 environment with CUDA PyTorch; the extra adds the wake word
uv run sohayok setup      # Bangla speech model (~3 GB) + the local language model
uv run sohayok doctor     # checks GPU, Ollama, microphone, voices, Chrome, GitHub CLI
uv run sohayok            # starts the assistant
```

Start it with Windows: `uv run sohayok startup` (undo with `--remove`).

### Settings

`uv run sohayok config` opens `%APPDATA%\Sohayok\config.toml`:

```toml
[assistant]
default_language = "bn"     # greeting and fallback language
home_city = "Dhaka"         # for weather
confirm_risky_actions = true

[brain]
backend = "ollama"          # or "anthropic" for Claude (set ANTHROPIC_API_KEY)
ollama_model = "qwen3.5:4b"

[activation]
hotkey = "<ctrl>+<alt>+<space>"
wake_word = true            # "hey jarvis" (false: hotkey only)
```

**Claude backend.** With `backend = "anthropic"` and an `ANTHROPIC_API_KEY`, Sohayok uses
Claude with the computer-use toolset: it sees screenshots and can operate any app or web page
with the mouse and keyboard, on top of the 48 tools. Screenshots are sent to Anthropic.

## Commands

```powershell
uv run sohayok                         # assistant window + hotkey (+ wake word)
uv run sohayok run --no-window         # background only
uv run sohayok do "ভলিউম ৩০ করো"         # run one typed command for real and print each step
uv run sohayok listen                  # transcribe one utterance from the microphone
uv run sohayok say "আমি সহায়ক"           # speak text
uv run sohayok devices                 # list microphones
```

The window is also served at <http://127.0.0.1:8765> (local only).

## Privacy

Stays on your PC: audio, transcripts, wake word, the local language model, files, and
screen contents. Leaves your PC: reply text (to Microsoft's text-to-speech service), web
searches and lookups (Google, YouTube, Wikipedia, DuckDuckGo, Open-Meteo), and everything
the Claude backend needs (requests, tool results, screenshots) if you enable it.

## Benchmarks

Measured on an RTX 4060 (8 GB). Methods, caveats, and full tables:
[docs/benchmarks.md](docs/benchmarks.md).

| What | Result |
| --- | --- |
| Bangla speech recognition, FLEURS (200 real sentences) | **21.6% WER / 9.1% CER** with BuzzASR, vs 77.2% / 39.4% for Whisper large-v3 |
| English speech recognition, FLEURS (100 sentences) | 4.7% WER (large-v3-turbo) |
| Bangla or English? (spoken commands / FLEURS) | 95–97% / 100% correct |
| Typed commands: right tool, qwen3.5:4b (478 cases) | 70.7% overall, **100% of 60 desktop commands**, 66.5% of MASSIVE requests |
| Spoken commands end to end (204 clips) | 75.5% right tool, 90% for desktop commands; 97.5% replies in the user's language |
| Speed | ~0.6 s to transcribe, ~1.2 s to understand (median) |

Local models compared: qwen3.5:4b (70.7%) beat qwen3.5:9b (61.3%, slower) and gemma4:e4b
(40.6%) on the same commands.

## Development

```powershell
uv sync --all-extras
uv run pytest -q                                       # unit tests, no hardware needed
$env:SOHAYOK_DESKTOP_TESTS=1; uv run pytest -m desktop   # read-only checks on the real desktop
uv run ruff check . ; uv run ruff format --check .
uv run python -m sohayok.bench.stt   --help             # speech recognition (FLEURS)
uv run python -m sohayok.bench.brain --help             # command understanding (MASSIVE + desktop)
uv run python -m sohayok.bench.voice --help             # spoken command → action
```

## Credits and licenses

Sohayok's code is Apache-2.0. Models and data keep their own licenses:
[Whisper large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) (MIT),
[BuzzASR Bengali](https://huggingface.co/BuzzASR/bengali) (MIT; Whisper large-v3 fine-tuned
for Bengali), [Qwen3.5](https://ollama.com/library/qwen3.5) (Apache-2.0),
[openWakeWord](https://github.com/dscripka/openWakeWord) (Apache-2.0; its pretrained
"hey jarvis" model is CC BY-NC-SA 4.0, for non-commercial use),
[Silero VAD](https://github.com/snakers4/silero-vad) (MIT),
[FLEURS](https://huggingface.co/datasets/google/fleurs) (CC BY 4.0), and
[MASSIVE](https://huggingface.co/datasets/mteb/amazon_massive_intent) (CC BY 4.0).
