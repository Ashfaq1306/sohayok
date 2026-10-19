# সহায়ক · Sohayok

**Your own Siri for Windows, in Bangla and English.** Say **"Tammy"** and Sohayok pops up, listens,
and does what you ask: opens apps, plays YouTube, changes the volume, answers questions, types
for you, and much more. It runs quietly in the background from the moment you sign in, and it
all happens on your own PC.

| You say | Sohayok |
| --- | --- |
| Tammy, ক্রোম খোলো | opens Google Chrome |
| Tammy, ভলিউম পঞ্চাশ করো | sets the volume to 50% |
| Tammy, ইউটিউবে আইয়ুব বাচ্চুর গান চালাও | plays the top YouTube result |
| Tammy, নোটপ্যাড খুলে লেখো: আমি ভালো আছি | opens a blank Notepad and types it |
| Tammy, ঢাকার আবহাওয়া কেমন? | tells you the weather, aloud |
| Tammy, what's on my GitHub? | reads your notifications and pull requests |
| Tammy, chrome bondho koro | closes Chrome |
| Tammy, কম্পিউটার রিস্টার্ট করো | asks "কম্পিউটার রিস্টার্ট করব? হ্যাঁ বা না বলুন।" first |

---

## Get started (one time, about 15 minutes)

You need Windows 10 or 11, a microphone, and ideally an NVIDIA graphics card (8 GB works well;
without one it still runs, just slower).

1. **Install the two helpers** (in PowerShell):

   ```powershell
   winget install astral-sh.uv          # Python environment manager
   winget install Ollama.Ollama         # runs the small AI model locally
   ```

2. **Get Sohayok and its models:**

   ```powershell
   git clone https://github.com/Ashfaq1306/sohayok.git
   cd sohayok
   uv sync --extra wake
   uv run sohayok setup                 # downloads the speech models, the decision model, and the AI model
   ```

3. **Check everything works:**

   ```powershell
   uv run sohayok doctor
   ```

   Every line should say `OK`. If one says `FIX`, it tells you what to do.

4. **Make it start with Windows** (and add Start-menu and Desktop shortcuts with its face icon):

   ```powershell
   uv run sohayok install
   ```

5. **Start it now:** double-click the **Sohayok** icon on your Desktop, or run `uv run sohayok`.

That's it. From now on Sohayok starts by itself when you sign in and waits in the system tray
(the little face next to the clock).

## Everyday use

- **Say "Tammy"**, wait for the soft chime, then say what you want. Or say it all at once:
  "Tammy, ক্রোম খোলো". A clear "Tammy" opens the window instantly; an unclear one is
  double-checked by the speech recognizers first (about half a second).
- **Say "Tammy" while it is talking** to cut it off and give a new command, like Siri.
- **Teach it your voice** once: `uv run sohayok train-wake --record 10` (say "Tammy" ten
  times; a few minutes on the CPU). It also learns from the "Tammy"s you said in daily use.
- **Or press <kbd>Ctrl</kbd>+<kbd>Alt</kbd>+<kbd>Space</kbd>** and speak, or type in the box.
- Speak **Bangla, English, Banglish, or a mix**. Sohayok answers in the language you used.
- The window **hides by itself** when you're done. Click ✕ to hide it now, 📌 to keep it open.
- **Tray icon** (right-click the face): show, listen now, hide, or quit.
- Say **"থামো"** or **"stop"** to make it stop talking.

## How it works

```
  "Tammy, ক্রোম খোলো"
        │
   1. Wake word ──── a tiny "Tammy" detector (trained on your voice) listens on the CPU;
        │            noises without speech are ignored, unclear calls are double-checked
        │
   2. Hearing ────── a small language classifier (with Whisper when unsure) tells Bangla from
        │            English in ~40 ms; BuzzASR (a Bangla Whisper) writes Bangla, Whisper English
        │
   3. Deciding ───── everyday commands are recognised instantly; otherwise Laya-Bangla (fine-tuned
        │            for these actions) picks one of 28 actions in ~0.2 s on the CPU, and the
        │            answer model fills in the details: qwen3.5:2b on this PC, or an API you choose
        │
   4. Doing ──────── one of 48 actions: apps, windows, keyboard, mouse, files, volume, Chrome,
        │            YouTube, search, weather, reminders, GitHub, settings, PowerShell…
        │
   5. Answering ──── a short spoken reply in a Bangladeshi Bangla or English voice
```

With the default settings, everything except the spoken voice and web searches runs on your
PC: the wake word, speech recognition, the decision model, and the answer model. Nothing is
recorded unless you speak to it.

**GPU use.** Speech recognition stays on the GPU (about 2.5 GB) so it answers quickly; the
decision model runs on the CPU, and the answer model unloads itself two minutes after your
last command. Quit Sohayok from the tray to free the GPU completely.

## Safety

Sohayok can do almost anything you can, so it is careful with what can't be undone:

- It **asks first** before shutting down, restarting, deleting (Recycle Bin only), moving or
  overwriting files, running PowerShell commands, or handing work to Claude Code. Answer
  "হ্যাঁ"/"না" or "yes"/"no", or press the buttons. A spoken "yes" must be short and clear.
- It **never claims** something happened when you said no or it failed.
- It asks before sending, posting, buying, or submitting anything in an app or website.
- It **never types passwords**, PINs, card numbers, or one-time codes.
- Text it reads from websites, files, or the screen is treated as information, never as
  instructions.

## Settings

Click **⚙** in the window:

- **Answer model:** "This PC (Ollama)" is free and private. Or pick **OpenAI, Google Gemini,
  Z.ai GLM, OpenRouter, Anthropic Claude**, or any **OpenAI-compatible** API (DeepSeek, Groq,
  LM Studio, …), paste your API key, press **Load models**, and choose a model. Keys are kept
  in Windows Credential Manager. With an API, the text of your request goes to that provider.
- **Behaviour:** speak replies, ask before risky actions, wake word.

Everything else is in `%APPDATA%\Sohayok\config.toml` (`uv run sohayok config`):

```toml
[assistant]
default_language = "bn"         # language of the greeting
home_city = "Dhaka"             # for weather
confirm_risky_actions = true

[activation]
wake_model = "tammy"            # or "hey_jarvis"
wake_threshold = 0.3            # lower = wakes more easily, higher = fewer false wake-ups
wake_instant_threshold = 0.8    # above this, wake at once without the Whisper check
hotkey = "<ctrl>+<alt>+<space>"

[brain]
backend = "lite"                # Laya decides + answer model; "ollama" or "anthropic" = one model does all
provider = "ollama"             # set from the Settings panel
lite_model = "qwen3.5:2b"

[speech]
keep_recordings = 20            # last voice commands kept on this PC for checking; 0 = off
```

**Claude.** With `backend = "anthropic"` and an `ANTHROPIC_API_KEY`, Claude does the thinking
and can also see the screen and use the mouse in any app. Screenshots then go to Anthropic.

## Troubleshooting

| Problem | Try |
| --- | --- |
| It doesn't wake up | Run `uv run sohayok train-wake --record 10` so it learns your voice; or use the hotkey |
| It wakes up by itself | Raise `wake_threshold` to 0.5, or `wake_instant_threshold` to 0.9 |
| It cuts you off mid-sentence | Raise `silence_seconds` (default 0.7) to 1.0 in `settings.json` |
| It hears the wrong words | Check the microphone in Windows Settings → Sound (input level 80–100%); the last recordings are in `%APPDATA%\Sohayok\recordings` with what it heard |
| "voice unavailable" | Run `uv run sohayok setup`, then `uv run sohayok doctor` |
| Slow answers | Other GPU programs (training, games) squeeze Sohayok out of GPU memory and make speech recognition very slow; close them. The first command after a pause reloads the answer model (2–3 s) |
| Remove it | `uv run sohayok install --remove`, then quit from the tray icon |

Logs: `%APPDATA%\Sohayok\sohayok.log`.

## Commands

```powershell
uv run sohayok                 # start (window + tray + wake word)
uv run sohayok run --quiet     # start hidden in the tray (what Windows start-up uses)
uv run sohayok do "ভলিউম ৩০ করো"   # run one typed command for real and show each step
uv run sohayok listen          # say something and see what it heard
uv run sohayok train-wake --record 10   # teach the "Tammy" detector your voice
uv run sohayok say "আমি সহায়ক"    # speak text
uv run sohayok devices         # list microphones
uv run sohayok doctor          # check the setup
```

## Benchmarks

Measured on an RTX 4060 (8 GB). Methods, caveats, and full tables:
[docs/benchmarks.md](docs/benchmarks.md).

| What | Result |
| --- | --- |
| Typed commands, right action (478: MASSIVE test + desktop) | **91.4%** with Laya + qwen3.5:2b, vs 70.7% for qwen3.5:4b alone; Bangla 90.8% (was 65.7%) |
| Decision model alone | 91.8% (63.4% before fine-tuning) |
| Bangla speech recognition, FLEURS (200 real sentences) | 21.6% WER / 9.1% CER, vs 77.2% / 39.4% for Whisper large-v3 |
| Spoken commands, Bangla / English WER | 15.9% / 9.9%; a quiet microphone does as well (15.7%) |
| Bangla or English: your real Bangla commands / accented English / FLEURS | 10 of 10 (Whisper alone: 5) / 97.1% / 100% |
| Wake word, the user's real "Tammy"s it was not trained on | 9 of 9 with the personal model (shared model: 4 of 9); 0 of 31 noises |
| Wake word on unseen voices / false wakes in 1 h of Bangla speech | 97.7% instant + checked rest / 0 |

Models: decider [`ashfaq1306/sohayok-laya-actions`](https://huggingface.co/ashfaq1306/sohayok-laya-actions)
(fine-tuned from [`algolyzer/laya-bangla`](https://huggingface.co/algolyzer/laya-bangla)),
wake word `src/sohayok/assets/tammy.onnx`.

## For developers

```powershell
uv sync --all-extras
uv run pytest -q                                          # unit tests, no hardware needed
$env:SOHAYOK_DESKTOP_TESTS=1; uv run pytest -m desktop      # read-only checks on the real desktop
uv run ruff check . ; uv run ruff format --check .
```

Training scripts: `sohayok.training.actions_data` and `finetune_decider` (the Laya action
decider; `uv run sohayok setup --train-decider` rebuilds it) and `sohayok.training.wake_word`
(the "Tammy" detector). Where the project stands and what to do next:
[docs/HANDOFF.md](docs/HANDOFF.md).

## Credits and licenses

Sohayok's code is Apache-2.0. Models keep their own licenses:
[Laya-Bangla](https://huggingface.co/algolyzer/laya-bangla) (Apache-2.0, from
[Laya](https://huggingface.co/convaiinnovations/laya) by Convai Innovations),
[Whisper large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) (MIT),
[BuzzASR Bengali](https://huggingface.co/BuzzASR/bengali) (MIT),
[Qwen3.5](https://huggingface.co/Qwen) (Apache-2.0),
[openWakeWord](https://github.com/dscripka/openWakeWord) (Apache-2.0),
[Silero VAD](https://github.com/snakers4/silero-vad) (MIT). Data used for benchmarks and
training: [FLEURS](https://huggingface.co/datasets/google/fleurs) (CC BY 4.0) and
[MASSIVE](https://huggingface.co/datasets/mteb/amazon_massive_intent) (CC BY 4.0).
