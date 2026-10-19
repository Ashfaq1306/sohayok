# Benchmarks

Measured on 2026-10-18 on one PC: NVIDIA GeForce RTX 4060 (8 GB), AMD Ryzen 7 9700X, 62 GB
RAM, Windows 11, Python 3.12, PyTorch 2.8.0+cu128, CTranslate2 4.8.3, faster-whisper 1.2.1,
Ollama 0.40.5. Commands to reproduce are at the end.

Three questions, three benchmarks:

1. **Does it hear Bangla and English correctly?** Speech recognition on FLEURS, real read
   speech by many speakers.
2. **Does it do the right thing?** Tool choice and reply language for typed commands: MASSIVE
   assistant requests in Bangla and English, plus hand-written desktop commands.
3. **Does it all work together from speech?** Spoken commands, end to end.

## 1. Speech recognition (FLEURS)

[FLEURS](https://huggingface.co/datasets/google/fleurs) (CC BY 4.0) test splits: 200 Bangla
(`bn_in`) and 100 English (`en_us`) sentences, one recording per sentence, chosen with a fixed
seed. Transcripts are compared after Unicode NFC, lowercasing, and removing punctuation
(including the dari "।"). Python's regex `\w` does not match Bangla vowel signs or the
hasanta, so punctuation is removed by Unicode category instead; an earlier version of this
script used `\w` and over-counted Bangla errors.

| Model | Bangla WER | Bangla CER | English WER | English CER |
| --- | ---: | ---: | ---: | ---: |
| Whisper large-v3-turbo (CTranslate2 int8) | 87.3% | 49.6% | 4.7% | 2.0% |
| Whisper large-v3 (CTranslate2 int8) | 77.2% | 39.4% | 4.3% | 2.0% |
| **BuzzASR Bengali (CTranslate2 int8, as shipped by `sohayok setup`)** | **21.6%** | **9.1%** | — | — |

- BuzzASR is Bangla-only, so Sohayok uses large-v3-turbo to tell Bangla from English and to
  transcribe English, and BuzzASR for Bangla. Off-the-shelf Whisper drifts into Devanagari and
  truncates long Bangla sentences; BuzzASR's native Bangla tokenizer avoids both.
- BuzzASR was fine-tuned on FLEURS and Common Voice **training** data; these are the
  held-out FLEURS **test** recordings, but they are in-domain for it. Its model card reports
  31.97% WER / 10.16% CER on FLEURS with its own normalization.
- On a 20-sentence subset, the int8 CTranslate2 conversion scored 16.0% WER / 4.1% CER and the
  original float16 Transformers model 29.7% / 8.5% with the decoding settings from its model
  card (repetition penalty 1.2, no repeated 3-grams), which visibly corrupted some words.
- Many remaining "errors" are numbers spoken as words ("সত্তর") where the reference has digits
  ("৭০"), and spelling variants (মন্ত্রিসভা / মন্ত্রীসভা).

### Converting BuzzASR correctly

Two problems made a naive conversion silently wrong:

1. **Token ids.** The repository's `vocab.json` lists ~1,500 special strings (`<|endoftext|>`,
   `<|transcribe|>`, …) at low ids, while `added_tokens.json` places them at 51865 and up. A
   fast tokenizer built from these files assigns `<|bn|>` the id 51908 instead of 51910, so
   the model would be prompted with the wrong language token. `sohayok setup` renames the
   clashing vocabulary entries, keeps every id, checks the special-token ids against
   `generation_config.json`, and confirms that 200/200 reference sentences tokenize
   identically and round-trip exactly.
2. **"English-only".** CTranslate2 infers from the vocabulary layout that the converted model
   is English-only, so faster-whisper would drop the `<|bn|>` prompt and decode as English.
   Sohayok records the 100 language tokens in the model config and tells faster-whisper the
   model is multilingual.

### Bangla or English?

**Update 2 (mixed Bangla).** In the next real session the user's commands mixed English words
into Bangla ("ইউটিউবের ট্যাবটা কেটে ফেলো"), and the classifier + Whisper rule sent most of them
to English Whisper ("YouTube tab KT Fellow"). The Bangla model writes such speech well, so the
order changed (`Transcriber._recognise`): clear English (classifier margin < −3.7, or < 3 with
Whisper p(en) ≥ 0.9) goes to Whisper; everything else to the Bangla model; a Bangla transcript
with no Bangla grammar (`lang.has_bangla_grammar`: pronouns, particles, verb forms, endings
like "-ের"/"-টা") is English written in Bangla letters and is redone in English.

| Set (clips) | v0.2 rule: right language / WER | **Now** |
| --- | --- | --- |
| The user's real commands (9) | 4 of 9 | **9 of 9** (incl. "Please open this PC" in English) |
| Spoken Bangla commands (37) | 94.6% / 18.9% | 94.6% / 19.4% |
| Spoken English commands, Indian voice (31) | 100% / 5.8% | 96.8% / 5.8% |
| FLEURS Bangla validation, English (20 + 20) | 100% / 19.9%, 4.6% | 100% / 19.9%, 4.6% |

107 of 117 clips need one decoding pass, 10 need two.

**GPU priority.** With other training jobs keeping the GPU at 100%, recognising a 2 s command
took 18 s at normal Windows GPU priority, 13 s at high, and 3.4 s at real-time (`sohayok.gpu`);
in the end-to-end run below (training still running) the median was 0.45 s. Sohayok raises
its priority only while it recognises speech (`speech.gpu_priority`).

**End to end** (`python -m sohayok.bench.dry_run`): 27 spoken test commands (Bangladeshi Bangla
and Indian/US English voices: This PC, drives, new folders, tabs, WhatsApp, YouTube, volume,
weather…) plus 6 of the user's own recordings, through the real recognizers, decider, and
answer model, with every tool replaced by a recorder. **32 of 33** reached the right action
with Claude Sonnet 5.5 or Haiku 5.5; 21 were handled by instant rules (~0.5 s after speech
ends), the rest took a median of 2.0 s (Sonnet) / 2.1 s (Haiku) for the answer model. The miss
is a recording the Bangla model hears as "গুল্ড্রমণ করো" ("Google Chrome open koro"); beam
search (3 or 5) does not fix it.

**Update (v0.2, real use).** Replaying the user's own recorded commands showed Whisper's
detector giving their Bangla a probability of **0.00 every time**; with the rule below, half
of their real Bangla commands (5 of 10) went to the English recognizer and came out as
nonsense ("Let the box go.", "you are stuck in the Jimin…"). Sohayok now asks a dedicated
spoken-language classifier first, [VoxLingua107 ECAPA](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa)
(21M parameters, CPU), and adds Whisper's opinion only when the classifier is not sure. The
classifier alone mistakes Indian-accented English for Bangla (30% of the en-IN clips), so
neither works alone: `stt.bangla_or_english` combines them (Bangla if margin − logit p(en) > 5.5).

| Set | Whisper rule (old) | VoxLingua alone | **Combined (now)** | Time |
| --- | ---: | ---: | ---: | ---: |
| The user's real Bangla commands (10) | 50% | 100% | **100%** | ~210 ms |
| Spoken Bangla commands, synthetic (102) | 95.1% | 98.0% | **97.1%** | ~30 ms |
| Spoken English commands, Indian voice (102) | 97.1% | 69.6% | **97.1%** | ~210 ms |
| FLEURS Bangla validation, full and first 3 s (100 + 100, never used for tuning) | – | – | **100%** | ~25–40 ms |
| FLEURS English, full and first 3 s (100 + 100) | 100% | 100% | **100%** | ~25–40 ms |

The rule's two numbers were fitted on the first three rows and FLEURS *test* clips; the
FLEURS Bangla validation rows are held out. The user's English (Bangladeshi accent) has not
been tested, as none of their kept recordings was in English.

Two more fixes from the same replay: a recording with no speech in it is no longer decoded
(the Bangla model ran for 13–17 s on 1 s of silence, retrying at higher temperatures), and
decoding is a single greedy pass with a length limit. The user's real commands now take
0.3–0.7 s to recognise (1.0–1.3 s for 10 s monologues).

The rest of this section is the earlier (v0.1) analysis.

Whisper's language detector often hears **short** Bangla commands as Hindi or Gujarati. Taking
the likelier of Bangla and English sent most spoken Bangla commands to the English
recognizer. Sohayok now treats anything that is not clearly English (p(en) < 0.5) as Bangla:

| Rule | Spoken Bangla commands | Spoken English commands | FLEURS Bangla | FLEURS English |
| --- | ---: | ---: | ---: | ---: |
| more likely of bn and en | 44.1% | 100% | 99% | 100% |
| **English only if p(en) ≥ 0.5** | **95.1%** | **97.1%** | **100%** | **100%** |

Spoken commands: the 102 + 102 clips of benchmark 3. FLEURS: 100 + 100 recordings. Two of
the three "English" clips sent to the Bangla recognizer are Banglish ("chrome bondho koro"),
which BuzzASR writes in Bangla script, where the fast path understands it. The five Bangla
clips sent to the English recognizer are mostly full of English loanwords ("ডাউনলোড ফোল্ডার
খোলো", "ওয়াইফাই সেটিংস খোলো").

## 2. Command understanding (typed)

Each case is one request; every tool is replaced by a stub that records the call and returns
a plausible result, and confirmations are answered "yes". A case passes when the expected
tools are called in order (any of the accepted tools per step), and "+ args" also requires
key arguments (for example volume level 50, city Sylhet). Reply language counts replies in
Bangla script for Bangla requests and in English otherwise.

- **MASSIVE** ([Amazon](https://huggingface.co/datasets/mteb/amazon_massive_intent),
  CC BY 4.0) test utterances in `bn-BD` and `en-US`, for the 18 intents that map onto
  Sohayok's tools (alarms → reminders, volume, music, time, weather, questions, news, email,
  jokes): up to 12 utterances per intent, in both languages, 418 cases. Several MASSIVE
  intents fit a PC poorly (tuning an FM radio, podcast episodes, "email query" for an email
  app the PC may not have), so these numbers understate desktop accuracy.
- **Desktop commands** (`src/sohayok/bench/desktop_commands.jsonl`): 60 hand-written
  requests in Bangla, English, and Banglish for what MASSIVE lacks (apps, windows, typing,
  files, settings, GitHub, power, clipboard, screen). Written by the developer: a smoke test,
  not a representative sample.

| Pipeline | All (478) | MASSIVE (418) | Desktop (60) | Bangla (239) | English (239) | Reply language | Median s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fast path only (no model) | 11.1% | 6.2% | 45.0% | 10.9% | 11.3% | — | 0.00 |
| gemma4:e4b-it-qat | 40.6% | 34.7% | 81.7% | 22.2% | 59.0% | 94.6% | 1.44 |
| qwen3.5:9b (prompt v2) | 61.3% | 56.5% | 95.0% | 52.3% | 70.3% | 97.7% | 1.83 |
| qwen3.5:4b (prompt v1) | 70.1% | 66.3% | 96.7% | 65.7% | 74.5% | 97.9% | 1.08 |
| qwen3.5:4b (final prompt) | 70.7% | 66.5% | 100% | 65.7% | 75.7% | 97.9% | 1.20 |
| Laya fine-tune + qwen3.5:0.8b | 82.2% | 80.1% | 96.7% | 76.6% | 87.9% | 96.4% | 0.95 |
| **Laya fine-tune + qwen3.5:2b (default, "lite")** | **91.4%** | **90.4%** | **98.3%** | **90.8%** | **92.1%** | **96.9%** | **2.37** |

The **lite** pipeline asks Laya-Bangla which kind of action is wanted (28 choices, ~15–60 ms on the GPU or ~160 ms on the CPU),
then gives the small model only that action's one to seven tools; simple successes get a
built-in Bangla or English reply. It is both lighter (one 2.7 GB model instead of 3.3 GB) and
far more accurate, mostly on Bangla. Its median time was measured while another training job
shared the GPU; the 0.8b row also swapped models in and out of GPU memory.

### The action decider

Laya-Bangla ([`algolyzer/laya-bangla`](https://huggingface.co/algolyzer/laya-bangla)) was
fine-tuned to pick one of Sohayok's 28 actions (`sohayok setup --train-decider`, 65 min on the
RTX 4060, peak 3.2 GB):

- **Training data:** MASSIVE `bn-BD` and `en-US` *train* splits (23,028 real requests) with
  their 60 intents mapped to actions (intents a PC cannot do become "other"), plus 1,301
  template sentences for desktop actions MASSIVE lacks, in Bangla, Banglish, and English
  (synthetic; generated sentences within 90% similarity of a benchmark command were
  dropped). Large classes were capped and rare ones topped up to 400 with varied phrasing.
- **Validation:** MASSIVE validation split plus 10% of the template sentences; best epoch by
  validation loss (epoch 2, 88.3% validation accuracy).

| Decider | All (478) | MASSIVE test (418) | Desktop (60) | Bangla | English |
| --- | ---: | ---: | ---: | ---: | ---: |
| Laya-Bangla, zero-shot | 63.4% | 65.1% | 51.7% | 65.3% | 61.5% |
| **Laya-Bangla fine-tuned** | **91.8%** | **92.3%** | **88.3%** | **91.6%** | **92.1%** |

The MASSIVE test split was never used for training or selection. The desktop commands were
written by the same developer as the templates, so treat that column as optimistic.

- Every row except "fast path only" includes the fast path, which answers 6.5% of cases
  instantly; 27 of its 31 answers pass, and the four others are MASSIVE requests that start
  with "search …" or say "next", where searching or skipping is a reasonable reading.
- qwen3.5:9b is both slower and less accurate here; at 6.4 GB it also does not fit beside the
  speech models in 8 GB, so part of it runs on the CPU.
- gemma4 and prompt v1 ran while another benchmark shared the GPU, so their times are high;
  accuracy is unaffected.
- **Caveat:** the prompt was revised twice against this same set (v1 → v2 fixed jokes being
  searched instead of told; v2 → final restored looking up facts instead of answering from
  memory). The final numbers are therefore somewhat optimistic.
- Remaining weak spots: radio, podcasts, and audiobooks (5–6 of 24 each), "check my email"
  when no mail app is configured (4/24), and vague volume requests.

## 3. Spoken commands, end to end

The 60 desktop commands plus 4 MASSIVE utterances per intent in both languages (204 clips)
were synthesized with Microsoft neural voices (`bn-BD-NabanitaNeural`, `bn-BD-PradeepNeural`,
`en-US-AvaMultilingualNeural`, `en-IN-PrabhatNeural`), resampled to 16 kHz, and passed through
the real pipeline: language detection → recognizer → fast path or qwen3.5:4b → stubbed tools.
Synthetic speech is clean and evenly paced, so treat this as an upper bound for real
microphones; FLEURS above covers real voices.

| | All (204) | Desktop (60) | MASSIVE (144) | Bangla (102) | English (102) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Right tool | **75.5%** | 90.0% | 69.4% | 71.6% | 79.4% |
| Right tool + args | 74.0% | 85.0% | 69.4% | 70.6% | 77.5% |
| Reply in the user's language | 97.5% | 98.3% | 97.2% | 99.0% | 96.1% |

| Speech recognition on the commands | WER | CER | Right language | Median time |
| --- | ---: | ---: | ---: | ---: |
| Bangla | 16.2% | 7.8% | 95.1% | 0.60 s |
| English | 10.4% | 6.4% | 97.1% | 0.39 s |

Before the language-detection fix, Bangla commands had 57.9% WER and picked the right tool
54.9% of the time. A typical spoken command takes about 0.6 s to transcribe after you stop
speaking (the recorder waits 0.9 s of silence first) and 1.2 s to understand; the action
itself is then immediate, except for web lookups.

The pipeline row above used qwen3.5:4b; the spoken-command run has not been repeated with the
lite pipeline yet, whose typed-command accuracy is 20 points higher (section 2).

**Quiet microphones and beam search** (same 204 clips, speech recognition only; times are
inflated because another training job shared the GPU):

| Setting | Bangla WER / CER | English WER / CER | Right language bn / en |
| --- | --- | --- | --- |
| Normal level, beam 1 (default) | 15.9% / 7.2% | 9.9% / 5.7% | 96.1% / 98.0% |
| Quiet microphone (−22 dB), beam 1 | 15.7% / 7.2% | 10.6% / 6.2% | 96.1% / 97.1% |
| Normal level, beam 5 | 16.2% / 7.2% | 9.9% / 5.6% | 96.1% / 98.0% |

Recordings are now levelled before recognition, and the speech detector adapts its gain, so a
quiet microphone recognizes as well as a normal one. A wider beam search does not help these
short commands, so Sohayok keeps beam 1.

Typical remaining errors: "নতুন ট্যাব" heard as "নতুন ট্যাপ" (now handled by the fast path),
punctuation inside dictated text, and Banglish with English-voice pronunciation.

## 4. Wake word ("Tammy")

There is no ready-made "Tammy" model, so one was trained (`sohayok.training.wake_word`) on top
of openWakeWord's frozen speech embeddings:

- **Positives:** "Tammy", "Hey Tammy", "Tammy!", "Tammy?", "টামি", "ট্যামি", "হেই টামি" from
  Microsoft neural voices in English (US, UK, India, …) and South Asian languages, at three
  speeds and pitches, each copied six times with speed changes, echo, noise, and volume
  from normal down to a very quiet microphone (−44 dB). 2,826 training clips.
- **Negatives:** FLEURS Bangla test and English test speech (2 s windows), look-alike words
  ("Tommy", "tummy", "Timmy", "timer", "তুমি", "আমি", "টমেটো", …), and silence. 8,664 clips.
- **Held out:** 20% of the voices (never heard in training) and FLEURS Bangla *validation*
  speech (never used at all).

Live, the detector runs on the CPU 12.5 times a second; when it fires, Whisper large-v3-turbo
transcribes the last 2 s and the wake is accepted only if it heard the name.

| Held-out test | Result |
| --- | ---: |
| "Tammy" phrases from 43 unseen voice/phrase pairs: detector fires | 100% |
| … and Whisper confirms (the app wakes) | **93%** |
| Unseen Bangla speech (1.0 h, FLEURS validation): detector candidates | 18 per hour |
| … false wake-ups after the Whisper check | **0 per hour** |

**Instant wake.** Waiting for Whisper makes waking feel slow, so the detector's own score is
trusted when it is high. On the same held-out audio, 97.7% of "Tammy"s scored ≥ 0.8 and no
frame of the hour of Bangla speech did (≥ 0.7: 2 frames per hour; ≥ 0.5: 15). Sohayok now
wakes immediately at ≥ 0.8 and asks Whisper only between 0.3 and 0.8.

### With the user's real voice (v0.2)

The 40 wake candidates kept from the user's first real session (2026-10-19) told a different
story from the synthetic test:

- **31 were noise** (keyboard, room, no speech at all) that the detector scored ≥ 0.3. Now a
  candidate needs ≥ 0.2 s of speech (Silero VAD over the last 2 s) before anything else runs.
- **9 were the user saying "Tammy"**, which English Whisper wrote as "Demi.", "Dummy.", "Can
  we?", "Damn it!" and refused. The Bangla model wrote them as ডেমি, টেমি, তামি, ড্যামি, কেমি, so
  the double-check now asks the Bangla model first, and accepts those spellings.
- The shared detector scored most of the user's "Tammy"s low (0.02–0.5).

`sohayok train-wake` retrains the detector with the user's own clips mixed into every batch
(their "Tammy"s, augmented 40×, and the noises that fooled it), and calibrates its two
thresholds on the hour of unseen Bangla speech (check: ≤ 10 detector hits per hour; instant:
≤ 1 per hour). Three-fold cross-validation over the user's clips, each fold tested on clips
it was not trained on (`bench/wake_user.py`):

| Detector | User's "Tammy"s that wake (9) | … instantly | Noises that wake (31) |
| --- | ---: | ---: | ---: |
| Shared model (synthetic voices) | 4 | 1 | 0 |
| **Personal model** | **9** | **5** | **0** |

The final personal model (all 9 clips) uses check 0.9 and instant 0.995; on unseen Bangla
speech it gives 9 detector hits per hour at the check threshold (each double-checked) and
under 1 instant one. Nine clips is a small sample: recording more with `--record 10` and
retraining is the next step.

The three misses were Whisper hearing "Hey Tammy" as "Thank you." or "Page camming." and
"ট্যামি" as "Can you?". Synthetic voices are cleaner than a real room, so expect somewhat lower
detection with a real microphone; lower `wake_threshold` if it misses you, raise it if it
wakes by itself.

## Live checks

- **Notepad dictation, for real**: "নোটপ্যাড খুলে লেখো: আমি ভালো আছি, তুমি কেমন আছো?" opened
  a blank Notepad and the text, read back through UI Automation, matched exactly. (A first
  attempt typed into a file that Notepad had restored from its last session, before Notepad
  had finished loading. That led to waiting for the window, pasting Bangla through the
  clipboard, never overwriting a selection, and opening a blank Notepad tab.)
- **Chrome and the desktop window**: Bangla questions answered with live weather and system
  status; a restart request showed "কম্পিউটার রিস্টার্ট করব? হ্যাঁ বা না বলুন।", "না"
  cancelled it, and the reply said it was not done.
- **Wake word**: "hey jarvis" scored 0.98–0.999 with five synthetic voices, including two
  Bangladeshi Bangla voices; other phrases stayed below 0.03.
- **Read-only desktop tests** (`SOHAYOK_DESKTOP_TESTS=1 pytest -m desktop`): Start-menu index
  (142 apps), windows, screenshot, UI Automation, volume, system status, microphone.
- The microphone itself was checked for input, but no human speech was recorded for these
  benchmarks.

## Reproduce

```powershell
# data: FLEURS test parquet (refs/convert/parquet, revision 70bb2e84) and MASSIVE test files
uv run python -m sohayok.bench.stt --data <fleurs> --output out/stt `
  --model deepdml/faster-whisper-large-v3-turbo-ct2 --model Systran/faster-whisper-large-v3
uv run python -m sohayok.bench.stt --data <fleurs> --output out/stt --en 0 `
  --model "$env:APPDATA\Sohayok\models\BuzzASR-bengali-ct2"
uv run python -m sohayok.bench.brain --massive <massive> --output out/brain --model qwen3.5:4b
uv run python -m sohayok.bench.voice --massive <massive> --output out/voice
uv run python -m sohayok.bench.report out
```

Every run writes per-item JSONL (reference, transcript, tool calls, reply) next to its summary.
