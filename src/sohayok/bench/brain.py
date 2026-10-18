"""Benchmark command understanding: does the assistant pick the right tool, in the right language?

Two sources:
- MASSIVE (Amazon, CC BY 4.0) test utterances in Bangla (bn-BD) and English (en-US), for the
  intents that map onto Sohayok's tools. Real, human-written assistant requests.
- desktop_commands.jsonl: hand-written desktop requests (apps, windows, files, GitHub, power)
  that MASSIVE does not cover. Small and written by the developer; read it as a smoke test.

Tools never run: every tool is replaced by a stub that records the call and returns a
plausible result, and confirmations are answered "yes".
"""

from __future__ import annotations

import gzip
import json
import random
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

from sohayok.lang import bangla_share
from sohayok.router import route
from sohayok.safety import ConfirmationBroker, Executor
from sohayok.tools import load_all
from sohayok.tools.registry import Registry, Tool

MASSIVE_TOOLS: dict[str, set[str]] = {
    "alarm_set": {"set_reminder"},
    "alarm_query": {"list_reminders"},
    "alarm_remove": {"cancel_reminder", "list_reminders"},
    "audio_volume_up": {"volume"},
    "audio_volume_down": {"volume"},
    "audio_volume_mute": {"volume"},
    "audio_volume_other": {"volume"},
    "play_music": {"play_youtube", "media"},
    "play_radio": {"play_youtube", "open_website", "web_search"},
    "play_podcasts": {"play_youtube", "web_search"},
    "play_audiobook": {"play_youtube", "web_search"},
    "datetime_query": {"current_time", "web_lookup"},
    "weather_query": {"weather"},
    "qa_factoid": {"wikipedia_summary", "web_lookup"},
    "qa_definition": {"wikipedia_summary", "web_lookup"},
    "news_query": {"web_lookup", "web_search", "open_website"},
    "email_query": {"open_website"},
    "general_joke": {""},
}
BANGLISH = {
    "koro",
    "komao",
    "barao",
    "thamao",
    "bondho",
    "khule",
    "kholo",
    "dao",
    "gaan",
    "ta",
    "ektu",
    "chalao",
}

STUB_RESULTS = {
    "volume": "volume 40% → 50%",
    "weather": {
        "place": "Dhaka, Bangladesh",
        "now": "31°C, partly cloudy, humidity 70%",
        "today": "26–33°C, rain chance 40%",
    },
    "current_time": "10:30 AM, Saturday 18 October 2026",
    "system_status": {
        "cpu": "12% of 16 threads",
        "memory": "18.2 of 62.0 GB used (29%)",
        "gpu": "NVIDIA GeForce RTX 4060: 20% busy, 3000/8188 MiB memory, 51°C",
    },
    "list_windows": [
        {"title": "Inbox - Gmail - Google Chrome", "program": "chrome.exe"},
        {"title": "sohayok - Visual Studio Code", "program": "Code.exe"},
    ],
    "find_files": [{"path": "C:\\Users\\me\\Documents\\resume.pdf", "kind": "file"}],
    "wikipedia_summary": "Rabindranath Tagore: Bengali poet, writer, and Nobel laureate (1861–1941).",
    "web_lookup": [
        {
            "title": "Result",
            "snippet": "A short snippet answering the question.",
            "url": "https://example.org",
        }
    ],
    "github_overview": {
        "unread_notifications": [],
        "my_open_prs": ["shamspias/laya-bangla: Fine-tune Laya"],
        "review_requests": [],
        "assigned_issues": [],
    },
    "github_repos": ["Ashfaq1306/laya-bangla: Bangla Laya fine-tune"],
    "get_clipboard": "meeting at 3 pm",
    "list_reminders": [{"id": "a1b2c3d4", "at": "Sat 07:00:00", "text": "wake up"}],
    "describe_screen": [
        {"window": "Untitled - Notepad", "type": "Window"},
        {"name": "Save", "type": "Button", "x": 100, "y": 20},
    ],
    "play_youtube": "playing 'song' on YouTube",
}


@dataclass
class Case:
    text: str
    language: str
    expect: list[set[str]]
    source: str
    label: str
    args: dict[str, dict[str, str]] = field(default_factory=dict)

    @property
    def variety(self) -> str:
        if self.language == "bn":
            return "bangla"
        return "banglish" if set(self.text.lower().split()) & BANGLISH else "english"


def stub_registry(base: Registry | None = None) -> tuple[Registry, list[dict[str, Any]]]:
    base = base or load_all()
    calls: list[dict[str, Any]] = []
    registry = Registry()
    for tool in base.tools.values():

        def stub(_name=tool.name, **kwargs):
            calls.append({"name": _name, "args": kwargs})
            return STUB_RESULTS.get(_name, f"ok ({_name} done)")

        registry.add(
            Tool(tool.name, tool.description, tool.parameters, stub, tool.risk, tool.summary)
        )
    return registry, calls


def massive_cases(data_dir: Path, per_intent: int, seed: int = 20261018) -> list[Case]:
    rows: dict[str, dict[str, dict]] = {}
    for language in ("bn", "en"):
        with gzip.open(data_dir / f"test.{language}.json.gz", "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                rows.setdefault(row["id"], {})[language] = row
    by_intent: dict[str, list[str]] = defaultdict(list)
    for row_id, pair in rows.items():
        if len(pair) == 2 and pair["en"]["label"] in MASSIVE_TOOLS:
            by_intent[pair["en"]["label"]].append(row_id)
    rng = random.Random(seed)
    cases = []
    for intent in sorted(by_intent):
        ids = sorted(by_intent[intent], key=int)
        rng.shuffle(ids)
        for row_id in ids[:per_intent]:
            for language in ("bn", "en"):
                cases.append(
                    Case(
                        rows[row_id][language]["text"],
                        language,
                        [MASSIVE_TOOLS[intent]],
                        "massive",
                        intent,
                    )
                )
    return cases


def desktop_cases() -> list[Case]:
    text = resources.files("sohayok.bench").joinpath("desktop_commands.jsonl").read_text("utf-8")
    cases = []
    for line in text.splitlines():
        if line.strip():
            raw = json.loads(line)
            cases.append(
                Case(
                    raw["text"],
                    raw["lang"],
                    [set(s) for s in raw["expect"]],
                    "desktop",
                    raw["expect"][0][0] or "chat",
                    raw.get("args", {}),
                )
            )
    return cases


def judge(case: Case, called: list[dict[str, Any]]) -> tuple[bool, bool]:
    """(tool sequence correct, arguments correct)."""
    names = [c["name"] for c in called]
    if case.expect == [{""}]:
        return not names, True
    position = 0
    for step in case.expect:
        while position < len(names) and names[position] not in step:
            position += 1
        if position == len(names):
            return False, False
        position += 1
    for tool, wanted in case.args.items():
        matching = [c["args"] for c in called if c["name"] == tool]
        if not matching:
            return True, False
        for key, value in wanted.items():
            if not any(value.lower() in str(args.get(key, "")).lower() for args in matching):
                return True, False
    return True, True


def reply_language_ok(case: Case, reply: str) -> bool:
    share = bangla_share(reply)
    return share >= 0.5 if case.language == "bn" else share <= 0.2


def evaluate(brain_factory, cases: list[Case], use_router: bool) -> list[dict[str, Any]]:
    registry, calls = stub_registry()
    broker = ConfirmationBroker()
    executor = Executor(registry, broker, confirm=False)
    executor.ask_user = lambda question: True
    brain = brain_factory() if brain_factory else None
    results = []
    for case in cases:
        calls.clear()
        executor.language = case.language
        started = time.perf_counter()
        reply, fast = "", False
        intent = route(case.text) if use_router else None
        if intent is not None:
            fast = True
            executor.run(intent.tool, intent.args)
            reply = intent.reply.get(case.language, "")
        elif brain is not None:
            try:
                reply = brain.respond(case.text, case.language, executor, [], {"window": ""}).text
            except Exception as exc:
                reply = f"<error {exc}>"
        seconds = time.perf_counter() - started
        tool_ok, args_ok = judge(case, list(calls))
        results.append(
            {
                "source": case.source,
                "label": case.label,
                "language": case.language,
                "variety": case.variety,
                "text": case.text,
                "calls": list(calls),
                "reply": reply,
                "tool_ok": tool_ok,
                "args_ok": tool_ok and args_ok,
                "language_ok": reply_language_ok(case, reply) if reply else False,
                "fast": fast,
                "seconds": round(seconds, 3),
            }
        )
    return results


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    def block(rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not rows:
            return {}
        seconds = sorted(r["seconds"] for r in rows)
        return {
            "n": len(rows),
            "tool_accuracy": round(sum(r["tool_ok"] for r in rows) / len(rows), 3),
            "args_accuracy": round(sum(r["args_ok"] for r in rows) / len(rows), 3),
            "reply_language": round(sum(r["language_ok"] for r in rows) / len(rows), 3),
            "fast_path": round(sum(r["fast"] for r in rows) / len(rows), 3),
            "median_seconds": round(statistics.median(seconds), 2),
            "p90_seconds": round(seconds[int(0.9 * (len(seconds) - 1))], 2),
        }

    out: dict[str, Any] = {"all": block(results)}
    for key in ("source", "language", "variety"):
        for value in sorted({r[key] for r in results}):
            out[f"{key}={value}"] = block([r for r in results if r[key] == value])
    return out


def main(argv: list[str] | None = None) -> None:
    import argparse

    from sohayok.brain.ollama import OllamaBrain

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--massive", type=Path, required=True, help="folder with test.{bn,en}.json.gz"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", action="append", default=[], help="Ollama model (repeatable)")
    parser.add_argument("--per-intent", type=int, default=12)
    parser.add_argument("--no-router", action="store_true", help="score the model alone")
    parser.add_argument("--router-only", action="store_true")
    parser.add_argument(
        "--lite", action="append", default=[], help="small model for the Laya pipeline (repeatable)"
    )
    parser.add_argument("--decider", default="auto", help="Laya checkpoint for --lite")
    parser.add_argument("--agent", default="qwen3.5:4b", help="fallback agent for --lite")
    args = parser.parse_args(argv)

    cases = massive_cases(args.massive, args.per_intent) + desktop_cases()
    args.output.mkdir(parents=True, exist_ok=True)
    runs = [("router", None)] if args.router_only else []
    runs += [(m, (lambda m=m: OllamaBrain(m, max_steps=6))) for m in args.model]
    if args.lite:
        from sohayok.brain.lite import LiteBrain
        from sohayok.decide import LayaDecider, resolve_decider

        decider = LayaDecider(*resolve_decider(args.decider))
        tag = "laya-ft" if "sohayok" in resolve_decider(args.decider)[0] else "laya"
        runs += [
            (
                f"{tag}+{m}",
                (lambda m=m: LiteBrain(decider, m, fallback=OllamaBrain(args.agent, max_steps=6))),
            )
            for m in args.lite
        ]
    summary = {}
    for name, factory in runs:
        results = evaluate(factory, cases, use_router=not args.no_router)
        safe = name.replace(":", "_").replace("/", "_")
        suffix = "" if not args.no_router else ".model-only"
        with (args.output / f"{safe}{suffix}.jsonl").open("w", encoding="utf-8") as handle:
            for row in results:
                handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        summary[name] = summarize(results)
        print(name, json.dumps(summary[name]["all"]), flush=True)
    path = args.output / f"summary{'.model-only' if args.no_router else ''}.json"
    previous = json.loads(path.read_text("utf-8")) if path.exists() else {}
    previous.update(summary)
    path.write_text(json.dumps(previous, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
