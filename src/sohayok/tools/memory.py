"""Things the user asks Sohayok to remember ("my office starts at nine"), kept across restarts.

Saved in %APPDATA%\\Sohayok\\memory.json and given to the answer model with every request.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from sohayok.tools.registry import ToolError, string, tool

LIMIT = 100  # the oldest facts are dropped beyond this


def memory_file() -> Path:
    from sohayok.config import app_dir

    return app_dir() / "memory.json"


def facts() -> list[dict]:
    try:
        return json.loads(memory_file().read_text("utf-8"))
    except (OSError, ValueError):
        return []


def _save(items: list[dict]) -> None:
    path = memory_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(items[-LIMIT:], ensure_ascii=False, indent=1), "utf-8")


def context() -> str:
    """What the answer model should know about the user ("" when nothing was remembered)."""
    items = facts()
    if not items:
        return ""
    return "Things the user asked you to remember: " + "; ".join(i["fact"] for i in items)


@tool(
    "Remember a fact or preference the user tells you to remember (their schedule, names, "
    "likes), for all later conversations. Use for 'মনে রাখো ...', 'remember that ...'.",
    {"fact": string("The fact, in the user's own words")},
)
def remember(fact: str) -> dict:
    fact = fact.strip()
    if not fact:
        raise ToolError("nothing to remember")
    items = [i for i in facts() if i["fact"] != fact]
    items.append({"fact": fact, "date": date.today().isoformat()})
    _save(items)
    return {"remembered": fact, "spoken_bn": "মনে রাখলাম।", "spoken_en": "Got it, I'll remember."}


@tool(
    "Forget remembered facts that mention something ('ভুলে যাও ...', 'forget ...'), or all "
    "of them with 'all'.",
    {"about": string("A word or phrase from the fact to forget, or 'all'")},
)
def forget(about: str) -> dict:
    about = about.strip().lower()
    items = facts()
    kept = [] if about in {"all", "সব"} else [i for i in items if about not in i["fact"].lower()]
    if len(kept) == len(items):
        return {
            "forgotten": 0,
            "spoken_bn": "এমন কিছু মনে রাখা ছিল না।",
            "spoken_en": "I had nothing like that remembered.",
        }
    _save(kept)
    return {
        "forgotten": len(items) - len(kept),
        "spoken_bn": "ভুলে গেলাম।",
        "spoken_en": "Forgotten.",
    }
