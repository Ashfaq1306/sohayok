"""Instant answers for everyday commands, in Bangla, Banglish, and English.

The fast path only claims a command when the whole sentence matches a known shape, so
anything ambiguous still goes to the language model.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from sohayok.lang import normalize, parse_number


@dataclass
class Intent:
    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    # Spoken confirmation, keyed by language; "{result}" is replaced with the tool result.
    reply: dict[str, str] = field(default_factory=dict)
    # Further (tool, args) steps run after the first one, e.g. typing after opening an app.
    then: list[tuple[str, dict[str, Any]]] = field(default_factory=list)


OPEN = r"(?:খোলো|খুলো|খুলুন|খোল|ওপেন করো|ওপেন করুন|ওপেন কর|চালু করো|চালু করুন|kholo|khulo|khol|open koro|open kor|chalu koro)"
CLOSE = r"(?:বন্ধ করো|বন্ধ করুন|বন্ধ কর|ক্লোজ করো|ক্লোজ করুন|কেটে দাও|bondho koro|bondho kor|close koro)"
PLEASE = r"(?:\s*(?:প্লিজ|please|plz|তো|একটু|ektu))*"
VOLUME = r"(?:ভলিউম|সাউন্ড|আওয়াজ|শব্দ|volume|sound|awaj)"
UP = r"(?:বাড়াও|বাড়িয়ে দাও|বাড়ান|বেশি করো|barao|bariye dao|up|increase|louder)"
DOWN = r"(?:কমাও|কমিয়ে দাও|কমান|কম করো|komao|komiye dao|down|decrease|lower|quieter)"
SET = r"(?:করো|করুন|কর|দাও|koro|kor|dao|set|to|at)"
FILLER = r"(?:\s+(?:ektu|একটু|আরেকটু|আরো|aro|ta|টা|টাকে))?"


def _rx(pattern: str) -> re.Pattern[str]:
    # Text is NFC-normalized before matching, so the patterns must be too.
    return re.compile(unicodedata.normalize("NFC", rf"^{pattern}{PLEASE}[.!?।]*$"))


RULES: list[tuple[re.Pattern[str], Any]] = []


def rule(*patterns: str):
    def decorator(build):
        for pattern in patterns:
            RULES.append((_rx(pattern), build))
        return build

    return decorator


@rule(
    rf"{VOLUME}{FILLER}\s*{UP}",
    rf"(?:turn (?:the )?)?{VOLUME} up",
    rf"(?:increase|raise) (?:the )?{VOLUME}",
)
def _volume_up(m, text):
    return Intent(
        "volume", {"action": "up", "level": 10}, {"bn": "ভলিউম বাড়ালাম।", "en": "Volume up."}
    )


@rule(
    rf"{VOLUME}{FILLER}\s*{DOWN}",
    rf"(?:turn (?:the )?)?{VOLUME} down",
    rf"(?:decrease|lower) (?:the )?{VOLUME}",
)
def _volume_down(m, text):
    return Intent(
        "volume", {"action": "down", "level": 10}, {"bn": "ভলিউম কমালাম।", "en": "Volume down."}
    )


@rule(
    rf"{VOLUME}\s*(?P<n>\S+(?: \S+)?)\s*(?:%|শতাংশ|পার্সেন্ট|percent)?\s*{SET}",
    rf"(?:set )?(?:the )?{VOLUME}\s*(?:to|at)?\s*(?P<n>\S+(?: \S+)?)\s*(?:%|percent)?",
)
def _volume_set(m, text):
    level = parse_number(m.group("n"))
    if level is None or not 0 <= level <= 100:
        return None
    return Intent(
        "volume",
        {"action": "set", "level": level},
        {"bn": "ভলিউম {n} করলাম।".replace("{n}", str(level)), "en": f"Volume set to {level}."},
    )


@rule(r"(?:mute|মিউট করো|মিউট করুন|মিউট কর|মিউট|sound off|সাউন্ড বন্ধ করো|আওয়াজ বন্ধ করো|mute koro)")
def _mute(m, text):
    return Intent("volume", {"action": "mute"}, {"bn": "মিউট করলাম।", "en": "Muted."})


@rule(r"(?:unmute|আনমিউট করো|আনমিউট করুন|আনমিউট|সাউন্ড চালু করো|আওয়াজ চালু করো|unmute koro)")
def _unmute(m, text):
    return Intent("volume", {"action": "unmute"}, {"bn": "সাউন্ড চালু করলাম।", "en": "Unmuted."})


@rule(
    r"(?:pause|resume|play|pause (?:the )?(?:music|song|video)|play (?:the )?(?:music|song|video)|"
    r"(?:গান|গানটা|গানটি|ভিডিও|ভিডিওটা|মিউজিক)\s*(?:থামাও|পজ করো|বন্ধ করো|আবার চালাও)|"
    r"থামাও|পজ করো|পজ|আবার চালাও|plz pause|pause koro|"
    r"(?:gaan|gan|video|music)\s*(?:ta|ti)?\s*(?:thamao|pause koro|bondho koro)|thamao)"
)
def _play_pause(m, text):
    return Intent("media", {"action": "playpause"}, {"bn": "ঠিক আছে।", "en": "Okay."})


@rule(
    r"(?:next|next (?:song|track|video)|skip|পরের (?:গান|গানটা|গানটি|ভিডিও|ভিডিওটা)|পরেরটা|"
    r"নেক্সট|নেক্সট গান|porer (?:gaan|gan)(?: ta)?|next (?:gaan|gan))"
    r"(?: চালাও| দাও| বাজাও| koro| dao| chalao)?"
)
def _next(m, text):
    return Intent("media", {"action": "next"}, {"bn": "পরেরটা চালাচ্ছি।", "en": "Next."})


@rule(r"(?:previous|previous (?:song|track)|আগের গান|আগেরটা|আগের ভিডিও|ager gaan)(?: চালাও| দাও)?")
def _previous(m, text):
    return Intent("media", {"action": "previous"}, {"bn": "আগেরটা চালাচ্ছি।", "en": "Previous."})


@rule(
    r"(?:what(?:'s| is) the time|what time is it|time|tell me the time|"
    r"কয়টা বাজে|কটা বাজে|কয়েকটা বাজে|এখন কয়টা বাজে|এখন কটা বাজে|এখন কয়েকটা বাজে|সময় কত|এখন সময় কত|ক'টা বাজে|"
    r"koyta baje|ekhon koyta baje|somoy koto)"
)
def _time(m, text):
    return Intent("current_time", {}, {"bn": "এখন {result}।", "en": "It's {result}."})


@rule(
    r"(?:what(?:'s| is) (?:the )?date(?: today)?|what day is it|today's date|"
    r"আজ কত তারিখ|আজকে কত তারিখ|আজ কী বার|আজ কি বার|আজকের তারিখ|aj koto tarikh)"
)
def _date(m, text):
    return Intent("current_time", {}, {"bn": "{result}।", "en": "{result}."})


@rule(r"(?:take a screenshot|screenshot|স্ক্রিনশট নাও|স্ক্রিনশট নিন|স্ক্রিনশট|screenshot nao)")
def _screenshot(m, text):
    return Intent("take_screenshot", {}, {"bn": "স্ক্রিনশট নিয়েছি।", "en": "Screenshot saved."})


@rule(
    r"(?:lock (?:the |my )?(?:computer|screen|pc|laptop)|lock|কম্পিউটার লক করো|লক করো|"
    r"স্ক্রিন লক করো|পিসি লক করো|lock koro)"
)
def _lock(m, text):
    return Intent("lock_screen", {}, {"bn": "লক করছি।", "en": "Locking."})


@rule(r"(?:show (?:the )?desktop|minimi[sz]e (?:all|everything)|ডেস্কটপ দেখাও|সব মিনিমাইজ করো)")
def _desktop(m, text):
    return Intent("show_desktop", {}, {"bn": "ডেস্কটপ দেখাচ্ছি।", "en": "Done."})


@rule(r"(?:close (?:this|the) (?:window|tab|app)|এটা বন্ধ করো|উইন্ডো বন্ধ করো|এই উইন্ডো বন্ধ করো)")
def _close_window(m, text):
    if "tab" in text or "ট্যাব" in text:
        return Intent("press_keys", {"keys": "ctrl+w"}, {"bn": "ট্যাব বন্ধ করলাম।", "en": "Closed."})
    return Intent("window_action", {"action": "close"}, {"bn": "বন্ধ করলাম।", "en": "Closed."})


@rule(r"(?:minimi[sz]e(?: (?:this|the) window)?|উইন্ডো মিনিমাইজ করো|মিনিমাইজ করো|ছোট করো)")
def _minimize(m, text):
    return Intent("window_action", {"action": "minimize"}, {"bn": "ঠিক আছে।", "en": "Done."})


@rule(r"(?:maximi[sz]e(?: (?:this|the) window)?|উইন্ডো ম্যাক্সিমাইজ করো|ম্যাক্সিমাইজ করো|বড় করো)")
def _maximize(m, text):
    return Intent("window_action", {"action": "maximize"}, {"bn": "ঠিক আছে।", "en": "Done."})


@rule(r"(?:new tab|open (?:a )?new tab|নতুন (?:ট্যাব|ট্যাপ)(?: খোলো)?|notun tab kholo)")
def _new_tab(m, text):
    return Intent("press_keys", {"keys": "ctrl+t"}, {"bn": "নতুন ট্যাব।", "en": "New tab."})


@rule(
    r"(?P<q>.+?)\s*(?:ইউটিউবে|ইউটিউব এ|youtube e|youtube-e)\s*(?:চালাও|বাজাও|প্লে করো|দেখাও|chalao|bajao|play koro)",
    r"play (?P<q>.+?) on youtube",
    r"(?:ইউটিউবে|youtube e)\s*(?P<q>.+?)\s*(?:চালাও|বাজাও|প্লে করো|chalao)",
)
def _youtube(m, text):
    return Intent(
        "play_youtube",
        {"query": m.group("q")},
        {"bn": "ইউটিউবে চালাচ্ছি।", "en": "Playing it on YouTube."},
    )


@rule(
    r"(?:search|google)(?: (?:for|on google for))? (?P<q>.+)",
    r"(?P<q>.+?)\s*(?:গুগলে|গুগল এ|google e)?\s*(?:সার্চ করো|সার্চ করুন|খোঁজো|খুঁজে দাও|খুঁজুন|search koro|khojo)",
)
def _search(m, text):
    query = m.group("q").strip()
    if not query or len(query) < 2 or re.search(r" and | then | তারপর | এবং ", f" {query} "):
        return None
    return Intent(
        "web_search", {"query": query, "engine": "google"}, {"bn": "খুঁজছি।", "en": "Searching."}
    )


@rule(
    r"(?P<x>\S+(?: \S+)?) খুলে (?:লেখো|লিখো|লিখে দাও|টাইপ করো)[:,]?\s+(?P<t>.+)",
    r"(?P<x>\S+(?: \S+)?) khule (?:likho|lekho|type koro)[:,]?\s+(?P<t>.+)",
    r"open (?P<x>\S+(?: \S+)?) and (?:type|write)[:,]?\s+(?P<t>.+)",
)
def _open_and_type(m, text):
    """The most common two-step request: open an app, then dictate into it."""
    from sohayok.tools.apps import resolve_app, start_apps

    try:
        app = resolve_app(m.group("x"), start_apps())
    except Exception:
        app = None
    if not app:
        return None
    words = m.group("t").strip().strip("\"'“”")
    return Intent(
        "open_app",
        {"name": app},
        {"bn": f"{app} খুলে লিখে দিলাম।", "en": f"Opened {app} and typed it."},
        then=[("type_text", {"text": words})],
    )


@rule(
    rf"(?P<x>.+?)\s*(?:সেটিংস|সেটিং|settings|setting)\s*{OPEN}",
    r"(?:open|show) (?:the )?(?P<x>.+?) settings",
)
def _settings(m, text):
    from sohayok.tools.system import SETTINGS

    page = SETTINGS_WORDS.get(normalize(m.group("x")), normalize(m.group("x")))
    if page not in SETTINGS:
        return None
    return Intent(
        "open_settings", {"page": page}, {"bn": "সেটিংস খুলছি।", "en": f"Opening {page} settings."}
    )


SETTINGS_WORDS = {
    "ওয়াইফাই": "wifi",
    "ওয়াই-ফাই": "wifi",
    "wi-fi": "wifi",
    "ব্লুটুথ": "bluetooth",
    "ডিসপ্লে": "display",
    "সাউন্ড": "sound",
    "নেটওয়ার্ক": "network",
    "আপডেট": "update",
    "windows update": "update",
    "উইন্ডোজ আপডেট": "update",
    "ব্যাটারি": "battery",
    "মাইক্রোফোন": "microphone",
    "ক্যামেরা": "camera",
    "ভাষা": "language",
    "তারিখ": "time",
    "সময়": "time",
    "প্রিন্টার": "printers",
    "ব্যাকগ্রাউন্ড": "background",
    "থিম": "themes",
}
SETTINGS_WORDS = {normalize(k): v for k, v in SETTINGS_WORDS.items()}


@rule(rf"(?P<x>.+?)\s*{OPEN}", r"(?:open|launch|start|run) (?P<x>.+)")
def _open(m, text):
    from sohayok.tools.browser import SITES

    target = m.group("x").strip()
    if _compound(target):
        return None
    target = re.sub(r"^(?:the|my|আমার)\s+", "", target)
    target = re.sub(r"\s+(?:app|application|অ্যাপ|অ্যাপটা|টা|টি|ta)$", "", target)
    key = normalize(target).removesuffix(".com")
    if key.removesuffix(" website").removesuffix(" ওয়েবসাইট") in SITES or re.search(
        r"\.\w{2,}$", target
    ):
        from sohayok.tools.browser import site_url

        try:
            url = site_url(target)
        except Exception:
            return None
        return Intent(
            "open_website", {"site": url}, {"bn": f"{target} খুলছি।", "en": f"Opening {target}."}
        )
    from sohayok.tools.files import known_folder

    folder = known_folder(target)
    if folder:
        name = re.sub(r"\s*(?:folder|ফোল্ডার)$", "", target)
        return Intent(
            "open_path",
            {"path": str(folder)},
            {"bn": f"{name} ফোল্ডার খুলছি।", "en": f"Opening {name}."},
        )
    from sohayok.tools.apps import resolve_app, start_apps

    try:
        app = resolve_app(target, start_apps())
    except Exception:
        app = None
    if not app:
        return None
    return Intent("open_app", {"name": app}, {"bn": f"{app} খুলছি।", "en": f"Opening {app}."})


@rule(rf"(?P<x>.+?)\s*{CLOSE}", r"(?:close|quit|exit) (?P<x>.+)")
def _close(m, text):
    target = m.group("x").strip()
    if _compound(target):
        return None
    if normalize(target) in {"this", "it", "এটা", "এটি", "এই উইন্ডো", "window", "this window"}:
        return Intent("window_action", {"action": "close"}, {"bn": "বন্ধ করলাম।", "en": "Closed."})
    from sohayok.tools.apps import resolve_app, start_apps

    try:
        app = resolve_app(target, start_apps())
    except Exception:
        app = None
    if not app:
        return None
    return Intent("close_app", {"name": app}, {"bn": f"{app} বন্ধ করছি।", "en": f"Closing {app}."})


WAKE_PREFIX = re.compile(
    unicodedata.normalize(
        "NFC",
        r"^(?:hey |hi |ok |okay |হেই |এই )?(?:tammy|tami|tammi|tommy|sohayok|jarvis|টামি|ট্যামি|তামি"
        r"|সহায়ক|জার্ভিস)(?![a-zঀ-৿])[\s,.!।?]*",
    ),
    re.IGNORECASE,
)


def strip_wake_name(text: str) -> str:
    """Remove a leading "Tammy," / "হেই টামি" so the rest is the command."""
    return WAKE_PREFIX.sub("", text.strip(), count=1).strip()


def _compound(target: str) -> bool:
    """True for multi-step requests ("chrome and search cats"), which need the model."""
    return len(target.split()) > 4 or bool(
        re.search(r",| and | then | এবং | তারপর | আর | তারপরে | তার পর | ar ", f" {target} ")
    )


def route(text: str) -> Intent | None:
    """Return an Intent if the whole command matches a fast-path rule."""
    cleaned = normalize(text).strip(" ,")
    cleaned = strip_wake_name(cleaned)
    cleaned = re.sub(r"^(?:please|plz|প্লিজ|একটু)\s+", "", cleaned)
    if not cleaned or len(cleaned) > 80:
        return None
    for pattern, build in RULES:
        match = pattern.match(cleaned)
        if match:
            intent = build(match, cleaned)
            if intent is not None:
                return intent
    return None
