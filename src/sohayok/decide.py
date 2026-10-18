"""Fast action decisions with Laya-Bangla: which kind of action does the user want?

Laya is a typed-decision model: given the request and a list of options with short
descriptions, it returns a probability for each option in about 15 ms on a GPU. Sohayok asks
it one question, "which action?", and only then extracts the details (which app, what to
search) for that single action.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

LAYA_REPO = "algolyzer/laya-bangla"
LAYA_REVISION = "caeb5631ddfe0efe0f2937aeeae06f8d2d1cf6da"
# Sohayok's fine-tune of Laya-Bangla for its 28 desktop actions (91.8% vs 63.4% zero-shot).
DECIDER_REPO = "ashfaq1306/sohayok-laya-actions"
DECIDER_REVISION = "3fc68f6bccc0ca992890b63cb87c76efe125cd58"


@dataclass(frozen=True)
class Action:
    bn: str
    en: str
    tools: tuple[str, ...]  # tools this action may use; the first is the usual one


ACTIONS: dict[str, Action] = {
    "open_app": Action(
        "কোনো অ্যাপ বা প্রোগ্রাম খোলা বা চালু করা", "Open or launch an app or program", ("open_app",)
    ),
    "close_app": Action("কোনো অ্যাপ বা প্রোগ্রাম বন্ধ করা", "Close or quit an app", ("close_app",)),
    "website": Action(
        "কোনো ওয়েবসাইট খোলা, যেমন জিমেইল বা ফেসবুক",
        "Open a website such as Gmail or Facebook",
        ("open_website",),
    ),
    "search": Action(
        "ইন্টারনেটে বা গুগলে কিছু খোঁজা", "Search the web or Google for something", ("web_search",)
    ),
    "play": Action(
        "গান, ভিডিও, রেডিও বা পডকাস্ট চালানো",
        "Play music, a video, radio, or a podcast",
        ("play_youtube",),
    ),
    "media": Action(
        "চলমান গান বা ভিডিও থামানো, আবার চালানো বা পরেরটা",
        "Pause, resume, or skip what is playing",
        ("media",),
    ),
    "volume": Action(
        "শব্দ বা ভলিউম বাড়ানো, কমানো, মিউট করা", "Change, raise, lower, or mute volume", ("volume",)
    ),
    "time": Action("এখন সময় বা আজকের তারিখ জানা", "Ask the time or today's date", ("current_time",)),
    "weather": Action("আবহাওয়া, তাপমাত্রা বা বৃষ্টি", "Weather, temperature, or rain", ("weather",)),
    "reminder": Action(
        "অ্যালার্ম, টাইমার বা রিমাইন্ডার সেট, দেখা বা বাতিল",
        "Set, check, or cancel an alarm, timer, or reminder",
        ("set_reminder", "list_reminders", "cancel_reminder"),
    ),
    "note": Action(
        "কোনো কথা নোট করে রাখা বা লিস্টে যোগ করা", "Save a note or add to a list", ("take_note",)
    ),
    "type": Action("কোনো লেখা টাইপ করা বা লিখে দেওয়া", "Type or write some text", ("type_text",)),
    "keys": Action(
        "কিবোর্ডের কোনো বোতাম বা শর্টকাট চাপা, নতুন ট্যাব",
        "Press a key or keyboard shortcut, new tab",
        ("press_keys",),
    ),
    "window": Action(
        "উইন্ডো ছোট, বড়, বন্ধ, বদলানো বা ডেস্কটপ দেখানো",
        "Minimize, maximize, close, or switch windows, show desktop",
        (
            "window_action",
            "switch_to_window",
            "show_desktop",
            "snap_window",
            "list_windows",
            "task_view",
        ),
    ),
    "files": Action(
        "ফাইল বা ফোল্ডার খোঁজা, খোলা, বানানো, সরানো বা মোছা",
        "Find, open, create, move, or delete files or folders",
        (
            "open_path",
            "find_files",
            "create_folder",
            "recycle",
            "move_path",
            "read_text_file",
            "write_text_file",
        ),
    ),
    "screenshot": Action("স্ক্রিনশট নেওয়া", "Take a screenshot", ("take_screenshot",)),
    "screen": Action(
        "স্ক্রিনে কী আছে দেখা বা কোনো বাটনে ক্লিক করা",
        "See what is on screen or click a button",
        ("describe_screen", "click_element"),
    ),
    "settings": Action(
        "উইন্ডোজ সেটিংস খোলা, যেমন ওয়াইফাই বা ব্লুটুথ",
        "Open Windows settings such as Wi-Fi or Bluetooth",
        ("open_settings",),
    ),
    "system": Action(
        "কম্পিউটারের সিপিইউ, র‍্যাম, জিপিইউ বা ব্যাটারির অবস্থা",
        "Computer status: CPU, RAM, GPU, or battery",
        ("system_status",),
    ),
    "lock": Action("কম্পিউটার লক করা", "Lock the computer", ("lock_screen",)),
    "power": Action(
        "কম্পিউটার বন্ধ, রিস্টার্ট, স্লিপ বা সাইন আউট",
        "Shut down, restart, sleep, or sign out",
        ("power", "cancel_shutdown"),
    ),
    "clipboard": Action(
        "ক্লিপবোর্ডে কপি করা বা ক্লিপবোর্ড পড়া",
        "Copy to or read the clipboard",
        ("set_clipboard", "get_clipboard"),
    ),
    "github": Action(
        "গিটহাবের রিপোজিটরি, নোটিফিকেশন বা পুল রিকোয়েস্ট",
        "GitHub repositories, notifications, or pull requests",
        ("github_overview", "github_repos"),
    ),
    "question": Action(
        "কোনো তথ্য, সংজ্ঞা, খবর বা মানুষ সম্পর্কে প্রশ্ন",
        "A question about facts, a definition, news, or a person",
        ("wikipedia_summary", "web_lookup"),
    ),
    "command": Action(
        "কোনো সফটওয়্যার ইনস্টল বা কমান্ড চালানো", "Install software or run a command", ("run_powershell",)
    ),
    "chat": Action("শুভেচ্ছা, কৌতুক, ধন্যবাদ বা সাধারণ গল্প", "Greetings, jokes, thanks, or small talk", ()),
    "multi": Action(
        "একসাথে একাধিক কাজ, যেমন অ্যাপ খুলে কিছু লেখা",
        "Several steps at once, like opening an app and typing",
        (),
    ),
    "other": Action("অন্য কোনো কাজ যা এখানে নেই", "Something else not listed here", ()),
}
TOOL_ACTION = {tool: name for name, action in ACTIONS.items() for tool in action.tools}

QUESTION = {
    "bn": "ব্যবহারকারী কম্পিউটারকে কোন কাজটি করতে বলছেন?",
    "en": "Which action is the user asking the computer to do?",
}


def resolve_decider(setting: str) -> tuple[str, str | None]:
    """(checkpoint, revision). "auto": a locally trained decider if there is one (setup
    --train-decider), otherwise the published fine-tune, pinned to its revision."""
    if setting != "auto":
        return setting, None
    from sohayok.config import app_dir

    local = app_dir() / "models" / "laya-sohayok-actions"
    if (local / "model.safetensors").exists():
        return str(local), None
    return DECIDER_REPO, DECIDER_REVISION


@dataclass
class Decision:
    action: str
    probability: float
    probabilities: dict[str, float] = field(default_factory=dict)

    @property
    def runner_up(self) -> tuple[str, float]:
        ranked = sorted(self.probabilities.items(), key=lambda kv: -kv[1])
        return ranked[1] if len(ranked) > 1 else ("", 0.0)


def questions(language: str = "bn") -> dict:
    wording = "bn" if language == "bn" else "en"
    return {
        "action": {
            "type": "choice",
            "instructions": QUESTION[wording],
            "criteria": {name: getattr(action, wording) for name, action in ACTIONS.items()},
        }
    }


class LayaDecider:
    def __init__(
        self, model: str = LAYA_REPO, revision: str | None = LAYA_REVISION, device: str = "cuda"
    ) -> None:
        import laya

        self.agent = laya.load(model, device=device, revision=revision)
        self._lock = threading.Lock()  # the model is not safe to call from two threads

    def decide(self, text: str, language: str = "bn") -> Decision:
        with self._lock:
            result = self.agent.predict(text, questions(language))
        answer = result["answers"]["action"]
        return Decision(
            answer["choice"],
            float(answer["answer_confidence"]),
            {k: float(v) for k, v in answer["probabilities"].items()},
        )

    def decide_batch(self, texts: list[str], language: str = "bn") -> list[Decision]:
        with self._lock:
            results = self.agent.predict_batch(texts, questions(language), batch_size=32)
        out = []
        for result in results:
            answer = result["answers"]["action"]
            out.append(
                Decision(
                    answer["choice"],
                    float(answer["answer_confidence"]),
                    {k: float(v) for k, v in answer["probabilities"].items()},
                )
            )
        return out
