"""Ask the user before risky actions, by voice or by clicking in the window."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from sohayok.lang import normalize, strip_punctuation
from sohayok.tools.registry import Registry, Risk

YES = {
    "yes",
    "yeah",
    "yep",
    "sure",
    "ok",
    "okay",
    "do it",
    "go ahead",
    "confirm",
    "haan",
    "ha",
    "hya",
    "hyan",
    "হ্যাঁ",
    "হ্যা",
    "হাঁ",
    "হা",
    "জি",
    "জী",
    "জি হ্যাঁ",
    "ঠিক আছে",
    "করো",
    "কর",
    "করুন",
    "ওকে",
    "অবশ্যই",
    "চালাও",
    "হ্যাঁ করো",
    "ইয়েস",
}
NO = {
    "no",
    "nope",
    "cancel",
    "stop",
    "don't",
    "do not",
    "na",
    "naa",
    "না",
    "নাহ",
    "না না",
    "বাতিল",
    "থামো",
    "করো না",
    "করবে না",
    "দরকার নেই",
    "নো",
    "ক্যানসেল",
}


def parse_answer(text: str) -> bool | None:
    """True for yes, False for no, None if the reply is neither."""
    words = strip_punctuation(normalize(text).replace("'", "")).split()
    phrase = " ".join(words)
    if not phrase:
        return None
    if phrase in NO or any(w in {"না", "no", "cancel", "বাতিল", "nope"} for w in words):
        return False
    if phrase in YES or (words and words[0] in YES) or phrase.startswith(("হ্যাঁ", "yes")):
        return True
    return None


@dataclass
class Pending:
    question: str
    language: str
    done: threading.Event = field(default_factory=threading.Event)
    answer: bool = False


class ConfirmationBroker:
    """One question at a time; the UI or the voice listener calls `resolve`."""

    def __init__(self, timeout: float = 30.0) -> None:
        self.timeout = timeout
        self.pending: Pending | None = None
        self._lock = threading.Lock()
        self.on_ask: Callable[[Pending], None] = lambda p: None
        self.on_done: Callable[[Pending], None] = lambda p: None

    def ask(self, question: str, language: str) -> bool:
        with self._lock:
            pending = self.pending = Pending(question, language)
        self.on_ask(pending)
        pending.done.wait(self.timeout)
        with self._lock:
            self.pending = None
        self.on_done(pending)
        return pending.answer

    def resolve(self, answer: bool) -> bool:
        with self._lock:
            pending = self.pending
        if pending is None or pending.done.is_set():
            return False
        pending.answer = answer
        pending.done.set()
        return True


# Tools that act on "the active window"; they must not land in Sohayok's own window.
FOCUS_TOOLS = {
    "type_text",
    "press_keys",
    "window_action",
    "snap_window",
    "describe_screen",
    "click_element",
    "click",
    "scroll",
    "computer",
}


class Executor:
    """Runs tools for the brain, confirming CONFIRM-level tools with the user first."""

    def __init__(
        self,
        registry: Registry,
        broker: ConfirmationBroker,
        confirm: bool = True,
        on_tool: Callable[[str, dict[str, Any], str], None] = lambda n, a, r: None,
    ) -> None:
        self.registry = registry
        self.broker = broker
        self.confirm = confirm
        self.on_tool = on_tool
        self.language = "en"
        # Set by the desktop app: refocus the user's last app if Sohayok's window is in front.
        self.focus_target: Callable[[], None] = lambda: None

    def prepare(self, name: str, args: dict[str, Any] | None = None) -> None:
        if name in FOCUS_TOOLS and not (name == "window_action" and (args or {}).get("query")):
            self.focus_target()

    def run(self, name: str, args: dict[str, Any]) -> str:
        tool = self.registry.get(name)
        if tool is not None and tool.risk is Risk.CONFIRM and self.confirm:
            if not self.broker.ask(tool.describe(args, self.language), self.language):
                result = (
                    "cancelled: the user said no (or did not answer), so this action was NOT "
                    "performed. Tell the user it was not done. Do not retry."
                )
                self.on_tool(name, args, result)
                return result
        self.prepare(name, args)
        result = self.registry.call(name, args)
        self.on_tool(name, args, result)
        return result

    def ask_user(self, question: str) -> bool:
        return self.broker.ask(question, self.language)
