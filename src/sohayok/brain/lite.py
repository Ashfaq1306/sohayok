"""The light pipeline: Laya decides, a small model fills in details, templates answer.

1. Laya-Bangla picks the action ("open_app", "volume", "weather", "chat", ...) in ~15 ms.
2. Actions with nothing to fill in run at once. Otherwise a small model (qwen3.5:2b by
   default) sees only that action's one to seven tools, which is far easier and faster than
   choosing among all 48.
3. Simple successes are confirmed with a built-in sentence in the user's language; results
   worth explaining (weather, a lookup, system status) are summarized by the small model.
4. Multi-step or unusual requests, and unsure decisions, go to the full agent loop.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable
from typing import Any

import httpx

from sohayok.brain.base import Reply
from sohayok.decide import ACTIONS, Decision
from sohayok.safety import Executor

log = logging.getLogger(__name__)

# Spoken confirmations for tools whose result needs no explaining. {x} fills from arguments.
DONE_REPLIES: dict[str, dict[str, str]] = {
    "open_app": {"bn": "{name} খুলছি।", "en": "Opening {name}."},
    "close_app": {"bn": "{name} বন্ধ করছি।", "en": "Closing {name}."},
    "open_website": {"bn": "খুলে দিলাম।", "en": "Opened it."},
    "web_search": {"bn": "খুঁজে দেখাচ্ছি।", "en": "Here are the results."},
    "play_youtube": {"bn": "চালাচ্ছি।", "en": "Playing it."},
    "media": {"bn": "ঠিক আছে।", "en": "Okay."},
    "set_reminder": {"bn": "ঠিক আছে, মনে করিয়ে দেব।", "en": "Okay, I'll remind you."},
    "cancel_reminder": {"bn": "বাতিল করলাম।", "en": "Cancelled."},
    "take_note": {"bn": "নোট করে রাখলাম।", "en": "Noted."},
    "type_text": {"bn": "লিখে দিলাম।", "en": "Typed it."},
    "press_keys": {"bn": "ঠিক আছে।", "en": "Done."},
    "window_action": {"bn": "ঠিক আছে।", "en": "Done."},
    "switch_to_window": {"bn": "সামনে আনলাম।", "en": "Switched."},
    "show_desktop": {"bn": "ডেস্কটপ দেখাচ্ছি।", "en": "Here's the desktop."},
    "snap_window": {"bn": "সরিয়ে দিলাম।", "en": "Done."},
    "task_view": {"bn": "সব উইন্ডো দেখাচ্ছি।", "en": "Here are your windows."},
    "open_path": {"bn": "খুলে দিলাম।", "en": "Opened it."},
    "create_folder": {"bn": "ফোল্ডার বানালাম।", "en": "Folder created."},
    "recycle": {"bn": "রিসাইকেল বিনে সরালাম।", "en": "Moved to the Recycle Bin."},
    "move_path": {"bn": "সরিয়ে দিলাম।", "en": "Moved."},
    "write_text_file": {"bn": "ফাইলে লিখে রাখলাম।", "en": "Saved."},
    "take_screenshot": {"bn": "স্ক্রিনশট নিয়েছি, ছবি ফোল্ডারে আছে।", "en": "Screenshot saved."},
    "click_element": {"bn": "ক্লিক করলাম।", "en": "Clicked."},
    "open_settings": {"bn": "সেটিংস খুলছি।", "en": "Opening settings."},
    "lock_screen": {"bn": "লক করছি।", "en": "Locking."},
    "power": {"bn": "ঠিক আছে।", "en": "Okay."},
    "cancel_shutdown": {"bn": "বন্ধ হওয়া বাতিল করলাম।", "en": "Shutdown cancelled."},
    "set_clipboard": {"bn": "কপি করে রাখলাম।", "en": "Copied."},
}
CHAT_PROMPT = (
    "You are Sohayok (সহায়ক), a friendly voice assistant on the user's PC. Answer in one or two "
    "short spoken sentences, no markdown or emoji. Reply in {language}."
)
FILL_PROMPT = (
    "You control a Windows PC. Call exactly one of the given tools with the right arguments "
    "for the user's request. Copy names, text, and numbers from the request; do not invent "
    "details. Never answer in text."
)
SUMMARY_PROMPT = (
    "You are Sohayok, a voice assistant. Using only the tool result, answer the user's request "
    "in one or two short spoken sentences in {language}. No markdown, lists, or URLs."
)
LANGUAGE = {"bn": "natural Bangla (Bangla script)", "en": "English"}


def _prepared_reply(result: str, language: str) -> str:
    """Tools such as weather return a ready sentence ("spoken_bn"/"spoken_en")."""
    if not result.startswith("{"):
        return ""
    try:
        data = json.loads(result)
    except json.JSONDecodeError:
        return ""
    return data.get(f"spoken_{language}", "") if isinstance(data, dict) else ""


class LiteBrain:
    def __init__(
        self,
        decider: Any,
        model: str = "qwen3.5:2b",
        url: str = "http://127.0.0.1:11434",
        fallback: Any = None,
        min_confidence: float = 0.7,
        keep_alive: str = "2m",
        client: httpx.Client | None = None,
        chat_model: Any = None,
    ) -> None:
        from sohayok.brain.providers import OllamaChat

        self.decider = decider
        self.fallback = fallback
        self.min_confidence = min_confidence
        # Any provider (brain.providers.ChatModel); a local Ollama model by default.
        self.chat_model = chat_model or OllamaChat(model, url, keep_alive, client)
        self.model = self.chat_model.name
        self.last_decision: Decision | None = None

    # ---- answer model ----------------------------------------------------------------------
    def _chat(
        self, messages: list[dict], tools: list[dict] | None = None, num_predict: int = 160
    ) -> dict:
        return self.chat_model.chat(messages, tools, num_predict)

    def warm_up(self) -> None:
        self.chat_model.warm_up()

    def _say(
        self,
        prompt: str,
        language: str,
        user: str,
        extra: list[dict] | None = None,
        history: list[dict] | None = None,
    ) -> str:
        messages = [
            {"role": "system", "content": prompt.format(language=LANGUAGE[language])},
            *(history or []),
            {"role": "user", "content": user},
            *(extra or []),
        ]
        return (self._chat(messages).get("content") or "").strip()

    # ---- pipeline --------------------------------------------------------------------------
    def respond(
        self,
        text: str,
        language: str,
        executor: Executor,
        history: list[dict[str, str]],
        context: dict[str, str],
        on_text: Callable[[str], None] | None = None,
    ) -> Reply:
        started = time.perf_counter()
        decision = self.decider.decide(text, language)
        self.last_decision = decision
        log.info(
            "decided %s (%.2f) in %.0f ms",
            decision.action,
            decision.probability,
            (time.perf_counter() - started) * 1000,
        )
        action = ACTIONS.get(decision.action)
        unsure = decision.probability < self.min_confidence
        if decision.action == "chat" and not unsure:
            return Reply(self._say(CHAT_PROMPT, language, text, history=history[-4:]), [], 1)
        if action is None or not action.tools or unsure:
            if self.fallback is not None:
                return self.fallback.respond(text, language, executor, history, context, on_text)
            return Reply(self._say(CHAT_PROMPT, language, text, history=history[-4:]), [], 1)

        tools = [t for t in executor.registry.schemas() if t["name"] in action.tools]
        call = self._fill(text, tools, executor)
        if call is None:
            if self.fallback is not None:
                return self.fallback.respond(text, language, executor, history, context, on_text)
            return Reply("", [], 1)
        name, args = call
        result = executor.run(name, args)
        calls = [{"name": name, "args": args, "result": result}]
        if result.startswith(("error", "cancelled")):
            return Reply("", calls, 2)  # the assistant explains failures and cancellations
        template = DONE_REPLIES.get(name, {}).get(language)
        if template:
            try:
                return Reply(template.format(**{k: v for k, v in args.items()}), calls, 2)
            except (KeyError, IndexError):
                return Reply(re.sub(r"\{\w+\}\s*", "", template), calls, 2)
        spoken = _prepared_reply(result, language)
        if spoken:  # the tool already phrased exact numbers; a small model could garble them
            return Reply(spoken, calls, 2)
        if name == "current_time":
            spoken = f"এখন {result}।" if language == "bn" else f"It's {result}."
            return Reply(spoken, calls, 2)
        summary = self._say(
            SUMMARY_PROMPT,
            language,
            text,
            extra=[
                {"role": "assistant", "content": f"Tool {name} returned: {result[:2500]}"},
                {"role": "user", "content": "Now answer my request briefly."},
            ],
        )
        return Reply(summary, calls, 3)

    def _fill(self, text: str, tools: list[dict], executor: Executor) -> tuple[str, dict] | None:
        """Pick the tool and its arguments. Tools without required arguments need no model."""
        if len(tools) == 1 and set(tools[0]["parameters"].get("properties", {})) <= {"language"}:
            schema = tools[0]["parameters"].get("properties", {})
            args = {"language": executor.language} if "language" in schema else {}
            return tools[0]["name"], args
        message = self._chat(
            [{"role": "system", "content": FILL_PROMPT}, {"role": "user", "content": text}],
            tools=tools,
            num_predict=200,
        )
        for call in message.get("tool_calls") or []:
            function = call.get("function", {})
            name, args = function.get("name", ""), function.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            if name in {t["name"] for t in tools}:
                schema = next(t for t in tools if t["name"] == name)["parameters"]
                if "language" in schema.get("properties", {}):
                    args.setdefault("language", executor.language)
                return name, args
        return None
