"""The assistant: listen, understand, act, and answer, one command at a time."""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from sohayok.brain.base import Brain
from sohayok.config import Config
from sohayok.lang import detect_language, normalize
from sohayok.router import route
from sohayok.safety import ConfirmationBroker, Executor, Pending, parse_answer
from sohayok.tools.registry import Registry

log = logging.getLogger(__name__)

STOP_WORDS = {
    "stop",
    "quiet",
    "be quiet",
    "shut up",
    "cancel",
    "never mind",
    "থামো",
    "চুপ",
    "চুপ করো",
    "থাক",
    "বাদ দাও",
    "থামুন",
    "বাতিল",
    "chup",
    "thamo",
    "thak",
    "bad dao",
}
GREETING = {
    "bn": "আসসালামু আলাইকুম! আমি সহায়ক। কী করতে পারি?",
    "en": "Hi, I'm Sohayok. What can I do for you?",
}
DONE = {"bn": "হয়ে গেছে।", "en": "Done."}
FAILED = {"bn": "দুঃখিত, কাজটা করতে পারিনি।", "en": "Sorry, I couldn't do that."}
NOT_HEARD = {"bn": "শুনতে পাইনি, আবার বলুন।", "en": "I didn't catch that. Please say it again."}
CONFIRM_PROMPT = {"bn": "{q}? হ্যাঁ বা না বলুন।", "en": "{q}. Shall I? Say yes or no."}
CANCELLED = {"bn": "ঠিক আছে, করলাম না।", "en": "Okay, I won't."}
BRAIN_DOWN = {
    "bn": "এখন মডেলের সাথে যোগাযোগ করতে পারছি না।",
    "en": "I can't reach the language model right now.",
}


@dataclass
class Command:
    text: str
    language: str | None = None
    source: str = "text"  # "voice" or "text"


class Assistant:
    def __init__(
        self,
        config: Config,
        registry: Registry,
        brain: Brain | None,
        speaker: Any = None,
        transcriber: Any = None,
        recorder: Any = None,
        window_title: Callable[[], str] = lambda: "",
    ) -> None:
        self.config = config
        self.registry = registry
        self.brain = brain
        self.speaker = speaker
        self.transcriber = transcriber
        self.recorder = recorder
        self.window_title = window_title
        self.chime: Callable[[], None] = lambda: None  # played when listening starts
        self.listeners: list[Callable[[dict[str, Any]], None]] = []
        self.history: list[dict[str, str]] = []
        self.state = "idle"
        self.language = config.assistant.default_language
        self.broker = ConfirmationBroker(config.assistant.confirm_timeout_seconds)
        self.broker.on_ask = self._on_confirm_ask
        self.broker.on_done = self._on_confirm_done
        self.executor = Executor(
            registry, self.broker, config.assistant.confirm_risky_actions, on_tool=self._on_tool
        )
        self.commands: queue.Queue[Command] = queue.Queue()
        self._listening = threading.Lock()
        self._answer_listening = False
        self._worker = threading.Thread(target=self._work, name="assistant", daemon=True)

    # ---- events -------------------------------------------------------------------------
    def emit(self, kind: str, **data: Any) -> None:
        event = {"type": kind, "time": time.time(), **data}
        for listener in list(self.listeners):
            try:
                listener(event)
            except Exception:
                log.exception("event listener failed")

    def set_state(self, state: str) -> None:
        self.state = state
        self.emit("state", state=state)

    def _on_tool(self, name: str, args: dict[str, Any], result: str) -> None:
        self.emit("tool", name=name, args=args, result=result[:500])

    def _on_confirm_ask(self, pending: Pending) -> None:
        prompt = CONFIRM_PROMPT[pending.language].format(q=pending.question.rstrip(".।?"))
        self.emit("confirm", question=pending.question, prompt=prompt)
        if self.speaker:
            self.speaker.say(prompt)
        if self.transcriber and self.recorder:
            # Listen for the spoken yes/no on a separate thread; the worker is waiting.
            threading.Thread(
                target=self.listen_once, args=(True,), name="confirm-listen", daemon=True
            ).start()

    def _on_confirm_done(self, pending: Pending) -> None:
        self.emit("confirm_done", answer=pending.answer)
        if self._answer_listening and self.recorder:
            self.recorder.stop()  # answered by click: stop waiting for a spoken answer

    # ---- lifecycle ----------------------------------------------------------------------
    def start(self, greet: bool = True) -> None:
        self._worker.start()
        if greet:
            self.reply(GREETING[self.language], self.language)

    def submit(self, text: str, language: str | None = None, source: str = "text") -> None:
        """Queue a command, unless it answers a pending confirmation or interrupts speech."""
        text = text.strip()
        if not text:
            return
        if self.broker.pending is not None:
            answer = parse_answer(text)
            if answer is not None:
                self.emit("heard", text=text, language=language or detect_language(text))
                self.broker.resolve(answer)
                return
        if normalize(text).strip(" .!।") in STOP_WORDS:
            self.emit("heard", text=text, language=language or detect_language(text))
            self.interrupt()
            return
        self.commands.put(Command(text, language, source))

    def interrupt(self) -> None:
        if self.speaker:
            self.speaker.stop()
        if self.recorder:
            self.recorder.stop()
        self.broker.resolve(False)

    # ---- voice --------------------------------------------------------------------------
    def listen_once(self, answer_only: bool = False) -> None:
        """Record one utterance, transcribe it, and submit it.

        With `answer_only`, the recording is for a pending yes/no question and is discarded
        once that question has been answered some other way.
        """
        if not (self.transcriber and self.recorder):
            self.emit("error", message="speech recognition is not available")
            return
        if not self._listening.acquire(blocking=False):
            self.recorder.stop()  # a second press ends the current recording
            return
        self._answer_listening = answer_only
        try:
            if self.speaker:
                self.speaker.stop()
            if answer_only and self.broker.pending is None:
                return
            self.set_state("listening")
            try:
                self.chime()
            except Exception:
                pass
            audio = self.recorder.record()
            if answer_only and self.broker.pending is None:
                return
            if audio.size < 16_000 * 0.3:
                self.set_state("idle")
                return
            self.set_state("transcribing")
            heard = self.transcriber.transcribe(audio)
            self.emit(
                "transcript",
                text=heard.text,
                language=heard.language,
                confidence=round(heard.language_probability, 3),
                seconds=round(heard.seconds, 2),
            )
            if not heard.text:
                self.set_state("idle")
                if self.broker.pending is None:
                    self.reply(NOT_HEARD[self.language], self.language)
                return
            if answer_only and self.broker.pending is None:
                return  # answered meanwhile; never turn this recording into a command
            if self.broker.pending is not None and not self._clear_answer(heard.text, audio):
                self.emit("heard", text=heard.text, language=heard.language, source="voice")
                self.emit("error", message="answer not clear; press Yes/No or say it again")
                return
            self.submit(heard.text, heard.language, "voice")
        except Exception as exc:
            log.exception("listening failed")
            self.emit("error", message=f"microphone or recognition error: {exc}")
        finally:
            self._answer_listening = False
            if self.state in {"listening", "transcribing"}:
                self.set_state("idle")
            self._listening.release()

    @staticmethod
    def _clear_answer(text: str, audio: Any) -> bool:
        """Accept a spoken confirmation only if it is short and really contains speech.

        A misheard "yes" could approve a shutdown, so noise, echoes, and long sentences that
        merely contain "yes" are not taken as answers ("no" is always safe to accept).
        """
        answer = parse_answer(text)
        if answer is False:
            return True
        return answer is True and len(text.split()) <= 4 and len(audio) >= 16_000 * 0.35

    # ---- understanding and acting -------------------------------------------------------
    def reply(self, text: str, language: str) -> None:
        self.emit("reply", text=text, language=language)
        if self.speaker and text:
            self.set_state("speaking")
            try:
                self.speaker.say(text)
            except Exception:
                log.exception("speech output failed")
        self.set_state("idle")

    def _work(self) -> None:
        while True:
            command = self.commands.get()
            try:
                self.handle(command)
            except Exception as exc:
                log.exception("command failed")
                self.emit("error", message=str(exc))
                self.reply(FAILED.get(self.language, FAILED["en"]), self.language)

    def handle(self, command: Command) -> str:
        language = command.language or detect_language(command.text, self.language)
        if language not in {"bn", "en"}:
            language = "en"
        self.language = language
        self.executor.language = language
        self.emit("heard", text=command.text, language=language, source=command.source)

        intent = route(command.text)
        if intent is not None:
            self.set_state("acting")
            args = dict(intent.args)
            tool = self.registry.get(intent.tool)
            if tool and "language" in tool.parameters.get("properties", {}):
                args.setdefault("language", language)  # e.g. the time read out in Bangla
            result = self.executor.run(intent.tool, args)
            if not result.startswith(("error", "cancelled")):
                text = intent.reply.get(language, intent.reply.get("en", DONE[language]))
                text = text.replace("{result}", result)
                self._remember(command.text, text)
                self.emit("route", tool=intent.tool, fast=True)
                self.reply(text, language)
                return text
            if result.startswith("cancelled"):
                self.reply(CANCELLED[language], language)
                return CANCELLED[language]
            log.info("fast path failed (%s); asking the model", result)

        if self.brain is None:
            self.reply(BRAIN_DOWN[language], language)
            return BRAIN_DOWN[language]
        self.set_state("thinking")
        started = time.monotonic()
        try:
            answer = self.brain.respond(
                command.text,
                language,
                self.executor,
                self.history[-2 * self.config.assistant.history_turns :],
                {"window": self._window()},
                on_text=lambda t: self.emit("partial", text=t),
            )
        except Exception as exc:
            log.exception("model call failed")
            self.emit("error", message=f"model error: {exc}")
            self.reply(BRAIN_DOWN[language], language)
            return BRAIN_DOWN[language]
        self.emit(
            "route",
            tool=",".join(c["name"] for c in answer.tool_calls) or "chat",
            fast=False,
            steps=answer.steps,
            seconds=round(time.monotonic() - started, 2),
        )
        text = answer.text
        if any(c["result"].startswith("cancelled") for c in answer.tool_calls):
            # Never let the model claim a declined action happened.
            text = CANCELLED[language]
        elif not text:
            failed = answer.tool_calls and answer.tool_calls[-1]["result"].startswith("error")
            text = (FAILED if failed else DONE)[language]
        self._remember(command.text, text)
        self.reply(text, language)
        return text

    def _window(self) -> str:
        try:
            return self.window_title()
        except Exception:
            return ""

    def _remember(self, user: str, assistant: str) -> None:
        self.history += [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
        self.history = self.history[-2 * self.config.assistant.history_turns :]
