import threading
import time

import numpy as np

from sohayok.assistant import Assistant, Command
from sohayok.brain.base import Reply
from sohayok.config import Config
from sohayok.stt import Transcript
from sohayok.tools.registry import Registry, Risk, Tool


def _registry(calls):
    registry = Registry()

    def add(name, result="ok", risk=Risk.SAFE):
        def run(**kwargs):
            calls.append((name, kwargs))
            return result

        registry.add(Tool(name, name, {"type": "object", "properties": {}}, run, risk))

    add("open_app", "opened Google Chrome")
    add("volume", "volume 40% → 50%")
    add("current_time", "সকাল ১০টা ৩০ মিনিট")
    add("power", "shutdown scheduled", Risk.CONFIRM)
    return registry


class FakeBrain:
    def __init__(self, text="ঠিক আছে।", tool=None):
        self.text, self.tool, self.seen = text, tool, []

    def respond(self, text, language, executor, history, context, on_text=None):
        self.seen.append((text, language, list(history)))
        calls = []
        if self.tool:
            result = executor.run(self.tool, {})
            calls.append({"name": self.tool, "args": {}, "result": result})
        return Reply(self.text, calls, 1)


def _assistant(brain=None, calls=None):
    config = Config()
    config.assistant.confirm_timeout_seconds = 2
    assistant = Assistant(config, _registry(calls if calls is not None else []), brain)
    events = []
    assistant.listeners.append(events.append)
    return assistant, events


def test_fast_path_acts_and_replies_in_bangla():
    calls = []
    brain = FakeBrain()
    assistant, events = _assistant(brain, calls)
    reply = assistant.handle(Command("ক্রোম খোলো"))
    assert calls == [("open_app", {"name": "Google Chrome"})]
    assert reply == "Google Chrome খুলছি।" and brain.seen == []
    assert [e for e in events if e["type"] == "route"][0]["fast"] is True


def test_fast_path_reply_can_include_the_result():
    assistant, _ = _assistant(FakeBrain())
    assert assistant.handle(Command("কয়টা বাজে?")) == "এখন সকাল ১০টা ৩০ মিনিট।"


def test_everything_else_goes_to_the_model_with_history():
    brain = FakeBrain("Here is a joke.")
    assistant, _ = _assistant(brain)
    assistant.handle(Command("ক্রোম খোলো"))
    assert assistant.handle(Command("tell me a joke")) == "Here is a joke."
    text, language, history = brain.seen[0]
    assert (text, language) == ("tell me a joke", "en")
    assert history[0] == {"role": "user", "content": "ক্রোম খোলো"}


def test_empty_model_reply_gets_a_spoken_fallback():
    assistant, _ = _assistant(FakeBrain("", tool="volume"))
    assert assistant.handle(Command("আওয়াজটা একটু ঠিক করে দাও")) == "হয়ে গেছে।"


def test_no_model_says_so():
    assistant, _ = _assistant(None)
    assert "মডেল" in assistant.handle(Command("একটা কবিতা শোনাও"))


def test_spoken_yes_approves_a_risky_action():
    calls = []
    assistant, events = _assistant(FakeBrain("বন্ধ করছি।", tool="power"), calls)
    assistant.start(greet=False)
    assistant.submit("কম্পিউটার বন্ধ করে দাও")
    for _ in range(100):
        if assistant.broker.pending:
            break
        time.sleep(0.02)
    assert assistant.broker.pending is not None
    assistant.submit("হ্যাঁ")  # resolves the question instead of becoming a command
    for _ in range(100):
        if calls:
            break
        time.sleep(0.02)
    assert calls == [("power", {})]
    assert any(e["type"] == "confirm" and "হ্যাঁ বা না বলুন" in e["prompt"] for e in events)


def test_spoken_no_cancels():
    calls = []
    assistant, _ = _assistant(FakeBrain("ঠিক আছে।", tool="power"), calls)
    assistant.broker.on_ask = lambda p: threading.Thread(
        target=lambda: (time.sleep(0.05), assistant.submit("না"))
    ).start()
    assistant.handle(Command("কম্পিউটার বন্ধ করে দাও"))
    assert calls == []


def test_stop_word_interrupts_speech():
    class Speaker:
        stopped = False

        def stop(self):
            Speaker.stopped = True

        def say(self, text):
            pass

    assistant, _ = _assistant(FakeBrain())
    assistant.speaker = Speaker()
    assistant.submit("থামো")
    assert Speaker.stopped and assistant.commands.empty()


def test_voice_command_flows_from_microphone_to_action():
    calls = []

    class Recorder:
        def record(self):
            return np.ones(16_000, dtype=np.float32)

        def stop(self):
            pass

    class Transcriber:
        def transcribe(self, audio):
            return Transcript("ভলিউম বাড়াও", "bn", 0.97, 0.2)

    assistant, events = _assistant(FakeBrain(), calls)
    assistant.recorder, assistant.transcriber = Recorder(), Transcriber()
    assistant.start(greet=False)
    assistant.listen_once()
    for _ in range(100):
        if calls:
            break
        time.sleep(0.02)
    assert calls == [("volume", {"action": "up", "level": 10})]
    assert [e["state"] for e in events if e["type"] == "state"][:2] == ["listening", "transcribing"]


def test_command_said_while_speech_loads_is_kept():
    """Calling Tammy before the recognizer finished loading said "not available"."""
    import threading

    calls = []

    class Recorder:
        def record(self):
            return np.ones(16_000, dtype=np.float32)

        def stop(self):
            pass

    class Transcriber:
        def transcribe(self, audio):
            return Transcript("ভলিউম বাড়াও", "bn", 0.97, 0.2)

    assistant, events = _assistant(FakeBrain(), calls)
    assistant.recorder = Recorder()
    assistant.start(greet=False)

    def finish_loading():
        time.sleep(0.2)
        assistant.transcriber = Transcriber()
        assistant.speech_ready.set()

    threading.Thread(target=finish_loading).start()
    assistant.listen_once()
    for _ in range(100):
        if calls:
            break
        time.sleep(0.02)
    assert calls == [("volume", {"action": "up", "level": 10})]
    assert not any(e["type"] == "error" for e in events)
    assert any(e["type"] == "reply" and "চালু হচ্ছি" in e["text"] for e in events)


def test_spoken_confirmations_must_be_clear():
    speech = np.ones(16_000, dtype=np.float32)
    assert Assistant._clear_answer("হ্যাঁ", speech)
    assert Assistant._clear_answer("না", np.ones(100, dtype=np.float32))  # no is always safe
    assert not Assistant._clear_answer("yes", np.ones(1_000, dtype=np.float32))  # a blip
    assert not Assistant._clear_answer("yes I was telling my friend about the movie", speech)
    assert not Assistant._clear_answer("ক্রোম খোলো", speech)


def test_model_cannot_claim_a_declined_action():
    calls = []
    brain = FakeBrain("কম্পিউটার রিস্টার্ট করা হলো।", tool="power")
    assistant, events = _assistant(brain, calls)
    assistant.broker.on_ask = lambda p: assistant.broker.resolve(False)
    assert assistant.handle(Command("কম্পিউটার রিস্টার্ট করো")) == "ঠিক আছে, করলাম না।"
    assert calls == []


def test_clicked_answer_discards_the_pending_recording():
    calls = []

    class Recorder:
        stopped = threading.Event()

        def record(self):
            Recorder.stopped.wait(2)  # recording until the click stops it
            return np.ones(16_000, dtype=np.float32)

        def stop(self):
            Recorder.stopped.set()

    class Transcriber:
        def transcribe(self, audio):
            return Transcript("Fica a secco l'amore.", "en", 0.6, 0.2)

    assistant, events = _assistant(FakeBrain("ঠিক আছে।", tool="power"), calls)
    assistant.recorder, assistant.transcriber = Recorder(), Transcriber()
    worker = threading.Thread(target=assistant.handle, args=(Command("রিস্টার্ট করো"),))
    worker.start()
    for _ in range(100):
        if assistant._answer_listening:
            break
        time.sleep(0.02)
    assert assistant.broker.resolve(False)  # the user clicked "No"
    worker.join(3)
    time.sleep(0.2)
    assert calls == [] and assistant.commands.empty()
    assert not any(e["type"] == "heard" and "amore" in e["text"] for e in events)


def test_fast_path_runs_every_step():
    calls = []
    assistant, _ = _assistant(FakeBrain(), calls)
    assistant.registry.add(
        Tool(
            "type_text",
            "t",
            {"type": "object", "properties": {}},
            lambda **kw: calls.append(("type_text", kw)) or "typed",
        )
    )
    reply = assistant.handle(Command("নোটপ্যাড খুলে লেখো আমি ভালো আছি"))
    assert calls == [("open_app", {"name": "Notepad"}), ("type_text", {"text": "আমি ভালো আছি"})]
    assert reply == "Notepad খুলে লিখে দিলাম।"
