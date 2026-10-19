import time

import numpy as np
from fastapi.testclient import TestClient

from sohayok.assistant import Assistant
from sohayok.bench.brain import Case, judge, reply_language_ok, stub_registry
from sohayok.bench.stt import error_rates, normalize_transcript
from sohayok.config import Config
from sohayok.server import create_app
from sohayok.stt import Transcriber, clean
from sohayok.tools.registry import Registry


def test_server_endpoints_and_events():
    assistant = Assistant(Config(), Registry(), None)
    client = TestClient(create_app(assistant, {"backend": "Ollama (local)", "model": "m"}))
    page = client.get("/")
    assert page.status_code == 200 and "সহায়ক" in page.text
    status = client.get("/api/status").json()
    assert status["backend"] == "Ollama (local)" and status["voice_input"] is False
    assert client.post("/api/confirm", json={"answer": True}).json() == {"resolved": False}
    with client.websocket_connect("/ws") as socket:
        hello = socket.receive_json()
        assert hello["type"] == "hello" and hello["model"] == "m"
        assistant.emit("reply", text="হ্যালো", language="bn")
        event = socket.receive_json()
        assert event["type"] == "reply" and event["text"] == "হ্যালো"
    assert client.post("/api/command", json={"text": "hello"}).json() == {"status": "queued"}
    assert assistant.commands.qsize() == 1


def test_clean_drops_whisper_hallucinations():
    assert clean("  Thank you. ") == ""
    assert clean("ধন্যবাদ।") == ""
    assert clean("ক্রোম   খোলো") == "ক্রোম খোলো"


def test_transcriber_routes_bangla_to_the_bangla_model(monkeypatch):
    import sohayok.stt as stt

    monkeypatch.setattr(stt, "speech_seconds", lambda audio, **_: 1.0)

    class Multi:
        def detect(self, audio, allowed):
            return "bn", 0.9

        def transcribe(self, audio, language):
            return "multi"

    class Bangla:
        def transcribe(self, audio, language):
            return "ভলিউম বাড়াও"

    transcriber = Transcriber.__new__(Transcriber)
    transcriber.languages, transcriber.multi, transcriber.bangla = ["bn", "en"], Multi(), Bangla()
    transcriber.detector = None
    heard = transcriber.transcribe(np.zeros(1600, dtype=np.float32))
    assert (heard.text, heard.language, heard.language_probability) == ("ভলিউম বাড়াও", "bn", 0.9)
    transcriber.bangla = None
    assert transcriber.transcribe(np.zeros(1600, dtype=np.float32)).text == "multi"

    class Detector:  # the spoken-language classifier is asked first when it is loaded
        def margin(self, audio, first, second):
            return -10.0  # clearly English

    transcriber.detector = Detector()
    assert transcriber.transcribe(np.zeros(1600, dtype=np.float32)).language == "en"


def test_bangla_or_english_combines_both_detectors():
    from sohayok.stt import bangla_or_english

    # the user's Bangla that Whisper called English (VoxLingua margin 17, Whisper p(en) 0.97)
    assert bangla_or_english(17.0, 0.97)[0] == "bn"
    # Indian-accented English that VoxLingua called Bangla (margin 6, Whisper p(en) 0.99)
    assert bangla_or_english(6.0, 0.99)[0] == "en"
    assert bangla_or_english(20.0, None) == ("bn", 1.0)
    assert bangla_or_english(-12.0, None)[0] == "en"


def test_silence_is_not_decoded(monkeypatch):
    import sohayok.stt as stt

    monkeypatch.setattr(stt, "speech_seconds", lambda audio, **_: 0.0)

    class Fails:
        def detect(self, audio, allowed):
            raise AssertionError("silence must not reach a model")

        transcribe = detect

    transcriber = Transcriber.__new__(Transcriber)
    transcriber.languages, transcriber.multi, transcriber.bangla = ["bn", "en"], Fails(), Fails()
    transcriber.detector = Fails()
    assert transcriber.transcribe(np.zeros(16_000, dtype=np.float32)).text == ""


def test_transcript_normalization_keeps_bangla_words_whole():
    assert normalize_transcript("আমি ভাত খাই। Hello, World!") == "আমি ভাত খাই hello world"
    rates = error_rates(["আমি ভাত খাই"], ["আমি ভাত খাই।"])
    assert rates == {"wer": 0.0, "cer": 0.0}
    assert error_rates(["আমি ভাত খাই"], ["আমি ভাত খায়"])["wer"] == 1 / 3


def test_brain_benchmark_judging():
    case = Case(
        "open notepad and type hi",
        "en",
        [{"open_app"}, {"type_text"}],
        "desktop",
        "open_app",
        {"type_text": {"text": "hi"}},
    )
    good = [
        {"name": "open_app", "args": {"name": "Notepad"}},
        {"name": "type_text", "args": {"text": "hi"}},
    ]
    assert judge(case, good) == (True, True)
    assert judge(case, good[::-1]) == (False, False)
    assert judge(case, [good[0], {"name": "type_text", "args": {"text": "bye"}}]) == (True, False)
    chat = Case("তুমি কে?", "bn", [{""}], "desktop", "chat")
    assert judge(chat, []) == (True, True) and judge(chat, good)[0] is False
    assert reply_language_ok(chat, "আমি সহায়ক।") and not reply_language_ok(chat, "I am Sohayok")


def test_stub_registry_never_runs_real_tools():
    registry, calls = stub_registry()
    started = time.monotonic()
    registry.call("power", {"action": "shutdown"})
    registry.call("volume", {"action": "up"})
    assert [c["name"] for c in calls] == ["power", "volume"]
    assert time.monotonic() - started < 0.5


def test_live_events_work_through_the_real_server():
    """Uvicorn needs a WebSocket library; TestClient alone would not catch a missing one."""
    import json
    import socket
    import threading

    import uvicorn
    from websockets.sync.client import connect

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    assistant = Assistant(Config(), Registry(), None)
    server = uvicorn.Server(
        uvicorn.Config(create_app(assistant), host="127.0.0.1", port=port, log_level="error")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        for _ in range(100):
            if server.started:
                break
            time.sleep(0.05)
        with connect(f"ws://127.0.0.1:{port}/ws") as socket_:
            assert json.loads(socket_.recv(timeout=5))["type"] == "hello"
            assistant.emit("reply", text="ঠিক আছে", language="bn")
            assert json.loads(socket_.recv(timeout=5))["text"] == "ঠিক আছে"
    finally:
        server.should_exit = True
        thread.join(5)


def test_multilingual_proxy_delegates():
    from sohayok.stt import _Multilingual

    class Model:
        is_multilingual = False
        n_mels = 128

        def generate(self, *args):
            return "generated"

    proxy = _Multilingual(Model())
    assert proxy.is_multilingual is True
    assert proxy.n_mels == 128 and proxy.generate() == "generated"


def test_choose_language_treats_indic_guesses_as_bangla():
    from sohayok.stt import choose_language

    # Whisper's typical guess for a short Bangla command: Hindi first, English low.
    assert choose_language({"hi": 0.51, "gu": 0.2, "en": 0.05, "bn": 0.03}, ["bn", "en"])[0] == "bn"
    assert choose_language({"en": 0.93, "bn": 0.01}, ["bn", "en"]) == ("en", 0.93)
    assert choose_language({"en": 0.2, "fr": 0.7, "de": 0.1}, ["fr", "de"])[0] == "fr"
    assert choose_language({}, ["bn", "en"])[0] == "bn"


def test_quiet_audio_is_brought_up_and_recordings_are_pruned(tmp_path):
    from sohayok.audio import normalize_level, save_recording

    quiet = (np.sin(np.linspace(0, 200, 16_000)) * 0.05).astype(np.float32)
    louder = normalize_level(quiet)
    assert 0.45 < float(np.abs(louder).max()) <= 1.0
    assert normalize_level(np.zeros(100, dtype=np.float32)).max() == 0
    for i in range(5):
        save_recording(tmp_path, quiet, {"text": f"t{i}"}, keep=3)
    assert len(list(tmp_path.glob("*.wav"))) == 3 and len(list(tmp_path.glob("*.json"))) == 3


def test_wake_name_handling():
    from sohayok.desktop import heard_wake_name
    from sohayok.router import strip_wake_name

    assert strip_wake_name("Tammy, open chrome") == "open chrome"
    assert strip_wake_name("টামি ক্রোম খোলো") == "ক্রোম খোলো"
    assert strip_wake_name("Tamil movies") == "Tamil movies"
    assert heard_wake_name("Tammy.") and heard_wake_name("Hey Tommy") and heard_wake_name("ট্যামি")
    assert not heard_wake_name("open the window") and not heard_wake_name("আমি ভালো আছি")
    # what Whisper wrote for the user's real "Tammy" attempts
    assert all(heard_wake_name(t) for t in ("Demi.", "Demi", "Damn me?", "Atatemi"))
    assert not any(heard_wake_name(t) for t in ("you", "Thank you.", "Ahem!"))


def test_wake_echo_is_dropped_from_one_breath_commands():
    from sohayok.router import strip_wake_echo

    assert strip_wake_echo("ডেমি ক্রোম খোলো") == "ক্রোম খোলো"
    assert strip_wake_echo("Dummy, open Chrome") == "open Chrome"
    assert strip_wake_echo("ডেমি") == "ডেমি"  # nothing else said: keep it
    # the user's own way of talking is kept
    assert strip_wake_echo("তুমি গানটা বন্ধ করো") == "তুমি গানটা বন্ধ করো"
    assert strip_wake_echo("you open the Chrome browser") == "you open the Chrome browser"


def test_whisper_loops_are_collapsed():
    from sohayok.stt import clean

    looped = (
        "you can open the tab and close the table and close the table and close the table "
        "and close the table"
    )
    assert clean(looped) == "you can open the tab and close the table"
    assert clean("ভলিউম বাড়াও") == "ভলিউম বাড়াও"
    assert clean("no no no") == "no"


def test_user_wake_spellings_in_bangla():
    from sohayok.desktop import heard_wake_name

    assert all(heard_wake_name(t) for t in ("ডেমি", "টেমি", "তামি", "ড্যামি", "কেমি", "dummy"))
    assert not any(heard_wake_name(t) for t in ("কী", "সার্চ", "লাইফস্টাইল", "Search."))


def test_wake_listener_reopens_a_lost_microphone(monkeypatch):
    """Bluetooth earbuds disconnecting used to kill the wake-word thread for good."""
    import sounddevice as sd

    from sohayok import wake

    listener = object.__new__(wake.WakeWordListener)
    listener.name, listener.device, listener.restarts = "tammy", None, 0
    listener.model = type("M", (), {"reset": lambda self: None})()
    opened = []

    class Done(BaseException):
        pass

    def listen(device):
        opened.append(device)
        if len(opened) == 1:
            raise sd.PortAudioError("There is no driver installed on your system.")
        raise Done

    monkeypatch.setattr(listener, "_listen", listen)
    monkeypatch.setattr(wake.time, "sleep", lambda s: None)
    monkeypatch.setattr(sd, "_terminate", lambda: None)
    monkeypatch.setattr(sd, "_initialize", lambda: None)
    try:
        listener.run()
    except Done:
        pass
    assert len(opened) == 2 and listener.restarts == 1


def test_wake_listener_follows_the_windows_default_microphone(monkeypatch):
    """Windows briefly switched to the built-in microphone and back; the wake word stayed on
    the built-in one."""
    import sounddevice as sd

    from sohayok import wake

    listener = object.__new__(wake.WakeWordListener)
    listener.name, listener.device, listener.restarts = "tammy", None, 0
    listener.model = type("M", (), {"reset": lambda self: None})()
    listener.paused = lambda: False
    listener.gain, listener.threshold = 1.0, 0.5
    listener.score = lambda frame: 0.0
    defaults = iter(["USB", "USB", "Realtek"])
    opened = []

    class Stream:
        device = 1

        def __init__(self, **kwargs):
            opened.append(kwargs["device"])
            if len(opened) > 1:
                raise KeyboardInterrupt  # stop the test once the stream is reopened

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, n):
            return np.zeros((n, 1), np.int16), False

    monkeypatch.setattr(sd, "InputStream", Stream)
    monkeypatch.setattr(sd, "query_devices", lambda d: {"name": "USB"})
    monkeypatch.setattr(sd, "_terminate", lambda: None)
    monkeypatch.setattr(sd, "_initialize", lambda: None)
    monkeypatch.setattr(wake, "default_microphone", lambda: next(defaults))
    try:
        listener.run()
    except KeyboardInterrupt:
        pass
    assert len(opened) == 2 and listener.restarts == 1
