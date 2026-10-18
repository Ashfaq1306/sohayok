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


def test_transcriber_routes_bangla_to_the_bangla_model():
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
    heard = transcriber.transcribe(np.zeros(1600, dtype=np.float32))
    assert (heard.text, heard.language, heard.language_probability) == ("ভলিউম বাড়াও", "bn", 0.9)
    transcriber.bangla = None
    assert transcriber.transcribe(np.zeros(1600, dtype=np.float32)).text == "multi"


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
