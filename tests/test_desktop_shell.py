import numpy as np
import pytest

from sohayok import popup
from sohayok.icon import face
from sohayok.wake import BUILT_IN, resolve_wake_model


def test_bundled_tammy_model_loads_and_scores(tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path))  # no personal model here
    model, phrase = resolve_wake_model("tammy")
    assert model.endswith("tammy.onnx") and phrase == "Tammy"
    openwakeword = pytest.importorskip("openwakeword")
    oww = openwakeword.Model(wakeword_models=[model], inference_framework="onnx")
    score = oww.predict(np.zeros(1280, dtype=np.int16))
    assert set(score) == {"tammy"} and 0.0 <= score["tammy"] <= 1.0


def test_personal_model_is_preferred_with_its_thresholds(tmp_path, monkeypatch):
    import json

    from sohayok.wake import personal_thresholds

    monkeypatch.setenv("APPDATA", str(tmp_path))
    personal = tmp_path / "Sohayok" / "models" / "tammy-personal.onnx"
    personal.parent.mkdir(parents=True)
    personal.write_bytes(b"")
    assert resolve_wake_model("tammy") == (str(personal), "Tammy")
    assert personal_thresholds(str(personal)) is None
    personal.with_suffix(".json").write_text(
        json.dumps({"threshold": 0.9, "instant_threshold": 0.98})
    )
    assert personal_thresholds(str(personal)) == (0.9, 0.98)


def test_wake_model_names():
    assert resolve_wake_model("hey_jarvis") == ("hey_jarvis", BUILT_IN["hey_jarvis"])
    with pytest.raises(FileNotFoundError):
        resolve_wake_model("no_such_word")


def test_popup_sits_above_the_taskbar(monkeypatch):
    monkeypatch.setattr(popup, "work_area", lambda: (0, 0, 1920, 1040))
    monkeypatch.setattr(popup, "dpi_scale", lambda: 1.0)
    x, y = popup.DesktopShell("http://x/", lambda: None, lambda: None)._position()
    assert x + popup.WIDTH <= 1920 - 16 and y + popup.HEIGHT <= 1040 - 16
    assert x > 1920 // 2 and y > 0


def test_icon_draws_every_mood():
    for mood in ("happy", "listening", "sleeping"):
        image = face(64, mood)
        assert image.size == (64, 64) and image.getpixel((32, 32))[3] == 255
