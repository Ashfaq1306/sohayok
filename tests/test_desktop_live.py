"""Read-only checks against the real Windows desktop: `SOHAYOK_DESKTOP_TESTS=1 pytest -m desktop`.

They change nothing (no clicks, keys, volume changes, or launches).
"""

import os

import pytest

pytestmark = [
    pytest.mark.desktop,
    pytest.mark.skipif(
        os.environ.get("SOHAYOK_DESKTOP_TESTS") != "1",
        reason="set SOHAYOK_DESKTOP_TESTS=1 to read the real desktop",
    ),
]


def test_start_menu_index(monkeypatch):
    from sohayok.tools import apps

    monkeypatch.undo()  # use the real Start menu, not the test fixture
    found = apps.start_apps(refresh=True)
    assert len(found) > 10
    assert apps.resolve_app("ক্রোম", found) in found


def test_windows_and_screen():
    from sohayok.tools import screen, windows

    listed = windows.list_windows()
    assert listed and all(w["title"] for w in listed)
    image = screen.grab()
    assert image.width > 0 and image.height > 0
    elements = screen.describe_screen(20)
    assert elements[0]["type"] == "Window"


def test_audio_and_status():
    from sohayok.tools import media, system

    level, muted = media.get_volume()
    assert 0 <= level <= 100 and isinstance(muted, bool)
    status = system.system_status()
    assert "cpu" in status and "memory" in status
    assert system.current_time("bn")


def test_microphone_is_present():
    from sohayok.audio import input_devices

    assert input_devices()
