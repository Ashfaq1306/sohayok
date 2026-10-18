import pytest

from sohayok.tools import apps

FAKE_APPS = {
    "Google Chrome": "Chrome",
    "Notepad": "Microsoft.WindowsNotepad_8wekyb3d8bbwe!App",
    "Visual Studio Code": "Microsoft.VisualStudioCode",
    "Calculator": "Microsoft.WindowsCalculator_8wekyb3d8bbwe!App",
    "File Explorer": "Microsoft.Windows.Explorer",
    "Settings": "windows.immersivecontrolpanel_cw5n1h2txyewy!microsoft.windows.immersivecontrolpanel",
    "Spotify": "SpotifyAB.SpotifyMusic_zpdnekdrzrea0!Spotify",
    "Telegram Desktop": "TelegramDesktop",
}


@pytest.fixture(autouse=True)
def fake_start_apps(monkeypatch):
    """Tests never shell out to PowerShell for the Start menu."""
    monkeypatch.setattr(apps, "start_apps", lambda refresh=False: dict(FAKE_APPS))


def pytest_configure(config):
    config.addinivalue_line("markers", "desktop: reads the real Windows desktop (read-only)")
