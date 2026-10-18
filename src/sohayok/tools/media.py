"""Volume and media playback."""

from __future__ import annotations

from sohayok import winapi
from sohayok.tools.registry import ToolError, integer, string, tool


def _endpoint():
    import comtypes
    from pycaw.pycaw import AudioUtilities

    comtypes.CoInitialize()  # tools run on worker threads
    speakers = AudioUtilities.GetSpeakers()
    if speakers is None:
        raise ToolError("no audio output device")
    return speakers.EndpointVolume


def get_volume() -> tuple[int, bool]:
    endpoint = _endpoint()
    return round(endpoint.GetMasterVolumeLevelScalar() * 100), bool(endpoint.GetMute())


@tool(
    "Set, raise, lower, mute, or unmute the speaker volume, or report it.",
    {
        "action": string("What to do", ["set", "up", "down", "mute", "unmute", "get"]),
        "level": integer("0-100 for 'set'; step size for 'up'/'down' (default 10)"),
    },
    required=["action"],
)
def volume(action: str, level: int | None = None) -> str:
    endpoint = _endpoint()
    current = round(endpoint.GetMasterVolumeLevelScalar() * 100)
    if action == "get":
        muted = " (muted)" if endpoint.GetMute() else ""
        return f"volume is {current}%{muted}"
    if action in {"mute", "unmute"}:
        endpoint.SetMute(1 if action == "mute" else 0, None)
        return f"{action}d"
    if action == "set":
        if level is None:
            raise ToolError("say which level, 0 to 100")
        target = level
    elif action in {"up", "down"}:
        step = 10 if level is None else abs(int(level))
        target = current + step if action == "up" else current - step
    else:
        raise ToolError(f"unknown volume action {action!r}")
    target = max(0, min(100, int(target)))
    endpoint.SetMute(0, None)
    endpoint.SetMasterVolumeLevelScalar(target / 100, None)
    return f"volume {current}% → {target}%"


@tool(
    "Control whatever is playing (YouTube, Spotify, VLC...): play/pause, next, previous, stop.",
    {"action": string("Media key", ["playpause", "next", "previous", "stop"])},
)
def media(action: str) -> str:
    key = {
        "playpause": "playpause",
        "play": "playpause",
        "pause": "playpause",
        "next": "nexttrack",
        "previous": "prevtrack",
        "stop": "stop",
    }.get(action)
    if key is None:
        raise ToolError(f"unknown media action {action!r}")
    winapi.press(key)
    return f"sent {action}"
