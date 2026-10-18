"""Small ctypes wrappers around Win32 input and window functions."""

from __future__ import annotations

import ctypes
import time
from ctypes import wintypes

user32 = ctypes.WinDLL("user32", use_last_error=True)

INPUT_MOUSE = 0
INPUT_KEYBOARD = 1
KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004
ULONG_PTR = ctypes.c_size_t


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [("uMsg", wintypes.DWORD), ("wParamL", wintypes.WORD), ("wParamH", wintypes.WORD)]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("ki", KEYBDINPUT), ("mi", MOUSEINPUT), ("hi", HARDWAREINPUT)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("union", _INPUTUNION)]


user32.SendInput.argtypes = (wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int)
user32.SendInput.restype = wintypes.UINT

# Virtual-key codes for names the model and the fast path use.
VK = {
    "backspace": 0x08,
    "tab": 0x09,
    "enter": 0x0D,
    "return": 0x0D,
    "shift": 0x10,
    "ctrl": 0x11,
    "control": 0x11,
    "alt": 0x12,
    "pause": 0x13,
    "capslock": 0x14,
    "esc": 0x1B,
    "escape": 0x1B,
    "space": 0x20,
    "pageup": 0x21,
    "pagedown": 0x22,
    "end": 0x23,
    "home": 0x24,
    "left": 0x25,
    "up": 0x26,
    "right": 0x27,
    "down": 0x28,
    "printscreen": 0x2C,
    "insert": 0x2D,
    "delete": 0x2E,
    "del": 0x2E,
    "win": 0x5B,
    "windows": 0x5B,
    "super": 0x5B,
    "cmd": 0x5B,
    "menu": 0x5D,
    "volumemute": 0xAD,
    "volumedown": 0xAE,
    "volumeup": 0xAF,
    "nexttrack": 0xB0,
    "prevtrack": 0xB1,
    "stop": 0xB2,
    "playpause": 0xB3,
    "plus": 0xBB,
    "comma": 0xBC,
    "minus": 0xBD,
    "period": 0xBE,
}
VK.update({f"f{i}": 0x6F + i for i in range(1, 25)})
_EXTENDED = {0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x2D, 0x2E, 0x5B, 0x5D}
_ALIASES = {
    "pgup": "pageup",
    "pgdn": "pagedown",
    "arrowleft": "left",
    "arrowright": "right",
    "arrowup": "up",
    "arrowdown": "down",
    "ctl": "ctrl",
    "option": "alt",
}


def _send(inputs: list[INPUT]) -> None:
    array = (INPUT * len(inputs))(*inputs)
    sent = user32.SendInput(len(inputs), array, ctypes.sizeof(INPUT))
    if sent != len(inputs):
        raise OSError(
            ctypes.get_last_error(), "SendInput was blocked (is an admin window focused?)"
        )


def _key(vk: int, up: bool) -> INPUT:
    flags = (KEYEVENTF_KEYUP if up else 0) | (KEYEVENTF_EXTENDEDKEY if vk in _EXTENDED else 0)
    return INPUT(INPUT_KEYBOARD, _INPUTUNION(ki=KEYBDINPUT(vk, 0, flags, 0, 0)))


def vk_for(name: str) -> int:
    key = name.strip().lower().replace(" ", "")
    key = _ALIASES.get(key, key)
    if key in VK:
        return VK[key]
    if len(key) == 1 and (key.isalnum()):
        return ord(key.upper())
    if len(key) == 1:
        scan = user32.VkKeyScanW(ord(key))
        if scan != -1:
            return scan & 0xFF
    raise ValueError(f"unknown key {name!r}")


def press(combo: str, repeat: int = 1) -> None:
    """Press a key combination such as "ctrl+shift+t", "alt+f4", or "enter"."""
    keys = [vk_for(part) for part in combo.replace("-", "+").split("+") if part.strip()]
    if not keys:
        raise ValueError("no keys given")
    for _ in range(max(1, repeat)):
        _send([_key(vk, False) for vk in keys] + [_key(vk, True) for vk in reversed(keys)])
        time.sleep(0.03)


def type_unicode(text: str, delay: float = 0.004) -> None:
    """Type any text, Bangla included, into the focused window as Unicode keystrokes."""
    for char in text:
        if char == "\n":
            press("enter")
            continue
        units = char.encode("utf-16-le")
        events = []
        for i in range(0, len(units), 2):
            code = int.from_bytes(units[i : i + 2], "little")
            for up in (False, True):
                flags = KEYEVENTF_UNICODE | (KEYEVENTF_KEYUP if up else 0)
                events.append(
                    INPUT(INPUT_KEYBOARD, _INPUTUNION(ki=KEYBDINPUT(0, code, flags, 0, 0)))
                )
        _send(events)
        if delay:
            time.sleep(delay)


def make_dpi_aware() -> None:
    """Use physical pixels so screenshots and mouse coordinates agree on scaled displays."""
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except (AttributeError, OSError):
        try:
            user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            pass


def screen_size() -> tuple[int, int]:
    return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)


def lock_workstation() -> None:
    if not user32.LockWorkStation():
        raise OSError(ctypes.get_last_error(), "LockWorkStation failed")


def foreground_window() -> int:
    return user32.GetForegroundWindow()


def force_foreground(hwnd: int) -> bool:
    """Bring a window to the front even when another process owns the foreground.

    Windows only lets the foreground process change the foreground window; a synthetic
    Alt press counts as user input and lifts that restriction for this call.
    """
    SW_RESTORE = 9
    if user32.IsIconic(hwnd):
        user32.ShowWindow(hwnd, SW_RESTORE)
    _send([_key(VK["alt"], False), _key(VK["alt"], True)])
    ok = bool(user32.SetForegroundWindow(hwnd))
    user32.BringWindowToTop(hwnd)
    return ok
