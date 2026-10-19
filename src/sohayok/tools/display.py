"""Screen brightness and dark mode.

Brightness is set the best way the screen allows: a laptop panel through Windows (WMI), an
external monitor over its cable (DDC/CI), and otherwise by dimming the picture in the graphics
card (a gamma ramp), which works on every screen but cannot make it brighter than normal.
"""

from __future__ import annotations

import ctypes
import subprocess
import winreg
from ctypes import wintypes

from sohayok.lang import to_bangla_digits
from sohayok.tools.registry import Risk, boolean, integer, string, tool

NO_WINDOW = 0x08000000
STEP = 20
SOFTWARE_FLOOR = 50  # Windows rejects gamma ramps darker than this (45% failed here)
_software = {"level": 100}  # the gamma-ramp level last set by Sohayok
PERSONALIZE = r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize"


def _powershell(command: str) -> str:
    done = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        capture_output=True,
        text=True,
        timeout=15,
        creationflags=NO_WINDOW,
    )
    return done.stdout.strip() if done.returncode == 0 else ""


def _wmi_brightness() -> int | None:
    out = _powershell(
        "(Get-CimInstance -Namespace root/WMI -ClassName WmiMonitorBrightness "
        "-ErrorAction Stop).CurrentBrightness"
    )
    return int(out.split()[0]) if out[:1].isdigit() else None


def _set_wmi(level: int) -> bool:
    out = _powershell(
        "Get-CimInstance -Namespace root/WMI -ClassName WmiMonitorBrightnessMethods "
        f"-ErrorAction Stop | Invoke-CimMethod -MethodName WmiSetBrightness "
        f"-Arguments @{{Timeout=1; Brightness={level}}} | Out-Null; 'ok'"
    )
    return out == "ok"


class _Physical(ctypes.Structure):
    _fields_ = [("handle", wintypes.HANDLE), ("description", wintypes.WCHAR * 128)]


def _ddc(level: int | None = None) -> int | None:
    """Read (or with `level`, set) brightness on monitors that answer DDC/CI. Returns the
    current level in percent, or None when no monitor answers."""
    dxva2, user32 = ctypes.WinDLL("dxva2"), ctypes.WinDLL("user32")
    handles: list = []
    callback = ctypes.WINFUNCTYPE(
        wintypes.BOOL,
        wintypes.HMONITOR,
        wintypes.HDC,
        ctypes.POINTER(wintypes.RECT),
        wintypes.LPARAM,
    )(lambda monitor, dc, rect, data: handles.append(monitor) or True)
    user32.EnumDisplayMonitors(None, None, callback, 0)
    found = None
    for monitor in handles:
        count = wintypes.DWORD()
        if not dxva2.GetNumberOfPhysicalMonitorsFromHMONITOR(monitor, ctypes.byref(count)):
            continue
        physical = (_Physical * count.value)()
        if not dxva2.GetPhysicalMonitorsFromHMONITOR(monitor, count, physical):
            continue
        try:
            for screen in physical:
                kind, current, maximum = wintypes.DWORD(), wintypes.DWORD(), wintypes.DWORD()
                if (
                    not dxva2.GetVCPFeatureAndVCPFeatureReply(
                        screen.handle,
                        0x10,
                        ctypes.byref(kind),
                        ctypes.byref(current),
                        ctypes.byref(maximum),
                    )
                    or not maximum.value
                ):
                    continue
                if level is not None:
                    dxva2.SetVCPFeature(screen.handle, 0x10, round(level * maximum.value / 100))
                    found = level
                else:
                    found = round(current.value * 100 / maximum.value)
        finally:
            dxva2.DestroyPhysicalMonitors(count, physical)
    return found


def _set_gamma(level: int) -> bool:
    user32, gdi32 = ctypes.WinDLL("user32"), ctypes.WinDLL("gdi32")
    factor = max(SOFTWARE_FLOOR, min(100, level)) / 100
    ramp = (wintypes.WORD * 768)()
    for i in range(256):
        value = min(65535, int(i * 257 * factor))
        ramp[i] = ramp[256 + i] = ramp[512 + i] = value
    dc = user32.GetDC(None)
    try:
        return bool(gdi32.SetDeviceGammaRamp(dc, ctypes.byref(ramp)))
    finally:
        user32.ReleaseDC(None, dc)


def _target(action: str, level: int | None, current: int) -> int:
    if action == "up":
        return min(100, current + (level or STEP))
    if action == "down":
        return max(0, current - (level or STEP))
    return max(0, min(100, level if level is not None else current))


@tool(
    "Set, raise, or lower the screen brightness, or report it. Works on laptop screens and "
    "monitors; on monitors without brightness control it dims the picture instead.",
    {
        "action": string("What to do", ["set", "up", "down", "get"]),
        "level": integer("Percent for 'set' (0-100), or how much to change for up/down"),
    },
    required=["action"],
)
def brightness(action: str = "set", level: int | None = None) -> dict:
    action = (action or "set").lower()
    current = _wmi_brightness()
    how = "screen"
    if current is None:
        current, how = _ddc(), "monitor"
    if current is None:
        current, how = _software["level"], "software"
    if action == "get":
        return {
            "brightness": current,
            "spoken_bn": f"ব্রাইটনেস এখন {to_bangla_digits(current)} শতাংশ।",
            "spoken_en": f"Brightness is at {current} percent.",
        }
    target = _target(action, level, current)
    if how == "screen":
        ok = _set_wmi(target)
    elif how == "monitor":
        ok = _ddc(target) is not None
    else:
        target = max(SOFTWARE_FLOOR, target)
        ok = _set_gamma(target)
        if ok:
            _software["level"] = target
    if not ok:
        return {
            "error": "could not change the brightness",
            "spoken_bn": "ব্রাইটনেস বদলাতে পারলাম না। মনিটরের বোতাম দিয়ে বদলাতে হবে।",
            "spoken_en": "I couldn't change the brightness; use the monitor's buttons.",
        }
    note_bn = note_en = ""
    if how == "software" and action != "down" and target == current == 100:
        note_bn = " এই মনিটরে এর বেশি উজ্জ্বল করা যায় না।"
        note_en = " This monitor can't go brighter from here."
    return {
        "brightness": target,
        "method": how,
        "spoken_bn": f"ব্রাইটনেস {to_bangla_digits(target)} শতাংশ করলাম।{note_bn}",
        "spoken_en": f"Brightness set to {target} percent.{note_en}",
    }


@tool(
    "Turn Windows dark mode on or off (apps and the taskbar).",
    {"on": boolean("True for dark mode, false for light mode")},
    risk=Risk.SAFE,
)
def dark_mode(on: bool = True) -> dict:
    light = 0 if on else 1
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, PERSONALIZE, 0, winreg.KEY_SET_VALUE) as key:
        winreg.SetValueEx(key, "AppsUseLightTheme", 0, winreg.REG_DWORD, light)
        winreg.SetValueEx(key, "SystemUsesLightTheme", 0, winreg.REG_DWORD, light)
    # Tell open windows the theme changed, as the Settings app does.
    ctypes.windll.user32.SendMessageTimeoutW(
        0xFFFF, 0x001A, 0, "ImmersiveColorSet", 0x0002, 2000, None
    )
    return {
        "dark": on,
        "spoken_bn": "ডার্ক মোড চালু করলাম।" if on else "লাইট মোড চালু করলাম।",
        "spoken_en": "Dark mode is on." if on else "Light mode is on.",
    }
